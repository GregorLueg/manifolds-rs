//! Device-resident three-kernel FFT t-SNE optimiser.
//!
//! The GPU twin of `optimise_fft3k_tsne`. Positions, momentum, gains and the
//! affinity graph live on the device; each epoch is a chain of launches with
//! no readback:
//!
//! 1. bucket points by grid box (`u32` atomics, then a scan and a scatter)
//! 2. splat a unit charge onto the zero-padded grid, one unit per node
//! 3. forward 2D FFT, multiply by the three kernel spectra (`q`, `q^2 dx`,
//!    `q^2 dy`), inverse 2D FFT with two real outputs packed per complex grid
//! 4. gather `Z_i` and the repulsive force per point
//! 5. attraction over the CSR graph fused with the gains / momentum update
//! 6. recentre
//!
//! The grid geometry is the CPU path's (`fft_grid_geometry`), zero-padded to
//! the next power of two, so both backends give comparable embeddings. The
//! embedding extent is read back every `GRID_CHECK_INTERVAL` epochs, and the
//! grid (with its kernel spectra, rebuilt on the device) is sized for the
//! extent expected at the next check, so points do not outrun it in between.
//!
//! Summation order inside a box follows the atomic bucketing, so results are
//! reproducible structurally but not bitwise.

#![allow(missing_docs)] // cubecl weirdness

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::data::graph::coo_to_adjacency_list;
use crate::data::structures::*;
use crate::prelude::*;
use crate::training::tsne_optimiser::*;
use crate::utils::fft2d_gpu::{Fft2dPlan, FFT2D_MIN_N};

////////////
// Consts //
////////////

/// Preferred workgroup width for the per-point and per-node kernels, before
/// device clamping. Rounded down to a power of two for the tree reductions.
const TSNE_GPU_WORKGROUP_SIZE: u32 = 256;

/// Epochs between extent readbacks.
const GRID_CHECK_INTERVAL: usize = 10;

/// Multiple of the extent growth over the last check interval that the grid
/// is sized ahead for. Sizing only for the current extent (the CPU rule, which
/// checks every epoch) let points outrun the grid between checks and
/// measurably shrank the final embedding.
const GRID_GROWTH_SAFETY: f64 = 1.5;

/////////////
// Helpers //
/////////////

/// Lagrange basis polynomial `k` on `ni` equispaced nodes in `[0, 1]`
/// (node `j` at `(j + 0.5) / ni`), evaluated at `t`.
///
/// ### Params
///
/// * `t` - Position inside the box, in `[0, 1]`
/// * `k` - Basis index
/// * `ni` - Interpolation nodes per box; comptime
///
/// ### Returns
///
/// The weight of node `k`.
#[cube]
fn lagrange_basis<F: Float + CubeElement>(t: F, k: u32, #[comptime] ni: u32) -> F {
    let inv = F::new(1.0_f32) / F::cast_from(ni);
    let half = F::new(0.5_f32);
    let sk = (F::cast_from(k) + half) * inv;
    let mut num = F::new(1.0_f32);
    let mut den = F::new(1.0_f32);
    #[unroll]
    for j in 0..ni {
        if j != k {
            let sj = (F::cast_from(j) + half) * inv;
            num *= t - sj;
            den *= sk - sj;
        }
    }
    num / den
}

/// Tree sum over a workgroup's shared buffer. All units must call it.
///
/// ### Params
///
/// * `sh` - Shared buffer of `wg` values, reduced in place into `sh[0]`
/// * `u` - Unit index
/// * `log2_wg` - `log2(wg)`; comptime, `wg` a power of two
#[cube]
fn workgroup_sum<F: Float + CubeElement>(
    sh: &mut SharedMemory<F>,
    u: u32,
    #[comptime] log2_wg: u32,
) {
    sync_cube();
    #[unroll]
    for k in 0..log2_wg {
        let s = (1u32 << log2_wg) >> (k + 1);
        if u < s {
            let v = sh[(u + s) as usize];
            sh[u as usize] += v;
        }
        sync_cube();
    }
}

/// Tree max over a workgroup's shared buffer. All units must call it.
///
/// ### Params
///
/// * `sh` - Shared buffer of `wg` values, reduced in place into `sh[0]`
/// * `u` - Unit index
/// * `log2_wg` - `log2(wg)`; comptime, `wg` a power of two
#[cube]
fn workgroup_max<F: Float + CubeElement>(
    sh: &mut SharedMemory<F>,
    u: u32,
    #[comptime] log2_wg: u32,
) {
    sync_cube();
    #[unroll]
    for k in 0..log2_wg {
        let s = (1u32 << log2_wg) >> (k + 1);
        if u < s {
            let a = sh[u as usize];
            let b = sh[(u + s) as usize];
            sh[u as usize] = F::max(a, b);
        }
        sync_cube();
    }
}

/////////////
// Kernels //
/////////////

/// Box assignment, Lagrange weights and in-box slot for every point.
///
/// ### Params
///
/// * `pos` - Interleaved positions `[2n]`
/// * `box_id` - Output box index `by * nb + bx` `[n]`
/// * `slot` - Output rank of the point inside its box `[n]`
/// * `wx` - Output x weights `[n * ni]`
/// * `wy` - Output y weights `[n * ni]`
/// * `box_count` - Per-box counters `[nb^2]`, zero on entry
/// * `n` - Number of points
/// * `nb` - Boxes per dimension
/// * `coord_min` - Lower grid edge
/// * `box_width` - Box width
/// * `ni` - Interpolation nodes per box; comptime
/// * `wg` - Workgroup width; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_prep<F: Float + CubeElement>(
    pos: &Tensor<F>,
    box_id: &mut Tensor<u32>,
    slot: &mut Tensor<u32>,
    wx: &mut Tensor<F>,
    wy: &mut Tensor<F>,
    box_count: &Tensor<Atomic<u32>>,
    n: u32,
    nb: u32,
    coord_min: F,
    box_width: F,
    #[comptime] ni: u32,
    #[comptime] wg: u32,
) {
    let i = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + UNIT_POS_X;
    if i >= n {
        terminate!();
    }
    let top = F::cast_from(nb - 1u32);
    let zero = F::new(0.0_f32);

    let rx = (pos[(2u32 * i) as usize] - coord_min) / box_width;
    let ry = (pos[(2u32 * i + 1u32) as usize] - coord_min) / box_width;
    let fx = F::max(F::min(F::floor(rx), top), zero);
    let fy = F::max(F::min(F::floor(ry), top), zero);
    // clamped: a point past the grid edge between extent checks snaps to the
    // edge instead of extrapolating the interpolation weights
    let one = F::new(1.0_f32);
    let tx = F::max(F::min(rx - fx, one), zero);
    let ty = F::max(F::min(ry - fy, one), zero);

    #[unroll]
    for k in 0..ni {
        wx[(i * ni + k) as usize] = lagrange_basis::<F>(tx, k, ni);
        wy[(i * ni + k) as usize] = lagrange_basis::<F>(ty, k, ni);
    }

    let b = u32::cast_from(fy) * nb + u32::cast_from(fx);
    box_id[i as usize] = b;
    slot[i as usize] = box_count[b as usize].fetch_add(1u32);
}

/// Exclusive scan of the box counts into box starts, resetting the counts.
///
/// Runs as a single workgroup; each unit owns a contiguous chunk.
///
/// ### Params
///
/// * `box_count` - Per-box counters `[len]`, zeroed on exit
/// * `box_start` - Output exclusive prefix `[len + 1]`
/// * `len` - Number of boxes
/// * `wg` - Workgroup width; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_scan(
    box_count: &Tensor<Atomic<u32>>,
    box_start: &mut Tensor<u32>,
    len: u32,
    #[comptime] wg: u32,
) {
    let mut sh = SharedMemory::<u32>::new(wg as usize);
    let u = UNIT_POS_X;
    // spelled out: `div_ceil` is not part of the cubecl expansion
    #[allow(clippy::manual_div_ceil)]
    let chunk = (len + wg - 1u32) / wg;
    let lo = u * chunk;
    let mut hi = lo + chunk;
    if hi > len {
        hi = len;
    }

    let mut acc: u32 = 0u32;
    let mut b: u32 = lo;
    while b < hi {
        acc += box_count[b as usize].load();
        b += 1u32;
    }
    sh[u as usize] = acc;
    sync_cube();

    if u == 0u32 {
        let mut run: u32 = 0u32;
        for k in 0..wg {
            let v = sh[k as usize];
            sh[k as usize] = run;
            run += v;
        }
        box_start[len as usize] = run;
    }
    sync_cube();

    let mut run: u32 = sh[u as usize];
    let mut b: u32 = lo;
    while b < hi {
        let c = box_count[b as usize].load();
        box_start[b as usize] = run;
        run += c;
        box_count[b as usize].store(0u32);
        b += 1u32;
    }
}

/// Scatter point indices into box order.
///
/// ### Params
///
/// * `box_id` - Box per point `[n]`
/// * `slot` - Rank inside the box `[n]`
/// * `box_start` - Box starts `[nb^2 + 1]`
/// * `order` - Output point indices grouped by box `[n]`
/// * `n` - Number of points
/// * `wg` - Workgroup width; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_scatter(
    box_id: &Tensor<u32>,
    slot: &Tensor<u32>,
    box_start: &Tensor<u32>,
    order: &mut Tensor<u32>,
    n: u32,
    #[comptime] wg: u32,
) {
    let i = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + UNIT_POS_X;
    if i >= n {
        terminate!();
    }
    let b = box_id[i as usize];
    let dst = box_start[b as usize] + slot[i as usize];
    order[dst as usize] = i;
}

/// Unit-charge splat onto the zero-padded FFT grid, one unit per element.
///
/// Node `(gy, gx)` lies in box `(gy / ni, gx / ni)` and sums the weights of
/// that box's points. Elements outside the `m x m` corner are zeroed.
///
/// ### Params
///
/// * `order` - Point indices grouped by box `[n]`
/// * `box_start` - Box starts `[nb^2 + 1]`
/// * `wx` - X weights `[n * ni]`
/// * `wy` - Y weights `[n * ni]`
/// * `g_re` - Output grid, real `[n_fft^2]`
/// * `g_im` - Output grid, imaginary, zeroed `[n_fft^2]`
/// * `nb` - Boxes per dimension
/// * `n_fft` - Grid side; comptime
/// * `ni` - Interpolation nodes per box; comptime
/// * `wg` - Workgroup width; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_splat<F: Float + CubeElement>(
    order: &Tensor<u32>,
    box_start: &Tensor<u32>,
    wx: &Tensor<F>,
    wy: &Tensor<F>,
    g_re: &mut Tensor<F>,
    g_im: &mut Tensor<F>,
    nb: u32,
    #[comptime] n_fft: u32,
    #[comptime] ni: u32,
    #[comptime] wg: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + UNIT_POS_X;
    if idx >= n_fft * n_fft {
        terminate!();
    }
    let gy = idx / n_fft;
    let gx = idx % n_fft;
    let m = nb * ni;

    let mut acc = F::new(0.0_f32);
    if gy < m && gx < m {
        let b = (gy / ni) * nb + gx / ni;
        let iy = gy % ni;
        let ix = gx % ni;
        let mut p = box_start[b as usize];
        let end = box_start[(b + 1u32) as usize];
        while p < end {
            let pt = order[p as usize];
            acc += wy[(pt * ni + iy) as usize] * wx[(pt * ni + ix) as usize];
            p += 1u32;
        }
    }
    g_re[idx as usize] = acc;
    g_im[idx as usize] = F::new(0.0_f32);
}

/// Multiply the charge spectrum by the three kernel spectra, packing the two
/// x-convolution outputs into one complex grid.
///
/// Every product has a real inverse, so `W K_1 + i W K_x` inverts to
/// `Z + i F_x` and `W K_y` to `F_y`.
///
/// ### Params
///
/// * `g_re` - Charge spectrum, real `[n_fft^2]`
/// * `g_im` - Charge spectrum, imaginary
/// * `k_re` - Kernel spectra, real `[3 * n_fft^2]`
/// * `k_im` - Kernel spectra, imaginary
/// * `p_re` - Output products, real `[2 * n_fft^2]`
/// * `p_im` - Output products, imaginary
/// * `nn` - `n_fft^2`; comptime
/// * `wg` - Workgroup width; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_hadamard<F: Float + CubeElement>(
    g_re: &Tensor<F>,
    g_im: &Tensor<F>,
    k_re: &Tensor<F>,
    k_im: &Tensor<F>,
    p_re: &mut Tensor<F>,
    p_im: &mut Tensor<F>,
    #[comptime] nn: u32,
    #[comptime] wg: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + UNIT_POS_X;
    if idx >= nn {
        terminate!();
    }
    let wr = g_re[idx as usize];
    let wi = g_im[idx as usize];

    let k1r = k_re[idx as usize];
    let k1i = k_im[idx as usize];
    let kxr = k_re[(nn + idx) as usize];
    let kxi = k_im[(nn + idx) as usize];
    let kyr = k_re[(2u32 * nn + idx) as usize];
    let kyi = k_im[(2u32 * nn + idx) as usize];

    let ar = wr * k1r - wi * k1i;
    let ai = wr * k1i + wi * k1r;
    let br = wr * kxr - wi * kxi;
    let bi = wr * kxi + wi * kxr;

    // A + i B
    p_re[idx as usize] = ar - bi;
    p_im[idx as usize] = ai + br;
    p_re[(nn + idx) as usize] = wr * kyr - wi * kyi;
    p_im[(nn + idx) as usize] = wr * kyi + wi * kyr;
}

/// Gather `Z_i` and the repulsive force per point from the convolved grids.
///
/// Writes the raw force and a per-workgroup partial of `sum_i (Z_i - 1)`.
///
/// ### Params
///
/// * `p_re` - Inverse products, real `[2 * n_fft^2]`
/// * `p_im` - Inverse products, imaginary
/// * `box_id` - Box per point `[n]`
/// * `wx` - X weights `[n * ni]`
/// * `wy` - Y weights `[n * ni]`
/// * `rep` - Output raw repulsion `sum_j q^2 (y_i - y_j)` `[2n]`
/// * `z_partial` - Output per-cube partial of `Z` `[n_cubes]`
/// * `n` - Number of points
/// * `nb` - Boxes per dimension
/// * `n_fft` - Grid side; comptime
/// * `ni` - Interpolation nodes per box; comptime
/// * `wg` - Workgroup width; comptime
/// * `log2_wg` - `log2(wg)`; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_gather<F: Float + CubeElement>(
    p_re: &Tensor<F>,
    p_im: &Tensor<F>,
    box_id: &Tensor<u32>,
    wx: &Tensor<F>,
    wy: &Tensor<F>,
    rep: &mut Tensor<F>,
    z_partial: &mut Tensor<F>,
    n: u32,
    nb: u32,
    #[comptime] n_fft: u32,
    #[comptime] ni: u32,
    #[comptime] wg: u32,
    #[comptime] log2_wg: u32,
) {
    let mut sh = SharedMemory::<F>::new(wg as usize);
    let cube = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let i = cube * wg + u;

    let mut z_contrib = F::new(0.0_f32);
    if i < n {
        let nn = n_fft * n_fft;
        let m = nb * ni;
        let b = box_id[i as usize];
        let by = b / nb;
        let bx = b % nb;
        let mut z = F::new(0.0_f32);
        let mut fx = F::new(0.0_f32);
        let mut fy = F::new(0.0_f32);
        #[unroll]
        for iy in 0..ni {
            let w_y = wy[(i * ni + iy) as usize];
            let row = (m + by * ni + iy) * n_fft + m + bx * ni;
            #[unroll]
            for ix in 0..ni {
                let w = w_y * wx[(i * ni + ix) as usize];
                let o = row + ix;
                z += w * p_re[o as usize];
                fx += w * p_im[o as usize];
                fy += w * p_re[(nn + o) as usize];
            }
        }
        rep[(2u32 * i) as usize] = fx;
        rep[(2u32 * i + 1u32) as usize] = fy;
        z_contrib = z - F::new(1.0_f32);
    }
    sh[u as usize] = z_contrib;
    workgroup_sum::<F>(&mut sh, u, log2_wg);
    if u == 0u32 {
        z_partial[cube as usize] = sh[0usize];
    }
}

/// Attraction over the CSR graph fused with the gains / momentum step.
///
/// Every workgroup first reduces the `Z` partials itself, which saves a
/// launch. Positions are ping-ponged because the attraction reads neighbours.
/// Also writes a per-workgroup partial of the new positions for the
/// recentring.
///
/// ### Params
///
/// * `pos_in` - Positions before the step `[2n]`
/// * `pos_out` - Positions after the step `[2n]`
/// * `vel` - Momentum buffer `[2n]`, updated
/// * `gains` - Adaptive gains `[2n]`, updated
/// * `rep` - Raw repulsion `[2n]`
/// * `z_partial` - Partials of `Z` `[n_zp]`
/// * `indptr` - CSR row pointers `[n + 1]`
/// * `indices` - CSR neighbours `[nnz]`
/// * `values` - CSR affinities `[nnz]`
/// * `mean_partial` - Output per-cube position sums `[2 * n_cubes]`
/// * `n` - Number of points
/// * `n_zp` - Number of `Z` partials
/// * `exag` - Exaggeration factor
/// * `lr` - Learning rate
/// * `momentum` - Momentum
/// * `min_gain` - Gain floor
/// * `max_step` - Step norm cap
/// * `z_floor` - `Z` below this is replaced by 1
/// * `wg` - Workgroup width; comptime
/// * `log2_wg` - `log2(wg)`; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_update<F: Float + CubeElement>(
    pos_in: &Tensor<F>,
    pos_out: &mut Tensor<F>,
    vel: &mut Tensor<F>,
    gains: &mut Tensor<F>,
    rep: &Tensor<F>,
    z_partial: &Tensor<F>,
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    values: &Tensor<F>,
    mean_partial: &mut Tensor<F>,
    n: u32,
    n_zp: u32,
    exag: F,
    lr: F,
    momentum: F,
    min_gain: F,
    max_step: F,
    z_floor: F,
    #[comptime] wg: u32,
    #[comptime] log2_wg: u32,
) {
    let mut sh = SharedMemory::<F>::new(wg as usize);
    let mut sh2 = SharedMemory::<F>::new(wg as usize);
    let cube = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let i = cube * wg + u;
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);

    let mut zacc = zero;
    let mut k = u;
    while k < n_zp {
        zacc += z_partial[k as usize];
        k += wg;
    }
    sh[u as usize] = zacc;
    workgroup_sum::<F>(&mut sh, u, log2_wg);
    let z_raw = sh[0usize];
    let z = if z_raw > z_floor { z_raw } else { one };
    sync_cube();

    let mut new_x = zero;
    let mut new_y = zero;
    if i < n {
        let px = pos_in[(2u32 * i) as usize];
        let py = pos_in[(2u32 * i + 1u32) as usize];

        // 4-way split accumulators keep several neighbour gathers in flight
        let mut ax0 = zero;
        let mut ay0 = zero;
        let mut ax1 = zero;
        let mut ay1 = zero;
        let start = indptr[i as usize];
        let end = indptr[(i + 1u32) as usize];
        let mut e = start;
        while e + 1u32 < end {
            let j0 = indices[e as usize];
            let j1 = indices[(e + 1u32) as usize];
            let dx0 = px - pos_in[(2u32 * j0) as usize];
            let dy0 = py - pos_in[(2u32 * j0 + 1u32) as usize];
            let dx1 = px - pos_in[(2u32 * j1) as usize];
            let dy1 = py - pos_in[(2u32 * j1 + 1u32) as usize];
            let f0 = values[e as usize] / (one + dx0 * dx0 + dy0 * dy0);
            let f1 = values[(e + 1u32) as usize] / (one + dx1 * dx1 + dy1 * dy1);
            ax0 += f0 * dx0;
            ay0 += f0 * dy0;
            ax1 += f1 * dx1;
            ay1 += f1 * dy1;
            e += 2u32;
        }
        if e < end {
            let j = indices[e as usize];
            let dx = px - pos_in[(2u32 * j) as usize];
            let dy = py - pos_in[(2u32 * j + 1u32) as usize];
            let f = values[e as usize] / (one + dx * dx + dy * dy);
            ax0 += f * dx;
            ay0 += f * dy;
        }

        let gx = exag * (ax0 + ax1) - rep[(2u32 * i) as usize] / z;
        let gy = exag * (ay0 + ay1) - rep[(2u32 * i + 1u32) as usize] / z;

        let ix = (2u32 * i) as usize;
        let iy = (2u32 * i + 1u32) as usize;
        let mut ux = vel[ix];
        let mut uy = vel[iy];
        let mut g_x = gains[ix];
        let mut g_y = gains[iy];

        // Jacobs gains: grow when gradient and update disagree in sign
        if (gx > zero) != (ux > zero) {
            g_x += F::new(0.2_f32);
        } else {
            g_x *= F::new(0.8_f32);
        }
        if (gy > zero) != (uy > zero) {
            g_y += F::new(0.2_f32);
        } else {
            g_y *= F::new(0.8_f32);
        }
        g_x = F::max(g_x, min_gain);
        g_y = F::max(g_y, min_gain);

        ux = momentum * ux - lr * g_x * gx;
        uy = momentum * uy - lr * g_y * gy;
        let step_sq = ux * ux + uy * uy;
        if step_sq > max_step * max_step {
            let scale = max_step / F::sqrt(step_sq);
            ux *= scale;
            uy *= scale;
        }

        vel[ix] = ux;
        vel[iy] = uy;
        gains[ix] = g_x;
        gains[iy] = g_y;
        new_x = px + ux;
        new_y = py + uy;
        pos_out[ix] = new_x;
        pos_out[iy] = new_y;
    }

    sh[u as usize] = new_x;
    sh2[u as usize] = new_y;
    workgroup_sum::<F>(&mut sh, u, log2_wg);
    workgroup_sum::<F>(&mut sh2, u, log2_wg);
    if u == 0u32 {
        mean_partial[(2u32 * cube) as usize] = sh[0usize];
        mean_partial[(2u32 * cube + 1u32) as usize] = sh2[0usize];
    }
}

/// Recentre the embedding on the origin and record the extent.
///
/// Every workgroup reduces the position partials itself, then shifts its own
/// points and writes the largest absolute coordinate it holds.
///
/// ### Params
///
/// * `pos` - Positions `[2n]`, shifted in place
/// * `mean_partial` - Per-cube position sums `[2 * n_mp]`
/// * `extent_partial` - Output per-cube max absolute coordinate `[n_cubes]`
/// * `n` - Number of points
/// * `n_mp` - Number of position partials
/// * `inv_n` - `1 / n`
/// * `wg` - Workgroup width; comptime
/// * `log2_wg` - `log2(wg)`; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_recentre<F: Float + CubeElement>(
    pos: &mut Tensor<F>,
    mean_partial: &Tensor<F>,
    extent_partial: &mut Tensor<F>,
    n: u32,
    n_mp: u32,
    inv_n: F,
    #[comptime] wg: u32,
    #[comptime] log2_wg: u32,
) {
    let mut sh = SharedMemory::<F>::new(wg as usize);
    let mut sh2 = SharedMemory::<F>::new(wg as usize);
    let cube = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let i = cube * wg + u;
    let zero = F::new(0.0_f32);

    let mut sx = zero;
    let mut sy = zero;
    let mut k = u;
    while k < n_mp {
        sx += mean_partial[(2u32 * k) as usize];
        sy += mean_partial[(2u32 * k + 1u32) as usize];
        k += wg;
    }
    sh[u as usize] = sx;
    sh2[u as usize] = sy;
    workgroup_sum::<F>(&mut sh, u, log2_wg);
    workgroup_sum::<F>(&mut sh2, u, log2_wg);
    let mx = sh[0usize] * inv_n;
    let my = sh2[0usize] * inv_n;
    sync_cube();

    let mut ext = zero;
    if i < n {
        let x = pos[(2u32 * i) as usize] - mx;
        let y = pos[(2u32 * i + 1u32) as usize] - my;
        pos[(2u32 * i) as usize] = x;
        pos[(2u32 * i + 1u32) as usize] = y;
        ext = F::max(F::abs(x), F::abs(y));
    }
    sh[u as usize] = ext;
    workgroup_max::<F>(&mut sh, u, log2_wg);
    if u == 0u32 {
        extent_partial[cube as usize] = sh[0usize];
    }
}

/// Spatial three-kernel grid for the spectrum rebuild, pre-scaled by
/// `1 / n_fft^2`.
///
/// Element `(r, c)` of kernel `t` holds the kernel at offset
/// `((c - m) h, (r - m) h)` for `1 <= r, c <= 2m - 1`, zero elsewhere.
///
/// ### Params
///
/// * `k_re` - Output `[3 * n_fft^2]`
/// * `k_im` - Output, zeroed
/// * `m` - Grid nodes per dimension
/// * `h` - Node spacing
/// * `scale` - `1 / n_fft^2`
/// * `n_fft` - Grid side; comptime
/// * `wg` - Workgroup width; comptime
#[cube(launch_unchecked)]
pub fn tsne3k_kernel_fill<F: Float + CubeElement>(
    k_re: &mut Tensor<F>,
    k_im: &mut Tensor<F>,
    m: u32,
    h: F,
    scale: F,
    #[comptime] n_fft: u32,
    #[comptime] wg: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg + UNIT_POS_X;
    let nn = n_fft * n_fft;
    if idx >= 3u32 * nn {
        terminate!();
    }
    let t = idx / nn;
    let rem = idx % nn;
    let r = rem / n_fft;
    let c = rem % n_fft;

    let mut v = F::new(0.0_f32);
    if r >= 1u32 && r < 2u32 * m && c >= 1u32 && c < 2u32 * m {
        let dx = (F::cast_from(c) - F::cast_from(m)) * h;
        let dy = (F::cast_from(r) - F::cast_from(m)) * h;
        let q = F::new(1.0_f32) / (F::new(1.0_f32) + dx * dx + dy * dy);
        if t == 0u32 {
            v = q;
        } else if t == 1u32 {
            v = q * q * dx;
        } else {
            v = q * q * dy;
        }
    }
    k_re[idx as usize] = v * scale;
    k_im[idx as usize] = F::new(0.0_f32);
}

//////////////
// Geometry //
//////////////

/// Grid geometry for the GPU path.
#[derive(Clone, Copy, Debug)]
struct GridGeometry {
    /// FFT grid side
    n_fft: usize,
    /// Boxes per dimension
    n_boxes: usize,
    /// Half-width of the grid in embedding units
    half: f64,
}

impl GridGeometry {
    /// Pick the grid for an embedding extent.
    ///
    /// Same box count and half-width as the CPU path; the FFT side is the
    /// next power of two that fits the zero-padded `2 m` nodes.
    ///
    /// ### Params
    ///
    /// * `half_span` - Largest absolute coordinate of the embedding
    /// * `ni` - Interpolation nodes per box
    ///
    /// ### Returns
    ///
    /// The geometry.
    fn for_extent(half_span: f64, ni: usize) -> Self {
        let (n_boxes, _, half) = fft_grid_geometry(half_span, TSNE_FFT_MIN_INTERVALS);
        Self {
            n_fft: (2 * ni * n_boxes).next_power_of_two().max(FFT2D_MIN_N),
            n_boxes,
            half,
        }
    }

    /// Box width in embedding units.
    fn box_width(&self) -> f64 {
        2.0 * self.half / self.n_boxes as f64
    }

    /// Whether the grid must be rebuilt, by the CPU path's rule: the box
    /// count for the new extent changed, or the embedding came within one
    /// box of the edge.
    ///
    /// ### Params
    ///
    /// * `half_span` - Largest absolute coordinate of the embedding
    /// * `ni` - Interpolation nodes per box
    ///
    /// ### Returns
    ///
    /// `true` when the grid must be rebuilt.
    fn needs_rebuild(&self, half_span: f64, ni: usize) -> bool {
        Self::for_extent(half_span, ni).n_boxes != self.n_boxes
            || half_span >= self.half - self.box_width()
    }
}

//////////////////
// Device state //
//////////////////

/// Buffers whose size depends on the FFT grid.
struct GridBuffers<R: Runtime, T: ManifoldsFloatGpu> {
    /// Geometry the buffers were built for
    geom: GridGeometry,
    /// Batched FFT plan for the forward charge transform
    plan_1: Fft2dPlan,
    /// Batched FFT plan for the two packed inverse transforms
    plan_2: Fft2dPlan,
    /// Batched FFT plan for the three kernel spectra
    plan_3: Fft2dPlan,
    /// Per-box counters `[nb^2]`
    box_count: GpuTensor<R, u32>,
    /// Box starts `[nb^2 + 1]`
    box_start: GpuTensor<R, u32>,
    /// Charge grid, real `[n_fft^2]`
    g_re: GpuTensor<R, T>,
    /// Charge grid, imaginary
    g_im: GpuTensor<R, T>,
    /// Products / convolved grids, real `[2 * n_fft^2]`
    p_re: GpuTensor<R, T>,
    /// Products / convolved grids, imaginary
    p_im: GpuTensor<R, T>,
    /// Kernel spectra, real `[3 * n_fft^2]`
    k_re: GpuTensor<R, T>,
    /// Kernel spectra, imaginary
    k_im: GpuTensor<R, T>,
    /// FFT transpose scratch, real `[3 * n_fft^2]`
    tmp_re: GpuTensor<R, T>,
    /// FFT transpose scratch, imaginary
    tmp_im: GpuTensor<R, T>,
    /// One cube per `wg` grid elements
    cubes_grid: CubeCount,
    /// One cube per `wg` elements of the three kernels
    cubes_kernel: CubeCount,
}

/// Plan a 1D-over-2D grid with one unit per element.
///
/// ### Params
///
/// * `kernel` - Kernel name for errors
/// * `total` - Number of elements
/// * `wg` - Workgroup width
/// * `limits` - Device limits
///
/// ### Returns
///
/// The cube count and the number of cubes in it.
fn plan_cubes(
    kernel: &'static str,
    total: usize,
    wg: u32,
    limits: &GpuLimits,
) -> Result<(CubeCount, usize), ManifoldsError> {
    let (gx, gy) = grid_2d((total as u32).div_ceil(wg), limits)?;
    Ok((
        checked_cube_count(kernel, gx, gy, 1, limits)?,
        (gx * gy) as usize,
    ))
}

impl<R: Runtime, T: ManifoldsFloatGpu> GridBuffers<R, T> {
    /// Allocate grid buffers and compute the kernel spectra on the device.
    ///
    /// ### Params
    ///
    /// * `geom` - Grid geometry
    /// * `ni` - Interpolation nodes per box
    /// * `wg` - Workgroup width
    /// * `limits` - Device limits
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// The buffers, spectra ready.
    fn new(
        geom: GridGeometry,
        ni: usize,
        wg: u32,
        limits: &GpuLimits,
        client: &ComputeClient<R>,
    ) -> Result<Self, ManifoldsError> {
        let nf = geom.n_fft;
        let nn = nf * nf;
        let nb2 = geom.n_boxes * geom.n_boxes;
        let (cubes_grid, _) = plan_cubes("tsne3k_splat", nn, wg, limits)?;
        let (cubes_kernel, _) = plan_cubes("tsne3k_kernel_fill", 3 * nn, wg, limits)?;

        let bufs = Self {
            geom,
            plan_1: Fft2dPlan::new(nf, 1, limits)?,
            plan_2: Fft2dPlan::new(nf, 2, limits)?,
            plan_3: Fft2dPlan::new(nf, 3, limits)?,
            box_count: GpuTensor::from_slice(&vec![0u32; nb2], vec![nb2], client)?,
            box_start: GpuTensor::empty(vec![nb2 + 1], client)?,
            g_re: GpuTensor::empty(vec![nn], client)?,
            g_im: GpuTensor::empty(vec![nn], client)?,
            p_re: GpuTensor::empty(vec![2 * nn], client)?,
            p_im: GpuTensor::empty(vec![2 * nn], client)?,
            k_re: GpuTensor::empty(vec![3 * nn], client)?,
            k_im: GpuTensor::empty(vec![3 * nn], client)?,
            tmp_re: GpuTensor::empty(vec![3 * nn], client)?,
            tmp_im: GpuTensor::empty(vec![3 * nn], client)?,
            cubes_grid,
            cubes_kernel,
        };

        let m = geom.n_boxes * ni;
        let h = geom.box_width() / ni as f64;
        unsafe {
            tsne3k_kernel_fill::launch_unchecked::<T, R>(
                client,
                bufs.cubes_kernel.clone(),
                CubeDim::new_1d(wg),
                bufs.k_re.into_tensor_arg(),
                bufs.k_im.into_tensor_arg(),
                m as u32,
                T::from_f64(h).unwrap(),
                T::from_f64(1.0 / nn as f64).unwrap(),
                nf as u32,
                wg,
            );
        }
        bufs.plan_3.execute(
            client,
            &bufs.k_re,
            &bufs.k_im,
            &bufs.tmp_re,
            &bufs.tmp_im,
            true,
        );

        Ok(bufs)
    }
}

/// Grid-independent device state.
struct TsneGpuState<R: Runtime, T: ManifoldsFloatGpu> {
    /// Positions, current `[2n]`
    pos_a: GpuTensor<R, T>,
    /// Positions, next `[2n]`
    pos_b: GpuTensor<R, T>,
    /// Momentum `[2n]`
    vel: GpuTensor<R, T>,
    /// Gains `[2n]`
    gains: GpuTensor<R, T>,
    /// CSR row pointers `[n + 1]`
    indptr: GpuTensor<R, u32>,
    /// CSR neighbours `[nnz]`
    indices: GpuTensor<R, u32>,
    /// CSR affinities `[nnz]`
    values: GpuTensor<R, T>,
    /// Box per point `[n]`
    box_id: GpuTensor<R, u32>,
    /// Rank inside the box `[n]`
    slot: GpuTensor<R, u32>,
    /// Points grouped by box `[n]`
    order: GpuTensor<R, u32>,
    /// X weights `[n * ni]`
    wx: GpuTensor<R, T>,
    /// Y weights `[n * ni]`
    wy: GpuTensor<R, T>,
    /// Raw repulsion `[2n]`
    rep: GpuTensor<R, T>,
    /// Partials of `Z` `[n_cubes]`
    z_partial: GpuTensor<R, T>,
    /// Partials of the position sums `[2 * n_cubes]`
    mean_partial: GpuTensor<R, T>,
    /// Partials of the extent `[n_cubes]`
    extent_partial: GpuTensor<R, T>,
    /// Number of points
    n: usize,
    /// Interpolation nodes per box
    ni: usize,
    /// Workgroup width, a power of two
    wg: u32,
    /// `log2(wg)`
    log2_wg: u32,
    /// One cube per `wg` points
    cubes_pts: CubeCount,
    /// Number of cubes in `cubes_pts`
    n_cubes_pts: usize,
}

impl<R: Runtime, T: ManifoldsFloatGpu> TsneGpuState<R, T> {
    /// Upload positions and the affinity graph (as CSR) and allocate the
    /// per-point scratch.
    ///
    /// ### Params
    ///
    /// * `embd` - Embedding `[n][2]`
    /// * `graph` - Symmetric t-SNE affinities over the same `n` points
    /// * `ni` - Interpolation nodes per box
    /// * `limits` - Device limits
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// The state and the largest absolute coordinate of `embd`.
    fn upload(
        embd: &[Vec<T>],
        graph: &CoordinateList<T>,
        ni: usize,
        limits: &GpuLimits,
        client: &ComputeClient<R>,
    ) -> Result<(Self, f64), ManifoldsError> {
        let n = embd.len();
        if graph.n_samples != n {
            return Err(ManifoldsError::GraphSizeMismatch {
                n_graph: graph.n_samples,
                n_embd: n,
            });
        }

        // power of two for the tree reductions
        let wg = {
            let w = resolve_workgroup_size(TSNE_GPU_WORKGROUP_SIZE, limits);
            1u32 << (31 - w.leading_zeros())
        };

        let adj = coo_to_adjacency_list(graph);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u32);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in &adj {
            for &(j, w) in row {
                indices.push(j as u32);
                values.push(w);
            }
            indptr.push(indices.len() as u32);
        }
        // keep bindings valid on an edgeless graph
        if indices.is_empty() {
            indices.push(0);
            values.push(T::zero());
        }
        let nnz = indices.len();

        let mut pos_flat = Vec::with_capacity(2 * n);
        let mut half_span = 0.0f64;
        for p in embd {
            pos_flat.push(p[0]);
            pos_flat.push(p[1]);
            half_span = half_span
                .max(p[0].to_f64().unwrap().abs())
                .max(p[1].to_f64().unwrap().abs());
        }

        let (cubes_pts, n_cubes_pts) = plan_cubes("tsne3k_points", n, wg, limits)?;
        let st = Self {
            pos_a: GpuTensor::from_slice(&pos_flat, vec![2 * n], client)?,
            pos_b: GpuTensor::empty(vec![2 * n], client)?,
            vel: GpuTensor::from_slice(&vec![T::zero(); 2 * n], vec![2 * n], client)?,
            gains: GpuTensor::from_slice(&vec![T::one(); 2 * n], vec![2 * n], client)?,
            indptr: GpuTensor::from_slice(&indptr, vec![n + 1], client)?,
            indices: GpuTensor::from_slice(&indices, vec![nnz], client)?,
            values: GpuTensor::from_slice(&values, vec![nnz], client)?,
            box_id: GpuTensor::empty(vec![n], client)?,
            slot: GpuTensor::empty(vec![n], client)?,
            order: GpuTensor::empty(vec![n], client)?,
            wx: GpuTensor::empty(vec![n * ni], client)?,
            wy: GpuTensor::empty(vec![n * ni], client)?,
            rep: GpuTensor::empty(vec![2 * n], client)?,
            z_partial: GpuTensor::empty(vec![n_cubes_pts], client)?,
            mean_partial: GpuTensor::empty(vec![2 * n_cubes_pts], client)?,
            extent_partial: GpuTensor::empty(vec![n_cubes_pts], client)?,
            n,
            ni,
            wg,
            log2_wg: wg.trailing_zeros(),
            cubes_pts,
            n_cubes_pts,
        };
        Ok((st, half_span))
    }

    /// Enqueue the repulsion for the positions in `pos_a`: bucketing, splat,
    /// convolution and gather. Leaves the raw forces in `rep` and the `Z`
    /// partials in `z_partial`.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `grid` - Grid buffers and spectra
    fn enqueue_repulsion(&self, client: &ComputeClient<R>, grid: &GridBuffers<R, T>) {
        let g = &grid.geom;
        let nb = g.n_boxes as u32;
        let nf = g.n_fft as u32;
        let ni = self.ni as u32;
        let wg = self.wg;
        let dim = CubeDim::new_1d(wg);

        unsafe {
            tsne3k_prep::launch_unchecked::<T, R>(
                client,
                self.cubes_pts.clone(),
                dim,
                self.pos_a.into_tensor_arg(),
                self.box_id.into_tensor_arg(),
                self.slot.into_tensor_arg(),
                self.wx.into_tensor_arg(),
                self.wy.into_tensor_arg(),
                grid.box_count.into_tensor_arg(),
                self.n as u32,
                nb,
                T::from_f64(-g.half).unwrap(),
                T::from_f64(g.box_width()).unwrap(),
                ni,
                wg,
            );
            tsne3k_scan::launch_unchecked::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                dim,
                grid.box_count.into_tensor_arg(),
                grid.box_start.into_tensor_arg(),
                nb * nb,
                wg,
            );
            tsne3k_scatter::launch_unchecked::<R>(
                client,
                self.cubes_pts.clone(),
                dim,
                self.box_id.into_tensor_arg(),
                self.slot.into_tensor_arg(),
                grid.box_start.into_tensor_arg(),
                self.order.into_tensor_arg(),
                self.n as u32,
                wg,
            );
            tsne3k_splat::launch_unchecked::<T, R>(
                client,
                grid.cubes_grid.clone(),
                dim,
                self.order.into_tensor_arg(),
                grid.box_start.into_tensor_arg(),
                self.wx.into_tensor_arg(),
                self.wy.into_tensor_arg(),
                grid.g_re.into_tensor_arg(),
                grid.g_im.into_tensor_arg(),
                nb,
                nf,
                ni,
                wg,
            );
        }

        grid.plan_1.execute(
            client,
            &grid.g_re,
            &grid.g_im,
            &grid.tmp_re,
            &grid.tmp_im,
            true,
        );

        unsafe {
            tsne3k_hadamard::launch_unchecked::<T, R>(
                client,
                grid.cubes_grid.clone(),
                dim,
                grid.g_re.into_tensor_arg(),
                grid.g_im.into_tensor_arg(),
                grid.k_re.into_tensor_arg(),
                grid.k_im.into_tensor_arg(),
                grid.p_re.into_tensor_arg(),
                grid.p_im.into_tensor_arg(),
                nf * nf,
                wg,
            );
        }

        grid.plan_2.execute(
            client,
            &grid.p_re,
            &grid.p_im,
            &grid.tmp_re,
            &grid.tmp_im,
            false,
        );

        unsafe {
            tsne3k_gather::launch_unchecked::<T, R>(
                client,
                self.cubes_pts.clone(),
                dim,
                grid.p_re.into_tensor_arg(),
                grid.p_im.into_tensor_arg(),
                self.box_id.into_tensor_arg(),
                self.wx.into_tensor_arg(),
                self.wy.into_tensor_arg(),
                self.rep.into_tensor_arg(),
                self.z_partial.into_tensor_arg(),
                self.n as u32,
                nb,
                nf,
                ni,
                wg,
                self.log2_wg,
            );
        }
    }

    /// Enqueue the attraction, the gains / momentum step from `pos_a` into
    /// `pos_b`, and the recentring of `pos_b`. The caller swaps the buffers.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `sched` - Per-epoch scalars
    fn enqueue_step(&self, client: &ComputeClient<R>, sched: &StepScalars<T>) {
        let wg = self.wg;
        let dim = CubeDim::new_1d(wg);
        unsafe {
            tsne3k_update::launch_unchecked::<T, R>(
                client,
                self.cubes_pts.clone(),
                dim,
                self.pos_a.into_tensor_arg(),
                self.pos_b.into_tensor_arg(),
                self.vel.into_tensor_arg(),
                self.gains.into_tensor_arg(),
                self.rep.into_tensor_arg(),
                self.z_partial.into_tensor_arg(),
                self.indptr.into_tensor_arg(),
                self.indices.into_tensor_arg(),
                self.values.into_tensor_arg(),
                self.mean_partial.into_tensor_arg(),
                self.n as u32,
                self.n_cubes_pts as u32,
                sched.exag,
                sched.lr,
                sched.momentum,
                sched.min_gain,
                sched.max_step,
                sched.z_floor,
                wg,
                self.log2_wg,
            );
            tsne3k_recentre::launch_unchecked::<T, R>(
                client,
                self.cubes_pts.clone(),
                dim,
                self.pos_b.into_tensor_arg(),
                self.mean_partial.into_tensor_arg(),
                self.extent_partial.into_tensor_arg(),
                self.n as u32,
                self.n_cubes_pts as u32,
                T::from_f64(1.0 / self.n as f64).unwrap(),
                wg,
                self.log2_wg,
            );
        }
    }
}

/// Scalars of one optimisation step.
struct StepScalars<T> {
    /// Exaggeration factor
    exag: T,
    /// Learning rate
    lr: T,
    /// Momentum
    momentum: T,
    /// Gain floor
    min_gain: T,
    /// Step norm cap
    max_step: T,
    /// `Z` below this is replaced by 1
    z_floor: T,
}

//////////
// Main //
//////////

/// Optimise a 2D embedding with device-resident three-kernel FFT t-SNE.
///
/// Same schedule and update rule as `optimise_fft3k_tsne` (momentum switch,
/// early / late exaggeration, Jacobs gains, step cap, recentring), with every
/// stage on the GPU, one small extent readback every `GRID_CHECK_INTERVAL`
/// epochs, and the embedding read back once at the end.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][2]`, overwritten with the result
/// * `params` - Optimisation hyperparameters
/// * `graph` - Symmetric t-SNE affinities
/// * `device` - CubeCL device
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `Ok(())` on success; `NoData` if the embedding is empty, `IncorrectDim` if
/// it is not 2D, `GraphSizeMismatch` if the graph covers a different number
/// of points, or a device limit or allocation error.
///
/// ### References
///
/// Linderman et al., Nature Methods, 2019 (FIt-SNE).
pub fn optimise_fft3k_tsne_gpu<R, T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    device: R::Device,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    let verbosity = parse_verbosity_level(verbose);
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let n_dim = embd[0].len();
    if n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim { n_dim });
    }
    let ni = params.n_interp_points;

    let client = R::client(&device);
    let limits = GpuLimits::from_client(&client);
    let (mut st, half_span) = TsneGpuState::<R, T>::upload(embd, graph, ni, &limits, &client)?;
    let mut grid = GridBuffers::<R, T>::new(
        GridGeometry::for_extent(half_span, ni),
        ni,
        st.wg,
        &limits,
        &client,
    )?;

    let lr = params.get_lr(n);
    let mut sched = StepScalars {
        exag: params.early_exag_factor,
        lr,
        momentum: T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap(),
        min_gain: T::from_f64(TSNE_MIN_GAIN).unwrap(),
        max_step: step_cap_from_lr(lr),
        z_floor: T::from_f64(TSNE_EPS).unwrap(),
    };

    if verbosity.normal_verbosity() {
        println!(
            "Running {} epochs of three-kernel FFT t-SNE on the GPU.",
            params.n_epochs
        );
    }

    let mut last_span = half_span;

    for epoch in 0..params.n_epochs {
        if epoch > 0 && epoch % GRID_CHECK_INTERVAL == 0 {
            let ext = st.extent_partial.clone().read(&client)?;
            let span = ext.iter().fold(0.0f64, |a, v| a.max(v.to_f64().unwrap()));
            let ahead = span + GRID_GROWTH_SAFETY * (span - last_span).max(0.0);
            last_span = span;
            if grid.geom.needs_rebuild(ahead, ni) {
                grid = GridBuffers::new(
                    GridGeometry::for_extent(ahead, ni),
                    ni,
                    st.wg,
                    &limits,
                    &client,
                )?;
            }
            if verbosity.normal_verbosity() && epoch % 50 == 0 {
                println!(
                    " Epoch {}/{} | extent = {:.1} | n_boxes = {}",
                    epoch, params.n_epochs, span, grid.geom.n_boxes
                );
            }
        }

        sched.momentum = T::from_f64(if epoch < TSNE_MOMENTUM_SWITCH_ITER {
            TSNE_INITIAL_MOMENTUM
        } else {
            TSNE_FINAL_MOMENTUM
        })
        .unwrap();
        sched.exag = if epoch < params.early_exag_iter {
            params.early_exag_factor
        } else {
            params.get_late_exag_factor()
        };

        st.enqueue_repulsion(&client, &grid);
        st.enqueue_step(&client, &sched);
        std::mem::swap(&mut st.pos_a, &mut st.pos_b);
    }

    let out = st.pos_a.read(&client)?;
    for (i, p) in embd.iter_mut().enumerate() {
        p[0] = out[2 * i];
        p[1] = out[2 * i + 1];
    }
    Ok(())
}

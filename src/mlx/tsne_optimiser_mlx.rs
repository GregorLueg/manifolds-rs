//! Three-kernel FFT t-SNE (and DREAMS) on MLX. Same schedule, grid geometry,
//! update rule and bucketing as the cubecl version in
//! `training::tsne_optimiser_gpu`: atomic box counting, a single-threadgroup
//! scan and a scatter, and per-threadgroup partials for `Z`, the recentring
//! mean, the DREAMS norm and the extent. The splat differs: one SIMD group
//! per box instead of one thread per grid node, 1.4 s faster over 1000
//! epochs at 100k points.
//!
//! The atomic ranks make the in-box point order, and so the splat summation
//! order, vary between runs: results are not bit-reproducible, as on cubecl.
//!
//! MLX's own ops only do the convolution: `rfft2`, one broadcast complex
//! multiply against the three cached kernel spectra and a batched `irfft2`.
//! All `f32`.

use crate::data::graph::coo_to_adjacency_list;
use crate::data::structures::*;
use crate::mlx::ffi::*;
use crate::prelude::*;
use crate::training::tsne_optimiser::*;

////////////
// Consts //
////////////

/// Threads per threadgroup for every kernel here. Not tuned on MLX. The
/// reduction kernels assume a multiple of 32.
const TSNE_MLX_THREADGROUP: usize = 256;

/// Epochs between extent readbacks. Same as the cubecl path.
const GRID_CHECK_INTERVAL: usize = 10;

/// Multiple of the extent growth over the last check interval that the grid
/// is sized ahead for. Same as the cubecl path.
const GRID_GROWTH_SAFETY: f64 = 1.5;

/// Entries in the per-epoch f32 parameter buffer of the update kernel, see
/// [`UPDATE_SOURCE`] for the layout.
const N_STEP_PARAMS: usize = 9;

////////////////////
// Kernel sources //
////////////////////

/// Lagrange basis `k` on `NI` equispaced nodes in `[0, 1]` (node `j` at
/// `(j + 0.5) / NI`), evaluated at `t`.
const LAGRANGE_HEADER: &str = r#"
template <int NI>
inline float lagrange_basis(float t, int k) {
    float inv = 1.0f / (float)NI;
    float sk = ((float)k + 0.5f) * inv;
    float num = 1.0f;
    float den = 1.0f;
    for (int j = 0; j < NI; j++) {
        if (j != k) {
            float sj = ((float)j + 0.5f) * inv;
            num *= t - sj;
            den *= sk - sj;
        }
    }
    return num / den;
}
"#;

/// Box id and Lagrange weights per point. A point past the grid edge between
/// extent checks snaps to the edge instead of extrapolating the weights.
///
/// Template int: `NI`. `gu = [n, nb, nf, n_groups]`, `gp = [coord_min,
/// box_width]`.
const PREP_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint n = gu[0];
    if (i >= n) return;
    uint nb = gu[1];
    float top = (float)(nb - 1);
    float rx = (pos[2 * i] - gp[0]) / gp[1];
    float ry = (pos[2 * i + 1] - gp[0]) / gp[1];
    float fx = clamp(floor(rx), 0.0f, top);
    float fy = clamp(floor(ry), 0.0f, top);
    float tx = clamp(rx - fx, 0.0f, 1.0f);
    float ty = clamp(ry - fy, 0.0f, 1.0f);
    for (int k = 0; k < NI; k++) {
        wx[i * NI + k] = lagrange_basis<NI>(tx, k);
        wy[i * NI + k] = lagrange_basis<NI>(ty, k);
    }
    box_id[i] = (uint)fy * nb + (uint)fx;
"#;

/// Rank of every point inside its box, counting the boxes as it goes. Atomic
/// kernel, outputs zero-filled; the rank order is whatever the atomics give.
///
/// `gu = [n, nb, nf, n_groups]`.
const COUNT_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    if (i >= gu[0]) return;
    uint r = atomic_fetch_add_explicit(&box_count[box_id[i]], 1u, memory_order_relaxed);
    atomic_store_explicit(&slot[i], r, memory_order_relaxed);
"#;

/// Exclusive scan of the box counts into `nb^2 + 1` box starts, one
/// threadgroup. Each thread sums a contiguous chunk, the chunk sums are
/// scanned with SIMD prefix sums, then each thread writes its chunk.
///
/// `gu = [n, nb, nf, n_groups]`.
const SCAN_SOURCE: &str = r#"
    uint t = thread_position_in_threadgroup.x;
    uint len = gu[1] * gu[1];
    uint chunk = (len + threads_per_threadgroup.x - 1) / threads_per_threadgroup.x;
    uint lo = min(t * chunk, len);
    uint hi = min(lo + chunk, len);
    uint acc = 0;
    for (uint b = lo; b < hi; b++) acc += box_count[b];

    threadgroup uint sg_total[32];
    uint pre = simd_prefix_exclusive_sum(acc);
    if (thread_index_in_simdgroup == threads_per_simdgroup - 1) {
        sg_total[simdgroup_index_in_threadgroup] = pre + acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        uint l = thread_index_in_simdgroup;
        uint v = l < simdgroups_per_threadgroup ? sg_total[l] : 0u;
        sg_total[l] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint run = sg_total[simdgroup_index_in_threadgroup] + pre;
    for (uint b = lo; b < hi; b++) {
        box_start[b] = run;
        run += box_count[b];
    }
    if (t == threads_per_threadgroup.x - 1) box_start[len] = run;
"#;

/// Scatter point indices into box order.
///
/// `gu = [n, nb, nf, n_groups]`.
const SCATTER_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    if (i >= gu[0]) return;
    order[box_start[box_id[i]] + slot[i]] = i;
"#;

/// Threadgroup sum and max over SIMD-group partials. `sh` holds one slot per
/// SIMD group; every thread gets the result. Every thread of the group must
/// call these, so kernels using them never return early.
const REDUCE_HEADER: &str = r#"
inline float tg_sum(float v, threadgroup float* sh, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint k = 0; k < nsg; k++) r += sh[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float tg_max(float v, threadgroup float* sh, uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = sh[0];
    for (uint k = 1; k < nsg; k++) r = max(r, sh[k]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}
"#;

/// Unit-charge splat onto the zero-filled grid, one SIMD group per box.
/// Lanes stride over the box's points, so each point's weights are read
/// once rather than once per node, and every lane keeps all `NI^2` node sums
/// in registers before a `simd_sum` per node.
///
/// Template int: `NI`. `gu = [n, nb, nf, n_groups]`.
const SPLAT_SOURCE: &str = r#"
    uint b = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint nb = gu[1];
    uint nf = gu[2];
    // b is uniform over the SIMD group, so the whole group leaves together
    if (b >= nb * nb) return;
    float acc[NI * NI];
    for (int k = 0; k < NI * NI; k++) acc[k] = 0.0f;
    uint end = box_start[b + 1];
    for (uint p = box_start[b] + lane; p < end; p += 32) {
        uint pt = order[p];
        float wxl[NI];
        for (int ix = 0; ix < NI; ix++) wxl[ix] = wx[pt * NI + ix];
        for (int iy = 0; iy < NI; iy++) {
            float w_y = wy[pt * NI + iy];
            for (int ix = 0; ix < NI; ix++) acc[iy * NI + ix] += w_y * wxl[ix];
        }
    }
    uint row0 = (b / nb) * NI;
    uint col0 = (b % nb) * NI;
    for (int k = 0; k < NI * NI; k++) {
        float v = simd_sum(acc[k]);
        if (lane == (uint)(k & 31)) grid[(row0 + k / NI) * nf + col0 + k % NI] = v;
    }
"#;

/// Interpolate `Z_i` and the raw repulsion per point from the three
/// convolved grids `conv = [q, q^2 dx, q^2 dy]`. Writes one partial of
/// `sum_i (Z_i - 1)`, the self term removed, per threadgroup.
///
/// Template int: `NI`. `gu = [n, nb, nf, n_groups]`.
const GATHER_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint n = gu[0];
    float zc = 0.0f;
    if (i < n) {
        uint nb = gu[1];
        uint nf = gu[2];
        uint nn = nf * nf;
        uint m = nb * NI;
        uint b = box_id[i];
        uint by = b / nb;
        uint bx = b % nb;
        float z = 0.0f;
        float fx = 0.0f;
        float fy = 0.0f;
        for (int iy = 0; iy < NI; iy++) {
            float w_y = wy[i * NI + iy];
            uint row = (m + by * NI + iy) * nf + m + bx * NI;
            for (int ix = 0; ix < NI; ix++) {
                float w = w_y * wx[i * NI + ix];
                uint o = row + ix;
                z += w * conv[o];
                fx += w * conv[nn + o];
                fy += w * conv[2 * nn + o];
            }
        }
        rep[2 * i] = fx;
        rep[2 * i + 1] = fy;
        zc = z - 1.0f;
    }
    threadgroup float sh[32];
    float tot = tg_sum(zc, sh, simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
                       simdgroups_per_threadgroup);
    if (thread_position_in_threadgroup.x == 0) zp[threadgroup_position_in_grid.x] = tot;
"#;

/// Attraction over the CSR graph fused with the Jacobs gains, momentum, step
/// cap and (with `DREAMS`) the regulariser step. Writes fresh position,
/// velocity and gain buffers, so neighbour reads always see the old step.
///
/// Every threadgroup reduces the `Z` partials (and with `DREAMS` the
/// `||Y||^2` partials of the incoming positions) itself, which saves a
/// launch, and writes one partial of the new positions for the recentring.
///
/// Template int: `DREAMS` (0 or 1). `gu = [n, nb, nf, n_groups]`. `fp =
/// [exag, lr, momentum, min_gain, max_step, z_floor, inv_ref_norm,
/// dreams_keep, dreams_coef]`.
const UPDATE_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint nsg = simdgroups_per_threadgroup;
    uint n = gu[0];
    uint n_groups = gu[3];
    threadgroup float sh[32];

    float zacc = 0.0f;
    for (uint k = t; k < n_groups; k += tpg) zacc += zp[k];
    float z_raw = tg_sum(zacc, sh, sg, lane, nsg);
    float zz = z_raw > fp[5] ? z_raw : 1.0f;
    float alpha = 0.0f;
    if (DREAMS) {
        float nacc = 0.0f;
        for (uint k = t; k < n_groups; k += tpg) nacc += np[k];
        alpha = sqrt(tg_sum(nacc, sh, sg, lane, nsg)) * fp[6];
    }

    float new_x = 0.0f;
    float new_y = 0.0f;
    if (i < n) {
        float exag = fp[0];
        float lr = fp[1];
        float momentum = fp[2];
        float min_gain = fp[3];
        float max_step = fp[4];

        float px = pos[2 * i];
        float py = pos[2 * i + 1];
        // two independent accumulators keep two neighbour gathers in flight
        float ax0 = 0.0f, ay0 = 0.0f, ax1 = 0.0f, ay1 = 0.0f;
        uint e = indptr[i];
        uint end = indptr[i + 1];
        for (; e + 1 < end; e += 2) {
            uint j0 = indices[e];
            uint j1 = indices[e + 1];
            float dx0 = px - pos[2 * j0];
            float dy0 = py - pos[2 * j0 + 1];
            float dx1 = px - pos[2 * j1];
            float dy1 = py - pos[2 * j1 + 1];
            float f0 = values[e] / (1.0f + dx0 * dx0 + dy0 * dy0);
            float f1 = values[e + 1] / (1.0f + dx1 * dx1 + dy1 * dy1);
            ax0 += f0 * dx0;
            ay0 += f0 * dy0;
            ax1 += f1 * dx1;
            ay1 += f1 * dy1;
        }
        if (e < end) {
            uint j = indices[e];
            float dx = px - pos[2 * j];
            float dy = py - pos[2 * j + 1];
            float f = values[e] / (1.0f + dx * dx + dy * dy);
            ax0 += f * dx;
            ay0 += f * dy;
        }

        float gx = exag * (ax0 + ax1) - rep[2 * i] / zz;
        float gy = exag * (ay0 + ay1) - rep[2 * i + 1] / zz;
        float ux = vel[2 * i];
        float uy = vel[2 * i + 1];
        float g_x = gains[2 * i];
        float g_y = gains[2 * i + 1];
        // Jacobs gains: grow when gradient and update disagree in sign
        g_x = ((gx > 0.0f) != (ux > 0.0f)) ? g_x + 0.2f : g_x * 0.8f;
        g_y = ((gy > 0.0f) != (uy > 0.0f)) ? g_y + 0.2f : g_y * 0.8f;
        g_x = max(g_x, min_gain);
        g_y = max(g_y, min_gain);
        ux = momentum * ux - lr * g_x * gx;
        uy = momentum * uy - lr * g_y * gy;
        float step_sq = ux * ux + uy * uy;
        if (step_sq > max_step * max_step) {
            float scale = max_step / sqrt(step_sq);
            ux *= scale;
            uy *= scale;
        }
        vel_out[2 * i] = ux;
        vel_out[2 * i + 1] = uy;
        gains_out[2 * i] = g_x;
        gains_out[2 * i + 1] = g_y;
        if (DREAMS) {
            float keep = fp[7];
            float coef = fp[8];
            new_x = px + keep * ux - coef * (px - alpha * reference[2 * i]);
            new_y = py + keep * uy - coef * (py - alpha * reference[2 * i + 1]);
        } else {
            new_x = px + ux;
            new_y = py + uy;
        }
        pos_out[2 * i] = new_x;
        pos_out[2 * i + 1] = new_y;
    }

    float sx = tg_sum(new_x, sh, sg, lane, nsg);
    float sy = tg_sum(new_y, sh, sg, lane, nsg);
    if (t == 0) {
        mp[2 * threadgroup_position_in_grid.x] = sx;
        mp[2 * threadgroup_position_in_grid.x + 1] = sy;
    }
"#;

/// Recentre on the origin and record, per threadgroup, the largest absolute
/// coordinate and `sum x^2 + y^2` (the DREAMS scale of the next epoch).
/// Every threadgroup reduces the position partials itself.
///
/// `gu = [n, nb, nf, n_groups]`.
const RECENTRE_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint t = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint nsg = simdgroups_per_threadgroup;
    uint n = gu[0];
    uint n_groups = gu[3];
    threadgroup float sh[32];

    float sx = 0.0f;
    float sy = 0.0f;
    for (uint k = t; k < n_groups; k += tpg) {
        sx += mp[2 * k];
        sy += mp[2 * k + 1];
    }
    float mx = tg_sum(sx, sh, sg, lane, nsg) / (float)n;
    float my = tg_sum(sy, sh, sg, lane, nsg) / (float)n;

    float ext = 0.0f;
    float sq = 0.0f;
    if (i < n) {
        float x = pos[2 * i] - mx;
        float y = pos[2 * i + 1] - my;
        pos_out[2 * i] = x;
        pos_out[2 * i + 1] = y;
        ext = max(abs(x), abs(y));
        sq = x * x + y * y;
    }
    float e = tg_max(ext, sh, sg, lane, nsg);
    float q = tg_sum(sq, sh, sg, lane, nsg);
    if (t == 0) {
        ep[threadgroup_position_in_grid.x] = e;
        np[threadgroup_position_in_grid.x] = q;
    }
"#;

//////////////
// Geometry //
//////////////

/// FFT grid geometry. The CPU path's box count and half-width; the FFT side
/// is the next power of two that holds the zero-padded `2 m` nodes.
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
            n_fft: (2 * ni * n_boxes).next_power_of_two(),
            n_boxes,
            half,
        }
    }

    /// Box width in embedding units.
    fn box_width(&self) -> f64 {
        2.0 * self.half / self.n_boxes as f64
    }

    /// Whether the grid must be rebuilt: the box count for the new extent
    /// changed, or the embedding came within one box of the edge.
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

/// Grid-dependent device state: the geometry scalars and the kernel spectra.
struct Grid {
    /// Geometry the spectra were built for
    geom: GridGeometry,
    /// `[n, nb, nf, n_groups]`
    gu: Array,
    /// `[coord_min, box_width]`
    gp: Array,
    /// Half spectra of `[q, q^2 dx, q^2 dy]`, complex `[3, nf, nf / 2 + 1]`
    kspec: Array,
}

impl Grid {
    /// Build the spatial kernels on the host and transform them on the
    /// device. Element `(r, c)` of kernel `t` holds the kernel at offset
    /// `((c - m) h, (r - m) h)` for `1 <= r, c <= 2m - 1`, zero elsewhere.
    ///
    /// ### Params
    ///
    /// * `geom` - Grid geometry
    /// * `n` - Number of points
    /// * `ni` - Interpolation nodes per box
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The grid state, spectra evaluated.
    fn new(geom: GridGeometry, n: usize, ni: usize, s: &Stream) -> Result<Self, ManifoldsError> {
        let nf = geom.n_fft;
        let nn = nf * nf;
        let m = geom.n_boxes * ni;
        let h = geom.box_width() / ni as f64;
        let mut k = vec![0.0f32; 3 * nn];
        for r in 1..2 * m {
            for c in 1..2 * m {
                let dx = (c as f64 - m as f64) * h;
                let dy = (r as f64 - m as f64) * h;
                let q = 1.0 / (1.0 + dx * dx + dy * dy);
                let o = r * nf + c;
                k[o] = q as f32;
                k[nn + o] = (q * q * dx) as f32;
                k[2 * nn + o] = (q * q * dy) as f32;
            }
        }
        let nfi = nf as i32;
        let kspec = Array::from_f32(&k, &[3, nfi, nfi]).rfft2([nfi, nfi], s)?;
        eval_all(&[&kspec], false)?;
        Ok(Self {
            geom,
            gu: Array::from_u32(
                &[n as u32, geom.n_boxes as u32, nf as u32, n_groups(n) as u32],
                &[4],
            ),
            gp: Array::from_f32(&[-geom.half as f32, geom.box_width() as f32], &[2]),
            kspec,
        })
    }
}

/////////////
// Kernels //
/////////////

/// The compiled kernels.
struct Kernels {
    /// [`PREP_SOURCE`]
    prep: MetalKernel,
    /// [`COUNT_SOURCE`], atomic
    count: MetalKernel,
    /// [`SCAN_SOURCE`]
    scan: MetalKernel,
    /// [`SCATTER_SOURCE`]
    scatter: MetalKernel,
    /// [`SPLAT_SOURCE`]
    splat: MetalKernel,
    /// [`GATHER_SOURCE`]
    gather: MetalKernel,
    /// [`UPDATE_SOURCE`]
    update: MetalKernel,
    /// [`RECENTRE_SOURCE`]
    recentre: MetalKernel,
}

impl Kernels {
    /// Register every kernel. MLX compiles each on first use.
    ///
    /// ### Returns
    ///
    /// The kernel set
    fn new() -> Self {
        Self {
            prep: MetalKernel::new(
                "manifolds_tsne_prep",
                &["pos", "gu", "gp"],
                &["box_id", "wx", "wy"],
                LAGRANGE_HEADER,
                PREP_SOURCE,
            ),
            count: MetalKernel::new_atomic(
                "manifolds_tsne_count",
                &["box_id", "gu"],
                &["box_count", "slot"],
                COUNT_SOURCE,
            ),
            scan: MetalKernel::new(
                "manifolds_tsne_scan",
                &["box_count", "gu"],
                &["box_start"],
                "",
                SCAN_SOURCE,
            ),
            scatter: MetalKernel::new(
                "manifolds_tsne_scatter",
                &["box_id", "slot", "box_start", "gu"],
                &["order"],
                "",
                SCATTER_SOURCE,
            ),
            splat: MetalKernel::new(
                "manifolds_tsne_splat",
                &["order", "box_start", "wx", "wy", "gu"],
                &["grid"],
                "",
                SPLAT_SOURCE,
            ),
            gather: MetalKernel::new(
                "manifolds_tsne_gather",
                &["conv", "box_id", "wx", "wy", "gu"],
                &["rep", "zp"],
                REDUCE_HEADER,
                GATHER_SOURCE,
            ),
            update: MetalKernel::new(
                "manifolds_tsne_update",
                &[
                    "pos",
                    "vel",
                    "gains",
                    "rep",
                    "zp",
                    "indptr",
                    "indices",
                    "values",
                    "reference",
                    "np",
                    "gu",
                    "fp",
                ],
                &["pos_out", "vel_out", "gains_out", "mp"],
                REDUCE_HEADER,
                UPDATE_SOURCE,
            ),
            recentre: MetalKernel::new(
                "manifolds_tsne_recentre",
                &["pos", "mp", "gu"],
                &["pos_out", "ep", "np"],
                REDUCE_HEADER,
                RECENTRE_SOURCE,
            ),
        }
    }
}

/// Number of threadgroups covering `n` points.
///
/// ### Params
///
/// * `n` - Number of points
///
/// ### Returns
///
/// `ceil(n / TSNE_MLX_THREADGROUP)`
fn n_groups(n: usize) -> usize {
    n.div_ceil(TSNE_MLX_THREADGROUP)
}

/// One-dimensional launch over `total` threads.
///
/// ### Params
///
/// * `total` - Number of threads
///
/// ### Returns
///
/// `(grid, threadgroup)` for [`MetalKernel::apply`]
fn launch_1d(total: usize) -> ([i32; 3], [i32; 3]) {
    ([total as i32, 1, 1], [TSNE_MLX_THREADGROUP as i32, 1, 1])
}

/// Launch over whole threadgroups covering `n` points, for the kernels that
/// write one partial per threadgroup: every group is full, so the partial
/// index is the threadgroup index.
///
/// ### Params
///
/// * `n` - Number of points
///
/// ### Returns
///
/// `(grid, threadgroup)` for [`MetalKernel::apply`]
fn launch_groups(n: usize) -> ([i32; 3], [i32; 3]) {
    launch_1d(n_groups(n) * TSNE_MLX_THREADGROUP)
}

/// Output spec shorthand.
///
/// ### Params
///
/// * `shape` - Output shape
/// * `dtype` - Element type
///
/// ### Returns
///
/// The spec
fn spec(shape: &[i32], dtype: mlx_dtype) -> OutputSpec<'_> {
    OutputSpec { shape, dtype }
}

//////////////////
// Device state //
//////////////////

/// Graph and DREAMS arrays, constant over the run.
struct Constants {
    /// CSR row pointers `[n + 1]`
    indptr: Array,
    /// CSR neighbours `[nnz]`
    indices: Array,
    /// CSR affinities `[nnz]`
    values: Array,
    /// Centred DREAMS reference `[n, 2]`, or a `[2]` stub without DREAMS
    reference: Array,
}

/// The arrays the step rewrites every epoch.
struct StepState {
    /// Positions `[n, 2]`, centred
    pos: Array,
    /// Momentum `[n, 2]`
    vel: Array,
    /// Gains `[n, 2]`
    gains: Array,
    /// Per-threadgroup largest absolute coordinate of `pos` `[n_groups]`
    ext: Array,
    /// Per-threadgroup `sum x^2 + y^2` of `pos` `[n_groups]`
    norm: Array,
}

/// Queue the repulsion for the current positions: bucketing, splat, FFT
/// convolution and gather.
///
/// ### Params
///
/// * `k` - Kernel set
/// * `grid` - Grid state
/// * `pos` - Positions `[n, 2]`
/// * `n` - Number of points
/// * `ni` - Interpolation nodes per box
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) raw repulsion `sum_j q^2 (y_i - y_j)` `[n, 2]` and the
/// per-threadgroup partials of `Z = sum_{i != j} q_ij` `[n_groups]`.
fn repulsion(
    k: &Kernels,
    grid: &Grid,
    pos: &Array,
    n: usize,
    ni: usize,
    s: &Stream,
) -> Result<(Array, Array), ManifoldsError> {
    let nf = grid.geom.n_fft;
    let nb2 = grid.geom.n_boxes * grid.geom.n_boxes;
    let ni_t = [("NI", ni as i32)];
    let (nn_i, nw_i) = (n as i32, (n * ni) as i32);

    // bucketing: box id and weights per point, atomic counts and ranks, scan,
    // scatter into box order
    let (g, tg) = launch_1d(n);
    let mut prep = k
        .prep
        .apply(
            &[pos, &grid.gu, &grid.gp],
            &[
                spec(&[nn_i], MLX_UINT32),
                spec(&[nw_i], MLX_FLOAT32),
                spec(&[nw_i], MLX_FLOAT32),
            ],
            g,
            tg,
            &ni_t,
            s,
        )?
        .into_iter();
    let box_id = prep.next().expect("prep has three outputs");
    let wx = prep.next().expect("prep has three outputs");
    let wy = prep.next().expect("prep has three outputs");
    let mut counted = k
        .count
        .apply_zeroed(
            &[&box_id, &grid.gu],
            &[spec(&[nb2 as i32], MLX_UINT32), spec(&[nn_i], MLX_UINT32)],
            g,
            tg,
            &[],
            s,
        )?
        .into_iter();
    let box_count = counted.next().expect("count has two outputs");
    let slot = counted.next().expect("count has two outputs");
    let (sg, stg) = launch_1d(TSNE_MLX_THREADGROUP);
    let box_start = k
        .scan
        .apply(
            &[&box_count, &grid.gu],
            &[spec(&[nb2 as i32 + 1], MLX_UINT32)],
            sg,
            stg,
            &[],
            s,
        )?
        .pop()
        .expect("scan has one output");
    let order = k
        .scatter
        .apply(
            &[&box_id, &slot, &box_start, &grid.gu],
            &[spec(&[nn_i], MLX_UINT32)],
            g,
            tg,
            &[],
            s,
        )?
        .pop()
        .expect("scatter has one output");

    // splat, convolve, gather
    let nfi = nf as i32;
    let (g, tg) = launch_1d(32 * nb2);
    let charges = k
        .splat
        .apply_zeroed(
            &[&order, &box_start, &wx, &wy, &grid.gu],
            &[spec(&[nfi, nfi], MLX_FLOAT32)],
            g,
            tg,
            &ni_t,
            s,
        )?
        .pop()
        .expect("splat has one output");
    let conv = charges
        .rfft2([nfi, nfi], s)?
        .multiply(&grid.kspec, s)?
        .irfft2([nfi, nfi], s)?;
    let (g, tg) = launch_groups(n);
    let mut gathered = k
        .gather
        .apply(
            &[&conv, &box_id, &wx, &wy, &grid.gu],
            &[
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[n_groups(n) as i32], MLX_FLOAT32),
            ],
            g,
            tg,
            &ni_t,
            s,
        )?
        .into_iter();
    let rep = gathered.next().expect("gather has two outputs");
    let zp = gathered.next().expect("gather has two outputs");
    Ok((rep, zp))
}

/// Queue one full epoch: repulsion, the fused step and the recentring.
///
/// ### Params
///
/// * `k` - Kernel set
/// * `grid` - Grid state
/// * `c` - Constant arrays
/// * `st` - State of the previous epoch
/// * `n` - Number of points
/// * `ni` - Interpolation nodes per box
/// * `dreams` - Compile in the DREAMS step
/// * `fp` - Per-epoch parameters, see [`UPDATE_SOURCE`]
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) state of the next epoch
#[allow(clippy::too_many_arguments)]
fn epoch(
    k: &Kernels,
    grid: &Grid,
    c: &Constants,
    st: &StepState,
    n: usize,
    ni: usize,
    dreams: bool,
    fp: &Array,
    s: &Stream,
) -> Result<StepState, ManifoldsError> {
    let (rep, zp) = repulsion(k, grid, &st.pos, n, ni, s)?;

    let nn_i = n as i32;
    let ng_i = n_groups(n) as i32;
    let (g, tg) = launch_groups(n);
    let mut out = k
        .update
        .apply(
            &[
                &st.pos,
                &st.vel,
                &st.gains,
                &rep,
                &zp,
                &c.indptr,
                &c.indices,
                &c.values,
                &c.reference,
                &st.norm,
                &grid.gu,
                fp,
            ],
            &[
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[2 * ng_i], MLX_FLOAT32),
            ],
            g,
            tg,
            &[("DREAMS", dreams as i32)],
            s,
        )?
        .into_iter();
    let pos = out.next().expect("update has four outputs");
    let vel = out.next().expect("update has four outputs");
    let gains = out.next().expect("update has four outputs");
    let mp = out.next().expect("update has four outputs");
    let mut out = k
        .recentre
        .apply(
            &[&pos, &mp, &grid.gu],
            &[
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[ng_i], MLX_FLOAT32),
                spec(&[ng_i], MLX_FLOAT32),
            ],
            g,
            tg,
            &[],
            s,
        )?
        .into_iter();
    Ok(StepState {
        pos: out.next().expect("recentre has three outputs"),
        vel,
        gains,
        ext: out.next().expect("recentre has three outputs"),
        norm: out.next().expect("recentre has three outputs"),
    })
}

//////////
// Main //
//////////

/// Optimise a 2D embedding with three-kernel FFT t-SNE on MLX.
///
/// Same schedule and update rule as `optimise_fft3k_tsne_gpu` (momentum
/// switch, early / late exaggeration, Jacobs gains, step cap, recentring,
/// optional DREAMS step), one extent readback every `GRID_CHECK_INTERVAL`
/// epochs, and the embedding read back once at the end. Computes in `f32`
/// whatever `T` is.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][2]`, overwritten with the result
/// * `params` - Optimisation hyperparameters
/// * `graph` - Symmetric t-SNE affinities
/// * `dreams` - DREAMS regulariser state, or `None` for plain t-SNE
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `Ok(())` on success; `NoData` if the embedding is empty, `IncorrectDim` if
/// it is not 2D, `GraphSizeMismatch` if the graph covers a different number
/// of points, `MlxError` if an mlx-c call fails.
///
/// ### References
///
/// Linderman et al., Nature Methods, 2019 (FIt-SNE).
pub fn optimise_fft3k_tsne_mlx<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dreams: Option<&DreamsState<T>>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
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
    if graph.n_samples != n {
        return Err(ManifoldsError::GraphSizeMismatch {
            n_graph: graph.n_samples,
            n_embd: n,
        });
    }
    let ni = params.n_interp_points;
    let f = |x: T| x.to_f32().unwrap();

    // CSR, with a dummy entry so an edgeless graph still binds
    let adj = coo_to_adjacency_list(graph);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in &adj {
        for &(j, w) in row {
            indices.push(j as u32);
            values.push(f(w));
        }
        indptr.push(indices.len() as u32);
    }
    if indices.is_empty() {
        indices.push(0);
        values.push(0.0);
    }

    let pos: Vec<f32> = embd.iter().flat_map(|p| [f(p[0]), f(p[1])]).collect();
    let half_span = pos.iter().fold(0.0f64, |a, &v| a.max(v.abs() as f64));
    let reference: Vec<f32> = match dreams {
        Some(d) => d.reference().iter().map(|&x| f(x)).collect(),
        None => vec![0.0; 2],
    };

    install_error_handler();
    let s = Stream::default_gpu();
    let k = Kernels::new();
    let nn_i = n as i32;
    let c = Constants {
        indptr: Array::from_u32(&indptr, &[nn_i + 1]),
        indices: Array::from_u32(&indices, &[indices.len() as i32]),
        values: Array::from_f32(&values, &[values.len() as i32]),
        reference: Array::from_f32(&reference, &[reference.len() as i32]),
    };
    // the first epoch reads the norm partials of the initial positions; one
    // full partial plus zeros sums to the same
    let ng = n_groups(n);
    let mut norm = vec![0.0f32; ng];
    norm[0] = pos.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() as f32;
    let mut st = StepState {
        pos: Array::from_f32(&pos, &[nn_i, 2]),
        vel: Array::from_f32(&vec![0.0; 2 * n], &[nn_i, 2]),
        gains: Array::from_f32(&vec![1.0; 2 * n], &[nn_i, 2]),
        ext: Array::from_f32(&vec![half_span as f32; ng], &[ng as i32]),
        norm: Array::from_f32(&norm, &[ng as i32]),
    };
    let mut grid = Grid::new(GridGeometry::for_extent(half_span, ni), n, ni, &s)?;

    let lr = params.get_lr(n);
    let mut fp = [0.0f32; N_STEP_PARAMS];
    fp[1] = f(lr);
    fp[3] = TSNE_MIN_GAIN as f32;
    fp[4] = f(step_cap_from_lr(lr));
    fp[5] = TSNE_EPS as f32;
    fp[6] = dreams.map_or(1.0, |d| 1.0 / d.ref_norm()) as f32;
    fp[7] = dreams.map_or(1.0, |d| f(d.keep()));

    if verbosity.normal_verbosity() {
        println!(
            "Running {} epochs of three-kernel FFT t-SNE on MLX.",
            params.n_epochs
        );
    }

    let mut last_span = half_span;
    for ep in 0..params.n_epochs {
        if ep > 0 && ep % GRID_CHECK_INTERVAL == 0 {
            eval_all(&[&st.ext], false)?;
            let span = st
                .ext
                .as_f32()?
                .iter()
                .fold(0.0f64, |a, &v| a.max(v as f64));
            let ahead = span + GRID_GROWTH_SAFETY * (span - last_span).max(0.0);
            last_span = span;
            if grid.geom.needs_rebuild(ahead, ni) {
                grid = Grid::new(GridGeometry::for_extent(ahead, ni), n, ni, &s)?;
            }
            if verbosity.normal_verbosity() && ep % 50 == 0 {
                println!(
                    " Epoch {}/{} | extent = {:.1} | n_boxes = {}",
                    ep, params.n_epochs, span, grid.geom.n_boxes
                );
            }
        }

        let exag = if ep < params.early_exag_iter {
            params.early_exag_factor
        } else {
            params.get_late_exag_factor()
        };
        fp[0] = f(exag);
        fp[2] = if ep < TSNE_MOMENTUM_SWITCH_ITER {
            TSNE_INITIAL_MOMENTUM
        } else {
            TSNE_FINAL_MOMENTUM
        } as f32;
        fp[8] = dreams.map_or(0.0, |d| f(d.step_coef(exag)));
        let fp_arr = Array::from_f32(&fp, &[N_STEP_PARAMS as i32]);

        let next = epoch(&k, &grid, &c, &st, n, ni, dreams.is_some(), &fp_arr, &s)?;
        // stay at most one epoch ahead of the device
        eval_all(
            &[&next.pos, &next.vel, &next.gains, &next.ext, &next.norm],
            true,
        )?;
        eval_all(&[&st.pos], false)?;
        st = next;
    }

    eval_all(&[&st.pos], false)?;
    let out = st.pos.as_f32()?;
    for (i, p) in embd.iter_mut().enumerate() {
        p[0] = T::from(out[2 * i]).unwrap();
        p[1] = T::from(out[2 * i + 1]).unwrap();
    }
    drop(st);
    drop(grid);
    clear_cache()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten clusters in a square of half-width `radius`, recentred, from a
    /// fixed LCG. Same fixture as the cubecl tests.
    fn clustered(n: usize, radius: f64) -> Vec<f32> {
        let mut state = 7u64;
        let mut unif = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let centres: Vec<(f64, f64)> = (0..10)
            .map(|_| ((unif() * 2.0 - 1.0) * radius, (unif() * 2.0 - 1.0) * radius))
            .collect();
        let mut pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let (cx, cy) = centres[i % 10];
                (cx + (unif() - 0.5) * 4.0, cy + (unif() - 0.5) * 4.0)
            })
            .collect();
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n as f64;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n as f64;
        pts.iter_mut().for_each(|p| {
            p.0 -= mx;
            p.1 -= my;
        });
        pts.iter()
            .flat_map(|&(x, y)| [x as f32, y as f32])
            .collect()
    }

    fn empty_graph(n: usize) -> CoordinateList<f32> {
        CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: n,
        }
    }

    #[test]
    fn test_mlx_repulsion_matches_direct() {
        let n = 2_000;
        let pos = clustered(n, 30.0);
        let half_span = pos.iter().fold(0.0f64, |a, &v| a.max(v.abs() as f64));
        let s = Stream::default_gpu();
        let k = Kernels::new();
        let grid = Grid::new(GridGeometry::for_extent(half_span, 3), n, 3, &s).unwrap();
        let (rep, zp) =
            repulsion(&k, &grid, &Array::from_f32(&pos, &[n as i32, 2]), n, 3, &s).unwrap();
        eval_all(&[&rep, &zp], false).unwrap();
        let rep = rep.as_f32().unwrap();
        let z: f64 = zp.as_f32().unwrap().iter().map(|&v| v as f64).sum();

        let (mut z_ex, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        let mut f_ex = vec![(0.0f64, 0.0f64); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = (pos[2 * i] - pos[2 * j]) as f64;
                    let dy = (pos[2 * i + 1] - pos[2 * j + 1]) as f64;
                    let q = 1.0 / (1.0 + dx * dx + dy * dy);
                    z_ex += q;
                    f_ex[i].0 += q * q * dx;
                    f_ex[i].1 += q * q * dy;
                }
            }
        }
        for i in 0..n {
            let (rx, ry) = (f_ex[i].0 / z_ex, f_ex[i].1 / z_ex);
            let (gx, gy) = (rep[2 * i] as f64 / z, rep[2 * i + 1] as f64 / z);
            num += (gx - rx).powi(2) + (gy - ry).powi(2);
            den += rx * rx + ry * ry;
        }
        let z_err = (z - z_ex).abs() / z_ex;
        let f_err = (num / den).sqrt();
        assert!(z_err < 1e-2, "Z relative error {z_err:.3e}");
        assert!(f_err < 0.05, "force relative RMS {f_err:.3e}");
    }

    #[test]
    fn test_mlx_optimiser_rejects_bad_input() {
        let params = TsneOptimParams::<f32>::default();
        let mut empty: Vec<Vec<f32>> = vec![];
        let res = optimise_fft3k_tsne_mlx(&mut empty, &params, &empty_graph(0), None, 0);
        assert!(matches!(res, Err(ManifoldsError::NoData)));

        let mut three_d = vec![vec![0.0f32; 3]; 4];
        let res = optimise_fft3k_tsne_mlx(&mut three_d, &params, &empty_graph(4), None, 0);
        assert!(matches!(
            res,
            Err(ManifoldsError::IncorrectDim { n_dim: 3 })
        ));

        let mut two_d = vec![vec![0.0f32; 2]; 4];
        let res = optimise_fft3k_tsne_mlx(&mut two_d, &params, &empty_graph(5), None, 0);
        assert!(matches!(
            res,
            Err(ManifoldsError::GraphSizeMismatch {
                n_graph: 5,
                n_embd: 4
            })
        ));
    }
}

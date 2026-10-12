//! Batched complex 2D FFT on the GPU for power-of-two sizes up to
//! `FFT2D_MAX_N`, used by the device-resident FFT t-SNE optimiser.
//!
//! Row / transpose / row decomposition. Each row is one workgroup that holds
//! the whole signal in shared memory, with the bit-reversal fused into the
//! load and a shared twiddle table. The forward transform leaves its output
//! transposed and the inverse takes a transposed input, so a convolution
//! needs no final transposes as long as the kernel spectrum is transposed
//! too. Both passes can skip rows known to be zero on the way in or unread
//! on the way out. Both directions are unnormalised.
//!
//! The butterfly and transpose are ported from the `gpu-fft` crate
//! (Eugene Hauptmann, MIT licence) and its 2D convolution fork.

#![allow(missing_docs)] // cubecl weirdness

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use std::f32::consts::PI;
use std::ops::Range;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Smallest supported transform size. Below this the transpose tiles do not
/// divide the matrix.
pub const FFT2D_MIN_N: usize = 64;

/// Largest supported transform size. One row runs in one workgroup of `n / 2`
/// units with `3 n` floats of shared memory, so this is bounded by the
/// workgroup limit and the 16 KB shared-memory floor of WebGPU.
pub const FFT2D_MAX_N: usize = 1024;

/// Side length of a transpose tile. Matches the plane width on every backend
/// we target; the shared tile is padded to `TILE + 1` columns so the
/// transposed read does not hit one bank 32 times.
const TRANSPOSE_TILE: usize = 32;

/// Rows of a transpose tile each unit handles. The tile is `32 x 32` but the
/// workgroup is `32 x 8`, well inside every device's workgroup limit.
const TRANSPOSE_ROWS_PER_UNIT: usize = 4;

/////////////
// Kernels //
/////////////

/// In-place 1D FFT of rows `row0..row0 + rows` of every matrix in a batch of
/// `n x n` complex matrices.
///
/// One workgroup per row, `n / 2` units, one butterfly pair per unit per
/// stage. The bit-reversed input order is produced by scattering into shared
/// memory on the load, which keeps the global reads contiguous. Entries at
/// column `len` and beyond are read as zero.
///
/// ### Params
///
/// * `re` - Real parts, `[batch * n * n]`, rows contiguous
/// * `im` - Imaginary parts, same layout
/// * `n_cubes` - `batch * rows`; the grid may hold more cubes than this
/// * `rows` - Rows transformed per matrix
/// * `row0` - First row transformed
/// * `len` - Columns of input that may be nonzero
/// * `n` - Row length, a power of two; comptime
/// * `log2n` - `log2(n)`; comptime
/// * `forward` - `true` for the forward transform (`exp(-i...)`); comptime
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> `(matrix, row)` as
///   `(c / rows, row0 + c % rows)`
/// * `UNIT_POS_X` -> butterfly pair
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn fft_rows<F: Float + CubeElement>(
    re: &mut Tensor<F>,
    im: &mut Tensor<F>,
    n_cubes: u32,
    rows: u32,
    row0: u32,
    len: u32,
    #[comptime] n: usize,
    #[comptime] log2n: usize,
    #[comptime] forward: bool,
) {
    let half = n / 2;
    let mut s_re = Shared::<[F]>::new_slice(n);
    let mut s_im = Shared::<[F]>::new_slice(n);
    let mut tw_re = Shared::<[F]>::new_slice(half);
    let mut tw_im = Shared::<[F]>::new_slice(half);

    // c is uniform over the cube, so the whole cube leaves before any sync
    let c = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if c >= n_cubes {
        terminate!();
    }
    let row = (c / rows) as usize * n + (row0 + c % rows) as usize;
    let local = UNIT_POS_X as usize;
    let base = row * n;
    let zero = F::new(0.0_f32);
    let in_a = local < len as usize;
    let in_b = local + half < len as usize;

    let mut br_a = 0usize;
    let mut br_b = 0usize;
    let mut xa = local;
    let mut xb = local + half;
    #[unroll]
    for _ in 0..log2n {
        br_a = (br_a << 1) | (xa & 1);
        br_b = (br_b << 1) | (xb & 1);
        xa >>= 1;
        xb >>= 1;
    }

    s_re[br_a] = select(in_a, re[base + local], zero);
    s_im[br_a] = select(in_a, im[base + local], zero);
    s_re[br_b] = select(in_b, re[base + local + half], zero);
    s_im[br_b] = select(in_b, im[base + local + half], zero);

    // T[j] = exp(-2 pi i j / n) for j < n / 2, one entry per unit
    let angle = F::new(-2.0 * PI / n as f32) * F::cast_from(local);
    tw_re[local] = F::cos(angle);
    tw_im[local] = F::sin(angle);

    sync_cube();

    #[unroll]
    for s in 0..log2n {
        let hs = 1usize << s;
        let k = local % hs;
        let i = (local / hs) * (hs * 2) + k;
        let j = i + hs;

        // stage twiddle exp(-i pi k / hs) = T[k * n / (2 hs)]; the inverse is
        // its conjugate
        let t = k * (n / (hs * 2));
        let c = tw_re[t];
        let sn = if forward { tw_im[t] } else { -tw_im[t] };

        let ur = s_re[i];
        let ui = s_im[i];
        let vr = c * s_re[j] - sn * s_im[j];
        let vi = sn * s_re[j] + c * s_im[j];

        s_re[i] = ur + vr;
        s_im[i] = ui + vi;
        s_re[j] = ur - vr;
        s_im[j] = ui - vi;

        sync_cube();
    }

    re[base + local] = s_re[local];
    im[base + local] = s_im[local];
    re[base + local + half] = s_re[local + half];
    im[base + local + half] = s_im[local + half];
}

/// Transpose real and imaginary parts of a block of tiles of a batch of
/// `n x n` matrices: source tile `(tr0 + y, tc0 + x)` lands at destination
/// tile `(tc0 + x, tr0 + y)`.
///
/// Tiled through padded shared memory so both the load and the store are
/// coalesced.
///
/// ### Params
///
/// * `in_re` - Input real parts `[batch * n * n]`
/// * `in_im` - Input imaginary parts
/// * `out_re` - Output real parts, transposed
/// * `out_im` - Output imaginary parts, transposed
/// * `tr0` - First source tile row
/// * `tc0` - First source tile column
/// * `n` - Matrix side, a multiple of 32; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_X, CUBE_POS_Y)` -> source tile column and row past
///   `(tc0, tr0)`, `CUBE_POS_Z` -> matrix
/// * `(UNIT_POS_X, UNIT_POS_Y)` -> column and first row inside the tile
#[cube(launch_unchecked)]
pub fn transpose_paired<F: Float + CubeElement>(
    in_re: &Tensor<F>,
    in_im: &Tensor<F>,
    out_re: &mut Tensor<F>,
    out_im: &mut Tensor<F>,
    tr0: u32,
    tc0: u32,
    #[comptime] n: usize,
) {
    let tile = TRANSPOSE_TILE;
    let stride = TRANSPOSE_TILE + 1;
    let rows_step = TRANSPOSE_TILE / TRANSPOSE_ROWS_PER_UNIT;
    let mut s_re = Shared::<[F]>::new_slice(TRANSPOSE_TILE * (TRANSPOSE_TILE + 1));
    let mut s_im = Shared::<[F]>::new_slice(TRANSPOSE_TILE * (TRANSPOSE_TILE + 1));

    let off = CUBE_POS_Z as usize * n * n;
    let tr = (CUBE_POS_Y + tr0) as usize;
    let tc = (CUBE_POS_X + tc0) as usize;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;

    #[unroll]
    for k in 0..TRANSPOSE_ROWS_PER_UNIT {
        let r = ty + k * rows_step;
        let idx = off + (tr * tile + r) * n + tc * tile + tx;
        s_re[r * stride + tx] = in_re[idx];
        s_im[r * stride + tx] = in_im[idx];
    }

    sync_cube();

    #[unroll]
    for k in 0..TRANSPOSE_ROWS_PER_UNIT {
        let r = ty + k * rows_step;
        let idx = off + (tc * tile + r) * n + tr * tile + tx;
        out_re[idx] = s_re[tx * stride + r];
        out_im[idx] = s_im[tx * stride + r];
    }
}

//////////
// Plan //
//////////

/// Validated launch geometry for a batched 2D FFT of one size.
#[derive(Clone, Debug)]
pub struct Fft2dPlan {
    /// Matrix side length
    pub n: usize,
    /// Number of matrices transformed per call
    pub batch: usize,
    /// Device limit on either grid axis; every partial pass launches fewer
    /// cubes than the full grids checked in [`Fft2dPlan::new`]
    max_grid_dim: u32,
}

impl Fft2dPlan {
    /// Plan a batched 2D FFT and check it against the device.
    ///
    /// ### Params
    ///
    /// * `n` - Matrix side; a power of two in `[FFT2D_MIN_N, FFT2D_MAX_N]`
    /// * `batch` - Number of matrices per call
    /// * `limits` - Device limits
    ///
    /// ### Returns
    ///
    /// The plan, or an error if the size is unsupported or the grid, the
    /// workgroup or the shared memory does not fit the device.
    pub fn new(n: usize, batch: usize, limits: &GpuLimits) -> Result<Self, ManifoldsError> {
        if !n.is_power_of_two() || !(FFT2D_MIN_N..=FFT2D_MAX_N).contains(&n) {
            return Err(ManifoldsError::UnsupportedFftSize {
                n,
                min: FFT2D_MIN_N,
                max: FFT2D_MAX_N,
            });
        }
        let float_bytes = std::mem::size_of::<f32>();
        fits_shared_memory("fft_rows", 3 * n * float_bytes, limits)?;
        let cap = limits.max_units_per_cube.min(limits.max_cube_dim.0);
        if (n / 2) as u32 > cap {
            return Err(ManifoldsError::UnsupportedFftSize {
                n,
                min: FFT2D_MIN_N,
                max: 2 * cap as usize,
            });
        }

        let (gx, gy) = grid_2d((batch * n) as u32, limits)?;
        checked_cube_count("fft_rows", gx, gy, 1, limits)?;
        let tiles = (n / TRANSPOSE_TILE) as u32;
        checked_cube_count("transpose_paired", tiles, tiles, batch as u32, limits)?;

        let (mx, my, _) = limits.max_cube_count;
        Ok(Self {
            n,
            batch,
            max_grid_dim: mx.min(my),
        })
    }

    /// Forward 2D FFT of `src`, written transposed to `dst`.
    ///
    /// Only rows `0..rows_nonzero` of each source matrix may be nonzero; the
    /// other rows are skipped. `src` is left in an unspecified state. Both
    /// buffers hold at least `batch * n * n` elements.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `src_re` - Input real parts, clobbered
    /// * `src_im` - Input imaginary parts, clobbered
    /// * `dst_re` - Output real parts, the transpose of the spectrum
    /// * `dst_im` - Output imaginary parts, same layout
    /// * `rows_nonzero` - Leading rows of the input that may be nonzero
    #[allow(clippy::too_many_arguments)]
    pub fn forward_transposed<F: ManifoldsFloatGpu>(
        &self,
        client: &Client,
        src_re: &GpuTensor<F>,
        src_im: &GpuTensor<F>,
        dst_re: &GpuTensor<F>,
        dst_im: &GpuTensor<F>,
        rows_nonzero: usize,
    ) {
        let n_tiles = self.n / TRANSPOSE_TILE;
        let used_tiles = rows_nonzero.div_ceil(TRANSPOSE_TILE);
        self.rows(client, src_re, src_im, 0..rows_nonzero, self.n, true);
        self.transpose(client, src_re, src_im, dst_re, dst_im, 0..used_tiles, 0..n_tiles);
        self.rows(client, dst_re, dst_im, 0..self.n, rows_nonzero, true);
    }

    /// Inverse 2D FFT of the matrix whose transpose is in `src`, written in
    /// normal layout to `dst`.
    ///
    /// Only rows `out_rows` of each output matrix are computed; the rest of
    /// `dst` is left in an unspecified state, as is `src`. Both buffers hold
    /// at least `batch * n * n` elements.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `src_re` - Transposed input real parts, clobbered
    /// * `src_im` - Transposed input imaginary parts, clobbered
    /// * `dst_re` - Output real parts
    /// * `dst_im` - Output imaginary parts
    /// * `out_rows` - Output rows that are read afterwards
    #[allow(clippy::too_many_arguments)]
    pub fn inverse_from_transposed<F: ManifoldsFloatGpu>(
        &self,
        client: &Client,
        src_re: &GpuTensor<F>,
        src_im: &GpuTensor<F>,
        dst_re: &GpuTensor<F>,
        dst_im: &GpuTensor<F>,
        out_rows: Range<usize>,
    ) {
        let n_tiles = self.n / TRANSPOSE_TILE;
        let tiles = out_rows.start / TRANSPOSE_TILE..out_rows.end.div_ceil(TRANSPOSE_TILE);
        self.rows(client, src_re, src_im, 0..self.n, self.n, false);
        self.transpose(client, src_re, src_im, dst_re, dst_im, 0..n_tiles, tiles);
        self.rows(client, dst_re, dst_im, out_rows, self.n, false);
    }

    /// Launch a row pass over rows `rows` of every matrix.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `re` - Real parts, transformed in place
    /// * `im` - Imaginary parts, transformed in place
    /// * `rows` - Rows to transform
    /// * `len` - Columns of input that may be nonzero
    /// * `forward` - Transform direction
    fn rows<F: ManifoldsFloatGpu>(
        &self,
        client: &Client,
        re: &GpuTensor<F>,
        im: &GpuTensor<F>,
        rows: Range<usize>,
        len: usize,
        forward: bool,
    ) {
        let n_cubes = (self.batch * rows.len()) as u32;
        let (gx, gy) = grid_2d_limited(n_cubes, self.max_grid_dim)
            .expect("no larger than the row grid checked in Fft2dPlan::new");
        unsafe {
            fft_rows::launch_unchecked::<F>(
                client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_1d((self.n / 2) as u32),
                re.into_tensor_arg(),
                im.into_tensor_arg(),
                n_cubes,
                rows.len() as u32,
                rows.start as u32,
                len as u32,
                self.n,
                self.n.trailing_zeros() as usize,
                forward,
            );
        }
    }

    /// Launch the paired transpose `src -> dst` over a block of source tiles.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `src_re` - Source real parts
    /// * `src_im` - Source imaginary parts
    /// * `dst_re` - Destination real parts
    /// * `dst_im` - Destination imaginary parts
    /// * `tile_rows` - Source tile rows to move
    /// * `tile_cols` - Source tile columns to move
    #[allow(clippy::too_many_arguments)]
    fn transpose<F: ManifoldsFloatGpu>(
        &self,
        client: &Client,
        src_re: &GpuTensor<F>,
        src_im: &GpuTensor<F>,
        dst_re: &GpuTensor<F>,
        dst_im: &GpuTensor<F>,
        tile_rows: Range<usize>,
        tile_cols: Range<usize>,
    ) {
        unsafe {
            transpose_paired::launch_unchecked::<F>(
                client,
                CubeCount::Static(
                    tile_cols.len() as u32,
                    tile_rows.len() as u32,
                    self.batch as u32,
                ),
                CubeDim::new_2d(
                    TRANSPOSE_TILE as u32,
                    (TRANSPOSE_TILE / TRANSPOSE_ROWS_PER_UNIT) as u32,
                ),
                src_re.into_tensor_arg(),
                src_im.into_tensor_arg(),
                dst_re.into_tensor_arg(),
                dst_im.into_tensor_arg(),
                tile_rows.start as u32,
                tile_cols.start as u32,
                self.n,
            );
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    /// Deterministic values in [-1, 1) from a fixed LCG.
    fn lcg_values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Which transform [`run`] applies.
    enum Pass {
        /// [`Fft2dPlan::forward_transposed`] with this many nonzero rows
        Forward(usize),
        /// [`Fft2dPlan::inverse_from_transposed`] for these output rows
        Inverse(Range<usize>),
    }

    /// Upload, transform, read back the destination.
    fn run(re: &[f32], im: &[f32], n: usize, batch: usize, pass: Pass) -> (Vec<f32>, Vec<f32>) {
        let client = <WgpuRuntime>::client(&WgpuDevice::default());
        let limits = GpuLimits::from_client(&client);
        let plan = Fft2dPlan::new(n, batch, &limits).unwrap();
        let len = batch * n * n;
        let s_re = GpuTensor::<f32>::from_slice(re, vec![len], &client).unwrap();
        let s_im = GpuTensor::<f32>::from_slice(im, vec![len], &client).unwrap();
        let d_re = GpuTensor::<f32>::empty(vec![len], &client).unwrap();
        let d_im = GpuTensor::<f32>::empty(vec![len], &client).unwrap();
        match pass {
            Pass::Forward(rows) => {
                plan.forward_transposed(&client, &s_re, &s_im, &d_re, &d_im, rows)
            }
            Pass::Inverse(rows) => {
                plan.inverse_from_transposed(&client, &s_re, &s_im, &d_re, &d_im, rows)
            }
        }
        (d_re.read(&client).unwrap(), d_im.read(&client).unwrap())
    }

    /// Naive 2D DFT of an `n x n` complex matrix in `f64`, `sign` -1 for
    /// forward and +1 for inverse, unnormalised.
    fn naive_dft(re: &[f32], im: &[f32], n: usize, sign: f64) -> (Vec<f64>, Vec<f64>) {
        let mut out_re = vec![0.0f64; n * n];
        let mut out_im = vec![0.0f64; n * n];
        for u in 0..n {
            for v in 0..n {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for r in 0..n {
                    for c in 0..n {
                        let a = sign * std::f64::consts::TAU * ((u * r + v * c) % n) as f64
                            / n as f64;
                        let (x, y) = (re[r * n + c] as f64, im[r * n + c] as f64);
                        sr += x * a.cos() - y * a.sin();
                        si += x * a.sin() + y * a.cos();
                    }
                }
                out_re[u * n + v] = sr;
                out_im[u * n + v] = si;
            }
        }
        (out_re, out_im)
    }

    #[test]
    fn test_fft2d_forward_transposed_matches_naive_dft() {
        // rows past `nonzero` are zero and skipped; 40 is not tile-aligned
        let (n, nonzero) = (64, 40);
        let mut re = lcg_values(n * n, 1);
        let mut im = lcg_values(n * n, 2);
        re[nonzero * n..].fill(0.0);
        im[nonzero * n..].fill(0.0);
        let (out_re, out_im) = run(&re, &im, n, 1, Pass::Forward(nonzero));
        let (x_re, x_im) = naive_dft(&re, &im, n, -1.0);
        let (mut max_err, mut max_mag) = (0.0f64, 0.0f64);
        for u in 0..n {
            for v in 0..n {
                let (k, kt) = (u * n + v, v * n + u);
                max_err = max_err
                    .max((out_re[kt] as f64 - x_re[k]).abs())
                    .max((out_im[kt] as f64 - x_im[k]).abs());
                max_mag = max_mag.max(x_re[k].abs()).max(x_im[k].abs());
            }
        }
        assert!(
            max_err < 1e-4 * max_mag,
            "max err {max_err:.3e} vs max |X| {max_mag:.3e}"
        );
    }

    #[test]
    fn test_fft2d_inverse_from_transposed_matches_naive_dft() {
        // only the output rows in 20..45 are computed; neither end is
        // tile-aligned
        let (n, rows) = (64, 20..45);
        let re = lcg_values(n * n, 3);
        let im = lcg_values(n * n, 4);
        let (out_re, out_im) = run(&re, &im, n, 1, Pass::Inverse(rows.clone()));
        let t = |a: &[f32]| (0..n * n).map(|k| a[(k % n) * n + k / n]).collect::<Vec<_>>();
        let (x_re, x_im) = naive_dft(&t(&re), &t(&im), n, 1.0);
        let (mut max_err, mut max_mag) = (0.0f64, 0.0f64);
        for u in rows {
            for v in 0..n {
                let k = u * n + v;
                max_err = max_err
                    .max((out_re[k] as f64 - x_re[k]).abs())
                    .max((out_im[k] as f64 - x_im[k]).abs());
                max_mag = max_mag.max(x_re[k].abs()).max(x_im[k].abs());
            }
        }
        assert!(
            max_err < 1e-4 * max_mag,
            "max err {max_err:.3e} vs max |X| {max_mag:.3e}"
        );
    }

    #[test]
    fn test_fft2d_batched_roundtrip() {
        let (n, batch) = (128, 3);
        let re = lcg_values(batch * n * n, 5);
        let im = lcg_values(batch * n * n, 6);
        let (f_re, f_im) = run(&re, &im, n, batch, Pass::Forward(n));
        let (b_re, b_im) = run(&f_re, &f_im, n, batch, Pass::Inverse(0..n));
        let scale = 1.0 / (n * n) as f32;
        for i in 0..re.len() {
            assert!((b_re[i] * scale - re[i]).abs() < 1e-4, "re[{i}]");
            assert!((b_im[i] * scale - im[i]).abs() < 1e-4, "im[{i}]");
        }
    }

    #[test]
    fn test_fft2d_plan_rejects_unsupported_sizes() {
        let client = <WgpuRuntime>::client(&WgpuDevice::default());
        let limits = GpuLimits::from_client(&client);
        for n in [32, 48, 2048] {
            assert!(matches!(
                Fft2dPlan::new(n, 1, &limits),
                Err(ManifoldsError::UnsupportedFftSize { .. })
            ));
        }
    }
}

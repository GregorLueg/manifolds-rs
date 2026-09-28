//! Batched complex 2D FFT on the GPU for power-of-two sizes up to
//! `FFT2D_MAX_N`, used by the device-resident FFT t-SNE optimiser.
//!
//! Row / transpose / row / transpose decomposition. Each row is one workgroup
//! that holds the whole signal in shared memory, with the bit-reversal fused
//! into the load and a shared twiddle table, so a 2D transform is four
//! launches. Both directions are unnormalised.
//!
//! The butterfly and transpose are ported from the `gpu-fft` crate
//! (Eugene Hauptmann, MIT licence) and its 2D convolution fork.

#![allow(missing_docs)] // cubecl weirdness

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use std::f32::consts::PI;

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

/// In-place 1D FFT of every row of a batch of `n x n` complex matrices.
///
/// One workgroup per row, `n / 2` units, one butterfly pair per unit per
/// stage. The bit-reversed input order is produced by scattering into shared
/// memory on the load, which keeps the global reads contiguous.
///
/// ### Params
///
/// * `re` - Real parts, `[batch * n * n]`, rows contiguous
/// * `im` - Imaginary parts, same layout
/// * `n` - Row length, a power of two; comptime
/// * `log2n` - `log2(n)`; comptime
/// * `forward` - `true` for the forward transform (`exp(-i...)`); comptime
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row index across the batch
/// * `UNIT_POS_X` -> butterfly pair
#[cube(launch_unchecked)]
pub fn fft_rows<F: Float + CubeElement>(
    re: &mut Tensor<F>,
    im: &mut Tensor<F>,
    #[comptime] n: usize,
    #[comptime] log2n: usize,
    #[comptime] forward: bool,
) {
    let half = n / 2;
    let mut s_re = SharedMemory::<F>::new(n);
    let mut s_im = SharedMemory::<F>::new(n);
    let mut tw_re = SharedMemory::<F>::new(half);
    let mut tw_im = SharedMemory::<F>::new(half);

    let row = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let local = UNIT_POS_X as usize;
    let base = row * n;

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

    s_re[br_a] = re[base + local];
    s_im[br_a] = im[base + local];
    s_re[br_b] = re[base + local + half];
    s_im[br_b] = im[base + local + half];

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

/// Transpose real and imaginary parts of a batch of `n x n` matrices.
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
/// * `n` - Matrix side, a multiple of 32; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_X, CUBE_POS_Y)` -> tile column and row, `CUBE_POS_Z` -> matrix
/// * `(UNIT_POS_X, UNIT_POS_Y)` -> column and first row inside the tile
#[cube(launch_unchecked)]
pub fn transpose_paired<F: Float + CubeElement>(
    in_re: &Tensor<F>,
    in_im: &Tensor<F>,
    out_re: &mut Tensor<F>,
    out_im: &mut Tensor<F>,
    #[comptime] n: usize,
) {
    let tile = TRANSPOSE_TILE;
    let stride = TRANSPOSE_TILE + 1;
    let rows_step = TRANSPOSE_TILE / TRANSPOSE_ROWS_PER_UNIT;
    let mut s_re = SharedMemory::<F>::new(TRANSPOSE_TILE * (TRANSPOSE_TILE + 1));
    let mut s_im = SharedMemory::<F>::new(TRANSPOSE_TILE * (TRANSPOSE_TILE + 1));

    let off = CUBE_POS_Z as usize * n * n;
    let tr = CUBE_POS_Y as usize;
    let tc = CUBE_POS_X as usize;
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
    /// One cube per row across the batch
    cubes_rows: CubeCount,
    /// One cube per `32 x 32` tile per matrix
    cubes_tiles: CubeCount,
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
        let cubes_rows = checked_cube_count("fft_rows", gx, gy, 1, limits)?;
        let tiles = (n / TRANSPOSE_TILE) as u32;
        let cubes_tiles =
            checked_cube_count("transpose_paired", tiles, tiles, batch as u32, limits)?;

        Ok(Self {
            n,
            batch,
            cubes_rows,
            cubes_tiles,
        })
    }

    /// Run the batched 2D FFT in place on `(re, im)`.
    ///
    /// `(tmp_re, tmp_im)` receive the intermediate transpose and are left in
    /// an unspecified state. All four buffers hold at least `batch * n * n`
    /// elements.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `re` - Real parts, overwritten with the transform
    /// * `im` - Imaginary parts, overwritten with the transform
    /// * `tmp_re` - Scratch
    /// * `tmp_im` - Scratch
    /// * `forward` - `true` for the forward transform
    pub fn execute<R: Runtime, F: ManifoldsFloatGpu>(
        &self,
        client: &ComputeClient<R>,
        re: &GpuTensor<R, F>,
        im: &GpuTensor<R, F>,
        tmp_re: &GpuTensor<R, F>,
        tmp_im: &GpuTensor<R, F>,
        forward: bool,
    ) {
        self.rows(client, re, im, forward);
        self.transpose(client, re, im, tmp_re, tmp_im);
        self.rows(client, tmp_re, tmp_im, forward);
        self.transpose(client, tmp_re, tmp_im, re, im);
    }

    /// Launch the row pass.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `re` - Real parts, transformed in place
    /// * `im` - Imaginary parts, transformed in place
    /// * `forward` - Transform direction
    fn rows<R: Runtime, F: ManifoldsFloatGpu>(
        &self,
        client: &ComputeClient<R>,
        re: &GpuTensor<R, F>,
        im: &GpuTensor<R, F>,
        forward: bool,
    ) {
        unsafe {
            fft_rows::launch_unchecked::<F, R>(
                client,
                self.cubes_rows.clone(),
                CubeDim::new_1d((self.n / 2) as u32),
                re.into_tensor_arg(),
                im.into_tensor_arg(),
                self.n,
                self.n.trailing_zeros() as usize,
                forward,
            );
        }
    }

    /// Launch the paired transpose `src -> dst`.
    ///
    /// ### Params
    ///
    /// * `client` - CubeCL compute client
    /// * `src_re` - Source real parts
    /// * `src_im` - Source imaginary parts
    /// * `dst_re` - Destination real parts
    /// * `dst_im` - Destination imaginary parts
    fn transpose<R: Runtime, F: ManifoldsFloatGpu>(
        &self,
        client: &ComputeClient<R>,
        src_re: &GpuTensor<R, F>,
        src_im: &GpuTensor<R, F>,
        dst_re: &GpuTensor<R, F>,
        dst_im: &GpuTensor<R, F>,
    ) {
        unsafe {
            transpose_paired::launch_unchecked::<F, R>(
                client,
                self.cubes_tiles.clone(),
                CubeDim::new_2d(
                    TRANSPOSE_TILE as u32,
                    (TRANSPOSE_TILE / TRANSPOSE_ROWS_PER_UNIT) as u32,
                ),
                src_re.into_tensor_arg(),
                src_im.into_tensor_arg(),
                dst_re.into_tensor_arg(),
                dst_im.into_tensor_arg(),
                self.n,
            );
        }
    }
}

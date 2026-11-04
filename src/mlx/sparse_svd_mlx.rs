//! Randomised SVD of a symmetric sparse matrix on MLX, for PHATE's spectral
//! landmark selection.
//!
//! Symmetry means one uploaded CSR serves both `A X` and `A^T X`. CholeskyQR2
//! orthonormalises the sketch: the Gram matrix and `Q = Y R^-1` are MLX GEMMs,
//! the factorisation of the small Gram matrix runs on the host in `f64`.
//!
//! ### Threading of the SpMM
//!
//! One SIMD group per CSR row. The lanes load 32 consecutive non-zeros of the
//! row at a time, then walk them `32 / L` at a time, each `(j, v)` handed to its
//! `L` lanes by `simd_shuffle`. Within such a sub-group lane `l` owns the float4
//! column blocks `l, l + L, ...`, read as unaligned `packed_float4`, so the read
//! of `X[j, :]` stays coalesced and the accumulators stay in registers. The
//! sub-groups' partial rows are summed by `simd_shuffle_xor` at the end.
//!
//! ### References
//!
//! Halko et al., SIAM Review, 2011; Fukaya et al., ScalA, 2014

use faer::Mat;

use super::ffi::*;
use crate::data::structures::CompressedSparseData;
use crate::prelude::*;
use crate::utils::math::{r_inverse_row_major, right_vectors_from_sketch, sketch_row_major_f32};

///////////////
// Constants //
///////////////

/// SIMD-group width on Apple GPUs; one SIMD group owns one CSR row.
const SIMD_WIDTH: usize = 32;

/// Cap on the lanes that share one non-zero. More lanes per non-zero means fewer
/// non-zeros per step and lanes left idle on the last float4 blocks.
const MAX_LANES_PER_NNZ: usize = 16;

/// Threads per threadgroup, i.e. eight rows per threadgroup.
const SPMM_THREADGROUP: i32 = 256;

/// Metal body of the SpMM, see the module doc. Inputs: `indptr` `[rows + 1]` u32,
/// `indices` `[nnz]` u32, `values` `[nnz]` f32, `x` `[rows, S]` f32, `n_rows`
/// `[1]` u32. Output: `y` `[rows, S]`. Template ints: `S` (output width), `L`
/// (lanes per non-zero) and `C4` (float4 blocks per lane).
const SPMM_SOURCE: &str = r#"
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.x / SIMD_WIDTH;
    if (row >= n_rows[0]) {
        return;
    }
    const uint G = SIMD_WIDTH / uint(L);
    uint sub = lane / uint(L);
    uint sl = lane % uint(L);
    uint lo = indptr[row];
    uint hi = indptr[row + 1];
    float4 acc[C4];
    for (uint c = 0; c < uint(C4); ++c) {
        acc[c] = float4(0.0f);
    }
    for (uint base = lo; base < hi; base += SIMD_WIDTH) {
        uint e = base + lane;
        uint j = 0;
        float v = 0.0f;
        if (e < hi) {
            j = indices[e];
            v = values[e];
        }
        // every lane stays in the loop so the shuffles never read an
        // inactive lane; past the end of the row `v` is zero
        uint cnt = min(SIMD_WIDTH, hi - base);
        for (uint t0 = 0; t0 < cnt; t0 += G) {
            uint t = t0 + sub;
            uint jt = simd_shuffle(j, ushort(t));
            float vt = simd_shuffle(v, ushort(t));
            const device float* xr = x + size_t(jt) * uint(S);
            for (uint c = 0; c < uint(C4); ++c) {
                uint col = (sl + c * uint(L)) * 4;
                if (col + 4 <= uint(S)) {
                    acc[c] += vt * float4(*(const device packed_float4*)(xr + col));
                } else {
                    for (uint q = 0; col + q < uint(S); ++q) {
                        acc[c][q] += vt * xr[col + q];
                    }
                }
            }
        }
    }
    for (uint c = 0; c < uint(C4); ++c) {
        for (uint off = uint(L); off < SIMD_WIDTH; off *= 2) {
            acc[c] += simd_shuffle_xor(acc[c], ushort(off));
        }
    }
    if (sub != 0) {
        return;
    }
    size_t yb = size_t(row) * uint(S);
    for (uint c = 0; c < uint(C4); ++c) {
        uint col = (sl + c * uint(L)) * 4;
        if (col + 4 <= uint(S)) {
            *(device packed_float4*)(y + yb + col) = packed_float4(acc[c]);
        } else {
            for (uint q = 0; col + q < uint(S); ++q) {
                y[yb + col + q] = acc[c][q];
            }
        }
    }
"#;

/////////////
// Helpers //
/////////////

/// The symmetric matrix on the device and the kernel multiplying by it.
struct SpmmOperator {
    /// The SpMM kernel
    kernel: MetalKernel,
    /// Row pointers `[n + 1]`
    indptr: Array,
    /// Column indices `[nnz]`
    indices: Array,
    /// Values `[nnz]`
    values: Array,
    /// Row count as a kernel input `[1]`
    n_rows: Array,
    /// Number of rows
    n: usize,
    /// Output width
    s: usize,
}

impl SpmmOperator {
    /// Queue `A * x`.
    ///
    /// ### Params
    ///
    /// * `x` - Dense right operand `[n, s]`, row-major
    /// * `st` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) product `[n, s]`
    fn apply(&self, x: &Array, st: &Stream) -> Result<Array, ManifoldsError> {
        let rows = self.n as i32;
        // `L` is the smallest power of two whose float4 blocks cover `S`.
        let blocks = self.s.div_ceil(4).max(1);
        let lanes = blocks.next_power_of_two().min(MAX_LANES_PER_NNZ);
        let mut out = self.kernel.apply(
            &[&self.indptr, &self.indices, &self.values, x, &self.n_rows],
            &[OutputSpec {
                shape: &[rows, self.s as i32],
                dtype: MLX_FLOAT32,
            }],
            [rows * SIMD_WIDTH as i32, 1, 1],
            [SPMM_THREADGROUP, 1, 1],
            &[
                ("S", self.s as i32),
                ("L", lanes as i32),
                ("C4", blocks.div_ceil(lanes) as i32),
            ],
            st,
        )?;
        Ok(out.remove(0))
    }
}

/// One CholeskyQR pass: `Q = Y R^-1` with `R^T R = Y^T Y`. Blocks on the Gram
/// matrix, which the host Cholesky needs.
///
/// ### Params
///
/// * `y` - Tall-skinny input `[n, s]`, row-major
/// * `s` - Number of columns
/// * `st` - Stream to run on
///
/// ### Returns
///
/// The (lazy) `Q`, or `CholeskyQrFailed` if the Gram matrix is not positive
/// definite
fn cholesky_qr_pass(y: &Array, s: usize, st: &Stream) -> Result<Array, ManifoldsError> {
    let g = y.transpose(st)?.matmul(y, st)?;
    eval_all(&[&g], false)?;
    let r_inv = r_inverse_row_major(g.as_f32()?, s)?;
    y.matmul(&Array::from_f32(&r_inv, &[s as i32, s as i32]), st)
}

/// CholeskyQR2: the second pass recovers the precision the first loses to the
/// squared condition number of `Y^T Y`.
///
/// ### Params
///
/// * `y` - Tall-skinny input `[n, s]`, row-major
/// * `s` - Number of columns
/// * `st` - Stream to run on
///
/// ### Returns
///
/// The (lazy) orthonormal `Q`, or `CholeskyQrFailed` if `y` is numerically rank
/// deficient
fn cholesky_qr2(y: &Array, s: usize, st: &Stream) -> Result<Array, ManifoldsError> {
    let q1 = cholesky_qr_pass(y, s, st)?;
    cholesky_qr_pass(&q1, s, st)
}

//////////
// Main //
//////////

/// Right singular vectors of a symmetric sparse matrix via randomised SVD on
/// MLX.
///
/// Draws the same sketch as [`sparse_randomised_svd`] for the same seed, so the
/// two agree up to `f32` rounding.
///
/// ### Params
///
/// * `matrix` - Symmetric matrix in CSR format, `n x n`
/// * `rank` - Target rank
/// * `seed` - Seed for the sketch
/// * `oversampling` - Extra sketch columns
/// * `n_power_iter` - Power iterations
///
/// ### Returns
///
/// The leading `rank` right singular vectors, `n x rank`, in `f64`
///
/// [`sparse_randomised_svd`]: crate::utils::math::sparse_randomised_svd
pub fn symmetric_svd_vectors_mlx<T>(
    matrix: &CompressedSparseData<T>,
    rank: usize,
    seed: u64,
    oversampling: usize,
    n_power_iter: usize,
) -> Result<Mat<f64>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = matrix.shape.0;
    let s = (rank + oversampling).min(n);
    let nnz = matrix.indices.len();

    install_error_handler();
    let st = Stream::default_gpu();

    let indptr: Vec<u32> = matrix.indptr.iter().map(|&x| x as u32).collect();
    let indices: Vec<u32> = matrix.indices.iter().map(|&x| x as u32).collect();
    let values: Vec<f32> = matrix.data.iter().map(|x| x.to_f32().unwrap()).collect();

    let op = SpmmOperator {
        kernel: MetalKernel::new(
            "manifolds_spmm_csr",
            &["indptr", "indices", "values", "x", "n_rows"],
            &["y"],
            &format!("constant constexpr uint SIMD_WIDTH = {SIMD_WIDTH};\n"),
            SPMM_SOURCE,
        ),
        indptr: Array::from_u32(&indptr, &[(n + 1) as i32]),
        indices: Array::from_u32(&indices, &[nnz as i32]),
        values: Array::from_f32(&values, &[nnz as i32]),
        n_rows: Array::from_u32(&[n as u32], &[1]),
        n,
        s,
    };

    let omega = Array::from_f32(&sketch_row_major_f32(n, s, seed), &[n as i32, s as i32]);
    let y = op.apply(&omega, &st)?;
    let mut q = cholesky_qr2(&y, s, &st)?;

    for _ in 0..n_power_iter {
        let z = op.apply(&q, &st)?;
        let y = op.apply(&z, &st)?;
        q = cholesky_qr2(&y, s, &st)?;
    }

    let z = op.apply(&q, &st)?;
    eval_all(&[&z], false)?;
    let v = right_vectors_from_sketch(z.as_f32()?, n, s, rank)?;

    drop((z, q, y, omega, op));
    clear_cache()?;

    Ok(v)
}

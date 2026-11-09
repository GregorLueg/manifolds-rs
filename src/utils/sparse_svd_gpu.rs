//! Randomised SVD of a symmetric sparse matrix on wgpu, for PHATE's spectral
//! landmark selection.
//!
//! Symmetry means one uploaded CSR serves both `A X` and `A^T X`, so there is no
//! transpose kernel. CholeskyQR2 orthonormalises the sketch: the Gram matrix and
//! `Q = Y R^-1` run on the device, the factorisation of the small Gram matrix on
//! the host in `f64`.
//!
//! Runs in `f32` (WGSL has no `f64`). Only the right singular vectors come back,
//! as PHATE reads nothing else.
//!
//! ### References
//!
//! Halko et al., SIAM Review, 2011; Fukaya et al., ScalA, 2014

// The `#[cube]` macro generates undocumented launcher structs and functions.
#![allow(missing_docs)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use faer::Mat;

use crate::data::structures::CompressedSparseData;
use crate::prelude::*;
use crate::utils::math::{r_inverse_row_major, right_vectors_from_sketch, sketch_row_major_f32};

////////////
// Consts //
////////////

/// Output tile edge of the Gram kernel, so a workgroup is `GRAM_TILE^2` threads.
const GRAM_TILE: u32 = 16;

/// Rows staged per shared-memory step in the Gram kernel. Every loaded element
/// feeds this many multiply-adds; wider staging leaves fewer workgroups resident
/// in the shared-memory budget.
const GRAM_ROWS_STEP: u32 = 8;

/// Upper bound on the Gram split-K chunks, which bounds the partials buffer.
const GRAM_MAX_CHUNKS: usize = 64;

/// Minimum rows per Gram chunk. Below this the per-chunk fixed cost and the
/// reduction outweigh the extra parallelism.
const GRAM_MIN_ROWS_PER_CHUNK: usize = 1024;

/// Rows of the left operand per workgroup in [`tall_skinny_mm`].
const TSMM_ROWS: u32 = 8;

/// Output columns per workgroup in [`tall_skinny_mm`].
const TSMM_COLS: u32 = 32;

/// Largest inner dimension the [`fn@tall_skinny_mm`] staging holds, sized so two
/// workgroups stay resident in a 32 KiB shared-memory budget.
pub const TSMM_K_MAX: u32 = 512;

/// Inner-loop unroll of [`tall_skinny_mm`], to keep several loads in flight.
const TSMM_UNROLL: u32 = 4;

/// Workgroup width of [`gram_reduce`].
const GRAM_REDUCE_WG: u32 = 256;

/////////////
// Kernels //
/////////////

/// Plain SpMM: `Y = A * X`.
///
/// ### Params
///
/// * `indptr` - CSR row pointers `[n + 1]`
/// * `indices` - Column indices `[nnz]`
/// * `values` - Values `[nnz]`
/// * `x` - Dense right operand `[n, s]`, row-major
/// * `y` - Dense output `[n, s]`, row-major
/// * `n_rows` - Number of output rows
/// * `s_width` - Output width
/// * `wg_size` - Workgroup size (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> output row
/// * `UNIT_POS_X` -> stride offset over output columns
#[cube(launch_unchecked)]
pub fn spmm_csr_plain<F: Float>(
    indptr: &Tensor<u32>,
    indices: &Tensor<u32>,
    values: &Tensor<F>,
    x: &Tensor<F>,
    y: &mut Tensor<F>,
    n_rows: u32,
    s_width: u32,
    #[comptime] wg_size: u32,
) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if row >= n_rows {
        terminate!();
    }

    let seg_start = indptr[row as usize];
    let seg_end = indptr[(row + 1u32) as usize];

    let mut col = UNIT_POS_X;
    while col < s_width {
        let mut acc = F::new(0.0_f32);
        let mut idx = seg_start;
        while idx < seg_end {
            let j = indices[idx as usize];
            acc += values[idx as usize] * x[j as usize * s_width as usize + col as usize];
            idx += 1u32;
        }
        y[row as usize * s_width as usize + col as usize] = acc;
        col += wg_size;
    }
}

/// Partial `G = Y^T Y` over one chunk of rows.
///
/// Each workgroup owns a `GRAM_TILE x GRAM_TILE` block of `G` and one chunk of
/// rows, and stages `GRAM_ROWS_STEP` rows of the two column slices in shared
/// memory. One thread per output element with a serial reduction over `n` rows
/// has nothing to parallelise over and re-reads the same rows for every element.
///
/// ### Params
///
/// * `y` - Tall-skinny input `[n, s]`, row-major
/// * `partials` - Output `[n_chunks, s, s]`
/// * `n` - Row count
/// * `s` - Column count
/// * `rows_per_chunk` - Rows per chunk; the last chunk is short
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> output tile row, `CUBE_POS_Y` -> output tile column
/// * `CUBE_POS_Z` -> row chunk
/// * `UNIT_POS_X`, `UNIT_POS_Y` -> position within the output tile
#[cube(launch_unchecked)]
pub fn gram_partial<F: Float>(
    y: &Tensor<F>,
    partials: &mut Tensor<F>,
    n: u32,
    s: u32,
    rows_per_chunk: u32,
) {
    let ti = UNIT_POS_X;
    let tj = UNIT_POS_Y;
    let i0 = CUBE_POS_X * GRAM_TILE;
    let j0 = CUBE_POS_Y * GRAM_TILE;
    let chunk = CUBE_POS_Z;

    let r_start = chunk * rows_per_chunk;
    let mut r_end = r_start + rows_per_chunk;
    if r_end > n {
        r_end = n;
    }

    let mut sa = Shared::<[F]>::new_slice((GRAM_ROWS_STEP * GRAM_TILE) as usize);
    let mut sb = Shared::<[F]>::new_slice((GRAM_ROWS_STEP * GRAM_TILE) as usize);

    let lid = tj * GRAM_TILE + ti;
    let n_threads = GRAM_TILE * GRAM_TILE;
    let stage = GRAM_ROWS_STEP * GRAM_TILE;

    let mut acc = F::new(0.0_f32);

    let mut r = r_start;
    while r < r_end {
        // Out-of-range rows and columns are zero-filled so the accumulation
        // below needs no guard.
        let mut li = lid;
        while li < stage {
            let rr = li / GRAM_TILE;
            let cc = li % GRAM_TILE;
            let row = r + rr;
            let in_row = row < r_end;

            if in_row && i0 + cc < s {
                sa[li as usize] = y[(row * s + i0 + cc) as usize];
            } else {
                sa[li as usize] = F::new(0.0_f32);
            }
            if in_row && j0 + cc < s {
                sb[li as usize] = y[(row * s + j0 + cc) as usize];
            } else {
                sb[li as usize] = F::new(0.0_f32);
            }
            li += n_threads;
        }
        sync_cube();

        #[unroll]
        for rr in 0..GRAM_ROWS_STEP {
            acc += sa[(rr * GRAM_TILE + ti) as usize] * sb[(rr * GRAM_TILE + tj) as usize];
        }
        sync_cube();

        r += GRAM_ROWS_STEP;
    }

    let gi = i0 + ti;
    let gj = j0 + tj;
    if gi < s && gj < s {
        partials[(chunk * s * s + gi * s + gj) as usize] = acc;
    }
}

/// Sum the per-chunk Gram contributions into `G`.
///
/// ### Params
///
/// * `partials` - Per-chunk contributions `[n_chunks, s, s]`
/// * `g` - Output Gram matrix `[s, s]`, row-major
/// * `s` - Column count of `Y`
/// * `n_chunks` - Number of row chunks
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> flat index into the `[s, s]` output
#[cube(launch_unchecked)]
pub fn gram_reduce<F: Float>(partials: &Tensor<F>, g: &mut Tensor<F>, s: u32, n_chunks: u32) {
    let idx = ABSOLUTE_POS_X;
    let total = s * s;
    if idx >= total {
        terminate!();
    }

    let mut acc = F::new(0.0_f32);
    let mut c = 0u32;
    while c < n_chunks {
        acc += partials[(c * total + idx) as usize];
        c += 1u32;
    }
    g[idx as usize] = acc;
}

/// `C = A * B` for a tall-skinny `A` and a small `B`.
///
/// Each workgroup owns `TSMM_ROWS` rows and `TSMM_COLS` output columns. The rows
/// of `A` are staged in shared memory once and reused across the column tile;
/// `B` is small enough to stay cache-resident and is read from global memory.
///
/// ### Params
///
/// * `a` - Left operand `[n, k_dim]`, row-major
/// * `b` - Right operand `[k_dim, s]`, row-major
/// * `c` - Output `[n, s]`, row-major
/// * `n` - Rows of `A`
/// * `k_dim` - Inner dimension, at most `TSMM_K_MAX`
/// * `s` - Output column count
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> row block, flattened because
///   `n / TSMM_ROWS` passes the per-dimension dispatch limit at large `n`
/// * `CUBE_POS_Z` -> output column block
/// * `UNIT_POS_X` -> flattened `(row within block, column within block)`
#[cube(launch_unchecked)]
pub fn tall_skinny_mm<F: Float>(
    a: &Tensor<F>,
    b: &Tensor<F>,
    c: &mut Tensor<F>,
    n: u32,
    k_dim: u32,
    s: u32,
) {
    let tid = UNIT_POS_X;
    let tr = tid / TSMM_COLS;
    let tc = tid % TSMM_COLS;
    let n_threads = TSMM_ROWS * TSMM_COLS;

    let blk = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let row0 = blk * TSMM_ROWS;
    let col0 = CUBE_POS_Z * TSMM_COLS;

    let mut sa = Shared::<[F]>::new_slice((TSMM_ROWS * TSMM_K_MAX) as usize);

    let stage = TSMM_ROWS * k_dim;
    let mut li = tid;
    while li < stage {
        let rr = li / k_dim;
        let cc = li % k_dim;
        let row = row0 + rr;
        if row < n {
            sa[li as usize] = a[(row * k_dim + cc) as usize];
        } else {
            sa[li as usize] = F::new(0.0_f32);
        }
        li += n_threads;
    }
    sync_cube();

    let row = row0 + tr;
    let col = col0 + tc;
    if row < n && col < s {
        let abase = tr * k_dim;
        let mut acc = F::new(0.0_f32);

        let mut i = 0u32;
        while i + TSMM_UNROLL <= k_dim {
            #[unroll]
            for u in 0..TSMM_UNROLL {
                acc += sa[(abase + i + u) as usize] * b[((i + u) * s + col) as usize];
            }
            i += TSMM_UNROLL;
        }
        while i < k_dim {
            acc += sa[(abase + i) as usize] * b[(i * s + col) as usize];
            i += 1u32;
        }

        c[(row * s + col) as usize] = acc;
    }
}

/////////////
// Helpers //
/////////////

/// The symmetric matrix and the workspace of one run on the device.
struct SvdWorkspace {
    /// Row pointers `[n + 1]`
    indptr: GpuTensor<u32>,
    /// Column indices `[nnz]`
    indices: GpuTensor<u32>,
    /// Values `[nnz]`
    values: GpuTensor<f32>,
    /// Scratch for the first CholeskyQR pass `[n, s]`
    q1: GpuTensor<f32>,
    /// Gram matrix `[s, s]`
    g: GpuTensor<f32>,
    /// Per-chunk Gram contributions `[n_chunks, s, s]`
    partials: GpuTensor<f32>,
    /// Number of rows
    n: usize,
    /// Sketch width
    s: usize,
}

/// Queue `y = A * x`.
///
/// ### Params
///
/// * `w` - Workspace holding `A`
/// * `x` - Dense right operand `[n, s]`
/// * `y` - Dense output `[n, s]`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CubeclUtilsError` if the grid busts a device limit
fn spmm(
    w: &SvdWorkspace,
    x: &GpuTensor<f32>,
    y: &GpuTensor<f32>,
    client: &Client,
) -> Result<(), ManifoldsError> {
    let limits = GpuLimits::from_client(client);
    let (gx, gy) = grid_2d(w.n as u32, &limits)?;
    let count = checked_cube_count("spmm_csr_plain", gx, gy, 1, &limits)?;

    // The kernel strides its column loop by the workgroup width, so a width
    // below `s` re-streams every row once per extra pass.
    macro_rules! dispatch {
        ($wg:expr) => {
            unsafe {
                spmm_csr_plain::launch_unchecked::<f32>(
                    client,
                    count,
                    CubeDim::new_1d($wg),
                    w.indptr.clone().into_tensor_arg(),
                    w.indices.clone().into_tensor_arg(),
                    w.values.clone().into_tensor_arg(),
                    x.clone().into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    w.n as u32,
                    w.s as u32,
                    $wg,
                );
            }
        };
    }
    match w.s {
        0..=64 => dispatch!(64),
        65..=128 => dispatch!(128),
        _ => dispatch!(256),
    }
    Ok(())
}

/// One CholeskyQR pass: `output = input * R^-1` with `R^T R = input^T input`.
///
/// ### Params
///
/// * `w` - Workspace with the Gram scratch
/// * `input` - Tall-skinny input `[n, s]`
/// * `output` - Tall-skinny output `[n, s]`, distinct from `input`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CholeskyQrFailed` if the Gram matrix is not positive definite
fn cholesky_qr_pass(
    w: &SvdWorkspace,
    input: &GpuTensor<f32>,
    output: &GpuTensor<f32>,
    client: &Client,
) -> Result<(), ManifoldsError> {
    let (n, s) = (w.n, w.s);
    let n_chunks = n
        .div_ceil(GRAM_MIN_ROWS_PER_CHUNK)
        .clamp(1, GRAM_MAX_CHUNKS);
    let rows_per_chunk = n.div_ceil(n_chunks) as u32;
    let tiles = (s as u32).div_ceil(GRAM_TILE);
    let total = (s * s) as u32;

    let limits = GpuLimits::from_client(client);
    let partial_count = checked_cube_count("gram_partial", tiles, tiles, n_chunks as u32, &limits)?;
    let reduce_count =
        checked_cube_count("gram_reduce", total.div_ceil(GRAM_REDUCE_WG), 1, 1, &limits)?;

    unsafe {
        gram_partial::launch_unchecked::<f32>(
            client,
            partial_count,
            CubeDim::new_2d(GRAM_TILE, GRAM_TILE),
            input.clone().into_tensor_arg(),
            w.partials.clone().into_tensor_arg(),
            n as u32,
            s as u32,
            rows_per_chunk,
        );
        gram_reduce::launch_unchecked::<f32>(
            client,
            reduce_count,
            CubeDim::new_1d(GRAM_REDUCE_WG),
            w.partials.clone().into_tensor_arg(),
            w.g.clone().into_tensor_arg(),
            s as u32,
            n_chunks as u32,
        );
    }

    let g_host = w.g.clone().read(client)?;
    let r_inv = r_inverse_row_major(&g_host, s)?;
    let r_inv_gpu = GpuTensor::<f32>::from_slice(&r_inv, vec![s, s], client)?;

    let (gx, gy) = grid_2d((n as u32).div_ceil(TSMM_ROWS), &limits)?;
    let count = checked_cube_count(
        "tall_skinny_mm",
        gx,
        gy,
        (s as u32).div_ceil(TSMM_COLS),
        &limits,
    )?;
    unsafe {
        tall_skinny_mm::launch_unchecked::<f32>(
            client,
            count,
            CubeDim::new_1d(TSMM_ROWS * TSMM_COLS),
            input.clone().into_tensor_arg(),
            r_inv_gpu.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            n as u32,
            s as u32,
            s as u32,
        );
    }
    Ok(())
}

/// CholeskyQR2: the second pass recovers the precision the first loses to the
/// squared condition number of `Y^T Y`.
///
/// ### Params
///
/// * `w` - Workspace with the scratch buffers
/// * `y` - Tall-skinny input `[n, s]`, untouched
/// * `q` - Orthonormal output `[n, s]`
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, or `CholeskyQrFailed` if `y` is numerically rank deficient
fn cholesky_qr2(
    w: &SvdWorkspace,
    y: &GpuTensor<f32>,
    q: &GpuTensor<f32>,
    client: &Client,
) -> Result<(), ManifoldsError> {
    cholesky_qr_pass(w, y, &w.q1, client)?;
    cholesky_qr_pass(w, &w.q1, q, client)
}

//////////
// Main //
//////////

/// Right singular vectors of a symmetric sparse matrix via randomised SVD on
/// wgpu.
///
/// Draws the same sketch as [`sparse_randomised_svd`] for the same seed, so the
/// two agree up to `f32` rounding.
///
/// ### Params
///
/// * `matrix` - Symmetric matrix in CSR format, `n x n`
/// * `rank` - Target rank
/// * `seed` - Seed for the sketch
/// * `oversampling` - Extra sketch columns. `rank + oversampling` must be at
///   most [`TSMM_K_MAX`]; the caller falls back to the CPU above it.
/// * `n_power_iter` - Power iterations
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The leading `rank` right singular vectors, `n x rank`, in `f64`
///
/// [`sparse_randomised_svd`]: crate::utils::math::sparse_randomised_svd
pub fn symmetric_svd_vectors_gpu<T>(
    matrix: &CompressedSparseData<T>,
    rank: usize,
    seed: u64,
    oversampling: usize,
    n_power_iter: usize,
    client: &Client,
) -> Result<Mat<f64>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = matrix.shape.0;
    let s = (rank + oversampling).min(n);

    let indptr: Vec<u32> = matrix.indptr.iter().map(|&x| x as u32).collect();
    let indices: Vec<u32> = matrix.indices.iter().map(|&x| x as u32).collect();
    let values: Vec<f32> = matrix.data.iter().map(|x| x.to_f32().unwrap()).collect();
    let n_chunks = n
        .div_ceil(GRAM_MIN_ROWS_PER_CHUNK)
        .clamp(1, GRAM_MAX_CHUNKS);

    let w = SvdWorkspace {
        indptr: GpuTensor::from_slice(&indptr, vec![n + 1], client)?,
        indices: GpuTensor::from_slice(&indices, vec![indices.len()], client)?,
        values: GpuTensor::from_slice(&values, vec![values.len()], client)?,
        q1: GpuTensor::empty(vec![n, s], client)?,
        g: GpuTensor::empty(vec![s, s], client)?,
        partials: GpuTensor::empty(vec![n_chunks, s, s], client)?,
        n,
        s,
    };

    let omega =
        GpuTensor::<f32>::from_slice(&sketch_row_major_f32(n, s, seed), vec![n, s], client)?;
    let y = GpuTensor::<f32>::empty(vec![n, s], client)?;
    let q = GpuTensor::<f32>::empty(vec![n, s], client)?;
    let z = GpuTensor::<f32>::empty(vec![n, s], client)?;

    spmm(&w, &omega, &y, client)?;
    cholesky_qr2(&w, &y, &q, client)?;

    for _ in 0..n_power_iter {
        spmm(&w, &q, &z, client)?;
        spmm(&w, &z, &y, client)?;
        cholesky_qr2(&w, &y, &q, client)?;
    }

    spmm(&w, &q, &z, client)?;
    let z_host = z.read(client)?;

    right_vectors_from_sketch(&z_host, n, s, rank)
}

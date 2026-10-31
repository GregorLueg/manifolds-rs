//! Math helper functions

use ann_search_rs::prelude::SimdDistance;
use faer::{Mat, MatMut, MatRef};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardNormal};
use rayon::prelude::*;

use crate::assert_same_len;
use crate::data::structures::*;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Threshold when to go through the sparse path in VNE
const VNE_DENSE_THRESHOLD: usize = 5000;

/// Number of PCs to return on the sparse path
const VNE_SPARSE_RANK: usize = 100;

/// Rows handled per parallel chunk in the Lanczos sparse matrix-vector product.
///
/// Sized to keep a chunk's working set in L2 while still balancing across
/// threads. The value cannot change the result: each output element is an
/// independent sum over its own row, so the partition never reorders an
/// accumulation.
pub(crate) const SPMV_ROW_CHUNK: usize = 512;

/// Vector length below which elementwise and row-parallel work runs serially.
///
/// The spectral path runs a full Lanczos solve per connected component, so on a
/// graph with many small components rayon's dispatch cost would otherwise
/// dominate the arithmetic. Going serial is bit-identical: neither the matvec
/// nor `normalise` reduces across rows.
pub(crate) const PARALLEL_VEC_MIN: usize = 4096;

////////////////////
// Randomised SVD //
////////////////////

/// Structure for randomised SVD results
#[derive(Clone, Debug)]
pub struct RandomSvdResults<T> {
    /// Matrix u of the SVD decomposition
    pub u: faer::Mat<T>,
    /// Matrix v of the SVD decomposition
    pub v: faer::Mat<T>,
    /// Eigen vectors of the SVD decomposition
    pub s: Vec<T>,
}

/// Randomised SVD
///
/// ### Params
///
/// * `x` - The matrix on which to apply the randomised SVD.
/// * `rank` - The target rank of the approximation (number of singular values,
///   vectors to compute).
/// * `seed` - Random seed for reproducible results.
/// * `oversampling` - Additional samples beyond the target rank to improve
///   accuracy. Defaults to 10 if not specified.
/// * `n_power_iter` - Number of power iterations to perform for better
///   approximation quality. More iterations generally improve accuracy but
///   increase computation time. Defaults to 2 if not specified.
///
/// ### Returns
///
/// The randomised SVD results in form of `RandomSvdResults`.
///
/// ### Algorithm Details
///
/// 1. Generate a random Gaussian matrix Ω of size n × (rank + oversampling)
/// 2. Compute `Y = X * Ω` to capture the range of X
/// 3. Orthogonalize Y using QR decomposition to get Q
/// 4. Apply power iterations: for each iteration, compute `Z = X^T * Q`, then
///    `Q = QR(X * Z)`
/// 5. Form B = Q^T * X and compute its SVD
/// 6. Reconstruct the final `SVD: U = Q * U_B, V = V_B, S = S_B`
pub fn randomised_svd<T>(
    x: MatRef<T>,
    rank: usize,
    seed: usize,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
) -> Result<RandomSvdResults<T>, ManifoldsError>
where
    T: ManifoldsFloat,
    StandardNormal: Distribution<T>,
{
    let ncol = x.ncols();
    let nrow = x.nrows();

    // Add adaptive oversampling for very small ranks
    let os = oversampling.unwrap_or({
        if rank < 10 {
            rank // 100% oversampling for small ranks
        } else {
            10
        }
    });
    let sample_size = (rank + os).min(ncol.min(nrow));
    let n_iter = n_power_iter.unwrap_or(2);

    // Create a random matrix
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let normal = Normal::new(T::from(0.0).unwrap(), T::from(1.0).unwrap()).unwrap();
    let omega = Mat::from_fn(ncol, sample_size, |_, _| normal.sample(&mut rng));

    // Multiply random matrix with original and use QR composition to get
    // low rank approximation of x
    let y = x * omega;

    let mut q = y.qr().compute_thin_Q();
    for _ in 0..n_iter {
        let z = x.transpose() * q;
        q = (x * z).qr().compute_thin_Q();
    }

    // Perform the SVD on the low-rank approximation
    let b = q.transpose() * x;
    let svd = b.thin_svd().map_err(|_| ManifoldsError::FaerSvdError)?;

    Ok(RandomSvdResults {
        u: q * svd.U(),
        v: svd.V().cloned(), // Use clone instead of manual copying
        s: svd.S().column_vector().iter().copied().collect(),
    })
}

///////////////////////////
// Sparse randomised SVD //
///////////////////////////

/// Sparse randomised SVD
///
/// Calculates the sparse randomised SVD.
///
/// ### Params
///
/// * `matrix` - Sparse matrix (CSR or CSC)
/// * `rank` - Target rank
/// * `seed` - For reproducibility
/// * `oversampling` - Additional samples (default 10)
/// * `n_power_iter` - Power iterations for accuracy (default 2)
///
/// ### Returns
///
/// The RandomSvdResults
pub fn sparse_randomised_svd<T>(
    matrix: &CompressedSparseData<T>,
    rank: usize,
    seed: u64,
    oversampling: Option<usize>,
    n_power_iter: Option<usize>,
) -> Result<RandomSvdResults<T>, ManifoldsError>
where
    T: Into<T> + ManifoldsFloat,
{
    let (n, m) = matrix.shape;
    let os = oversampling.unwrap_or(10);
    let sample_size = (rank + os).min(m).min(n);
    let n_iter = n_power_iter.unwrap_or(2);
    let ncols = sample_size;

    // keep both representations... this makes the matrix math faster
    let (csr, csc);
    let csr_owned;
    let csc_owned;

    match matrix.cs_type {
        CompressedSparseFormat::Csr => {
            csr = matrix;
            csc_owned = matrix.transform();
            csc = &csc_owned;
        }
        CompressedSparseFormat::Csc => {
            csc = matrix;
            csr_owned = matrix.transform();
            csr = &csr_owned;
        }
    };

    // borrow slices to prevent allocations
    let data_csr = csr.data.as_slice();
    let data_csc = csc.data.as_slice();

    let spmm = |sparse: &CompressedSparseData<T>, data: &[T], x_flat: &[T], y_flat: &mut [T]| {
        let n_outer = sparse.indptr.len() - 1;

        y_flat
            .par_chunks_exact_mut(ncols)
            .take(n_outer)
            .enumerate()
            .for_each(|(i, y_row)| {
                y_row.fill(T::zero());

                for idx in sparse.indptr[i]..sparse.indptr[i + 1] {
                    let j = sparse.indices[idx];
                    let a_val = data[idx];
                    let x_row = &x_flat[j * ncols..(j + 1) * ncols];

                    for col in 0..ncols {
                        y_row[col] += a_val * x_row[col];
                    }
                }
            });
    };

    let to_row_major = |mat: MatRef<T>| -> Vec<T> {
        let rows = mat.nrows();
        let mut flat = vec![T::zero(); rows * ncols];
        flat.par_chunks_exact_mut(ncols)
            .enumerate()
            .for_each(|(i, row_slice)| {
                for col in 0..ncols {
                    row_slice[col] = mat[(i, col)];
                }
            });
        flat
    };

    let from_row_major = |flat: &[T], mut mat: MatMut<T>| {
        let rows = mat.nrows();
        for i in 0..rows {
            let row_slice = &flat[i * ncols..(i + 1) * ncols];
            for col in 0..ncols {
                mat[(i, col)] = row_slice[col];
            }
        }
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let omega = Mat::from_fn(m, sample_size, |_, _| {
        T::from_f64(normal.sample(&mut rng)).unwrap()
    });

    let mut x_flat = to_row_major(omega.as_ref());
    let mut y_flat = vec![T::zero(); n * ncols];
    let mut z_flat = vec![T::zero(); m * ncols];

    // Y = A * Omega
    spmm(csr, data_csr, &x_flat, &mut y_flat);

    let mut y = Mat::<T>::zeros(n, sample_size);
    from_row_major(&y_flat, y.as_mut());
    let mut q = y.qr().compute_thin_Q();

    // Power Iterations
    for _ in 0..n_iter {
        // Z = A^T * Q  --> Notice we use `csc` here!
        x_flat = to_row_major(q.as_ref());
        spmm(csc, data_csc, &x_flat, &mut z_flat);

        // Y_new = A * Z
        spmm(csr, data_csr, &z_flat, &mut y_flat);

        from_row_major(&y_flat, y.as_mut());
        q = y.qr().compute_thin_Q();
    }

    // B_t = A^T * Q
    x_flat = to_row_major(q.as_ref());
    let mut b_t_flat = vec![T::zero(); m * ncols];
    spmm(csc, data_csc, &x_flat, &mut b_t_flat);

    let mut b_t = Mat::<T>::zeros(m, sample_size);
    from_row_major(&b_t_flat, b_t.as_mut());

    let b = b_t.transpose().to_owned();

    // Final SVD on the small B matrix
    let svd = b.thin_svd().map_err(|_| ManifoldsError::FaerSvdError)?;
    let u = (&q * svd.U()).get(.., ..rank).to_owned();
    let s: Vec<T> = svd.S().column_vector().iter().copied().take(rank).collect();
    let v = svd.V().get(.., ..rank).to_owned();

    Ok(RandomSvdResults { u, s, v })
}

/////////////////////////////////////
// Lanczos Eigenvalue calculations //
/////////////////////////////////////

/////////////////////
// Lanczos: consts //
/////////////////////

/// Ritz vectors retained across a restart, above the requested pair count.
const LANCZOS_RESTART_GUARD: usize = 5;

/// Norm ratio below which a second Gram-Schmidt pass is run.
///
/// The DGKS criterion. A drop past `1 / sqrt(2)` means the projection cancelled
/// at least half the vector, which is where the round-off it leaves behind
/// stops being negligible.
const LANCZOS_ORTHO_REFINE_RATIO: f64 = 0.707;

/// Lanczos breakdown threshold, relative to the running `||A||` estimate.
///
/// The attainable floor on the residual norm after two Gram-Schmidt passes is
/// `~eps * sqrt(n) * ||A||`, so this sits above it while still only firing on
/// a genuinely invariant subspace.
const LANCZOS_BREAKDOWN_RATIO: f64 = 1e-12;

/// Rows per partial sum in the reductions over basis-length vectors.
///
/// Fixed rather than derived from the thread count, so the summation order and
/// with it the result are the same on any machine. Not tuned.
const REDUCE_CHUNK: usize = 8192;

//////////////////////
// Lanczos: structs //
//////////////////////

/// Parameters for the thick-restarted Lanczos eigensolver.
#[derive(Clone, Copy, Debug)]
pub struct LanczosParams {
    /// Krylov basis vectors held at once. `None` derives it from the requested
    /// component count as `max(2k + 10, k + 20)`. Together with `n` this sets
    /// the solver's memory: `basis_size * n` f64.
    pub basis_size: Option<usize>,
    /// Maximum restart cycles. Each cycle costs one basis worth of
    /// matrix-vector products plus the orthogonalisation against the basis.
    pub max_restarts: usize,
    /// Relative residual tolerance. A Ritz pair counts as converged once
    /// `||A x - lambda x||` drops below `tol * ||A||`.
    pub tol: f64,
}

impl LanczosParams {
    /// Generate new Lanczos parameters.
    ///
    /// ### Params
    ///
    /// * `basis_size` - Krylov basis vectors held at once; `None` derives it
    ///   from the requested component count.
    /// * `max_restarts` - Maximum restart cycles.
    /// * `tol` - Relative residual tolerance.
    ///
    /// ### Returns
    ///
    /// Self
    pub fn new(basis_size: Option<usize>, max_restarts: usize, tol: f64) -> Self {
        Self {
            basis_size,
            max_restarts,
            tol,
        }
    }

    /// Resolve the basis size for a given component count.
    ///
    /// ### Params
    ///
    /// * `n_components` - Eigenpairs requested, already capped at `n`.
    /// * `n` - Matrix dimension, which caps the basis.
    ///
    /// ### Returns
    ///
    /// The number of basis vectors to hold, at least one and never above `n`.
    fn resolve_basis(&self, n_components: usize, n: usize) -> usize {
        self.basis_size
            .unwrap_or_else(|| (n_components * 2 + 10).max(n_components + 20))
            .max(n_components + 1)
            .min(n)
            .max(1)
    }
}

impl Default for LanczosParams {
    /// Derived basis size, 64 restarts and a relative tolerance of `1e-8`.
    fn default() -> Self {
        Self::new(None, 64, 1e-8)
    }
}

/// Ritz pairs from [`lanczos_thick_restart`].
#[derive(Clone, Debug)]
pub struct RitzPairs {
    /// Ritz values sorted descending.
    pub values: Vec<f64>,
    /// Unit-norm Ritz vectors, column-major: vector `c` is
    /// `vectors[c * n..(c + 1) * n]`.
    pub vectors: Vec<f64>,
    /// Whether the largest residual over the returned pairs met
    /// `tol * ||A||`.
    pub converged: bool,
}

//////////////////////
// Lanczos: helpers //
//////////////////////

/// Dot product of two vectors
///
/// Below `PARALLEL_VEC_MIN` a single SIMD pass. Above it, partial sums over
/// fixed `REDUCE_CHUNK` blocks run in parallel and are added in block order,
/// so the result does not depend on the thread count.
///
/// ### Params
///
/// * `a` - Vector a
/// * `b` - Vector b
///
/// ### Returns
///
/// Dot product of the two vectors
fn dot<T>(a: &[T], b: &[T]) -> T
where
    T: ManifoldsFloat,
{
    assert_same_len!(a, b);
    if a.len() < PARALLEL_VEC_MIN {
        return T::dot_simd(a, b);
    }
    let partials: Vec<T> = a
        .par_chunks(REDUCE_CHUNK)
        .zip(b.par_chunks(REDUCE_CHUNK))
        .map(|(x, y)| T::dot_simd(x, y))
        .collect();
    partials.into_iter().fold(T::zero(), |acc, p| acc + p)
}

/// Euclidean norm of a vector
///
/// ### Params
///
/// * `v` - Initial vector
///
/// ### Returns
///
/// `||v||_2`
fn norm<T>(v: &[T]) -> T
where
    T: ManifoldsFloat,
{
    dot(v, v).sqrt()
}

/// Helper function to normalise a vector
///
/// ### Params
///
/// * `v` - Mutable reference of the vector to normalise
fn normalise<T>(v: &mut [T])
where
    T: ManifoldsFloat,
{
    let n = norm(v);
    if v.len() < PARALLEL_VEC_MIN {
        v.iter_mut().for_each(|x| *x /= n);
    } else {
        v.par_iter_mut().for_each(|x| *x /= n);
    }
}

/// Project a vector onto the leading columns of the basis: `out = V^T v`.
///
/// Partial products over fixed `REDUCE_CHUNK` row blocks, summed in block
/// order, so the result is independent of the thread count. Each block of `v`
/// stays in cache across the `width` basis columns.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Leading columns to project against.
/// * `v` - Vector of length `n`.
/// * `out` - Receives `V^T v`; must be exactly `width` long.
fn basis_project(basis: &[f64], n: usize, width: usize, v: &[f64], out: &mut [f64]) {
    let block = |(c, v_chunk): (usize, &[f64])| -> Vec<f64> {
        let start = c * REDUCE_CHUNK;
        (0..width)
            .map(|k| {
                let col = &basis[k * n + start..k * n + start + v_chunk.len()];
                <f64 as SimdDistance>::dot_simd(col, v_chunk)
            })
            .collect()
    };

    let partials: Vec<Vec<f64>> = if n < PARALLEL_VEC_MIN {
        v.chunks(REDUCE_CHUNK).enumerate().map(block).collect()
    } else {
        v.par_chunks(REDUCE_CHUNK).enumerate().map(block).collect()
    };

    out.fill(0.0);
    for p in &partials {
        for (o, &x) in out.iter_mut().zip(p) {
            *o += x;
        }
    }
}

/// Accumulate a basis expansion into a vector: `v += scale * V c`.
///
/// Elementwise over rows with the columns added in ascending order, so the
/// row partition never changes the result.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `coeffs` - Coefficients, one per leading basis column used.
/// * `scale` - Factor applied to the expansion (`-1` subtracts it).
/// * `v` - Vector of length `n`, updated in place.
fn basis_axpy(basis: &[f64], n: usize, coeffs: &[f64], scale: f64, v: &mut [f64]) {
    let block = |(c, v_chunk): (usize, &mut [f64])| {
        let start = c * REDUCE_CHUNK;
        for (k, &coeff) in coeffs.iter().enumerate() {
            let a = scale * coeff;
            let col = &basis[k * n + start..k * n + start + v_chunk.len()];
            for (x, &b) in v_chunk.iter_mut().zip(col) {
                *x += a * b;
            }
        }
    };

    if n < PARALLEL_VEC_MIN {
        v.chunks_mut(REDUCE_CHUNK).enumerate().for_each(block);
    } else {
        v.par_chunks_mut(REDUCE_CHUNK).enumerate().for_each(block);
    }
}

/// Expand Ritz coefficient columns back into the full space.
///
/// ### Params
///
/// * `basis` - Column-major basis buffer with `n` rows.
/// * `n` - Length of a basis vector.
/// * `width` - Basis columns spanned by the coefficients.
/// * `coeffs` - Rayleigh-Ritz eigenvectors, column-wise.
/// * `n_out` - Leading coefficient columns to expand.
/// * `out` - Receives the `n_out` expanded vectors, column-major. Overwritten.
fn basis_expand(
    basis: &[f64],
    n: usize,
    width: usize,
    coeffs: MatRef<f64>,
    n_out: usize,
    out: &mut [f64],
) {
    for (o, column) in out.chunks_exact_mut(n).take(n_out).enumerate() {
        let c: Vec<f64> = (0..width).map(|k| coeffs[(k, o)]).collect();
        column.fill(0.0);
        basis_axpy(basis, n, &c, 1.0, column);
    }
}

/// Eigen-decompose the leading `size` block of a symmetric matrix, descending.
///
/// ### Params
///
/// * `h` - Symmetric matrix; only the leading `size` by `size` block is read.
/// * `size` - Size of the block to decompose.
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)` sorted by descending eigenvalue, the
/// eigenvectors held column-wise.
fn symmetric_eigen_descending(
    h: MatRef<f64>,
    size: usize,
) -> Result<(Vec<f64>, Mat<f64>), ManifoldsError> {
    let block = Mat::<f64>::from_fn(size, size, |i, j| h[(i, j)]);
    let eig = block
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| ManifoldsError::FaerEigenError)?;

    let raw_values = eig.S().column_vector();
    let raw_vectors = eig.U();

    let mut order: Vec<usize> = (0..size).collect();
    order.sort_by(|&a, &b| raw_values[b].total_cmp(&raw_values[a]));

    let values: Vec<f64> = order.iter().map(|&i| raw_values[i]).collect();
    let vectors = Mat::<f64>::from_fn(size, size, |i, j| raw_vectors[(i, order[j])]);

    Ok((values, vectors))
}

/// Largest eigenpairs of a symmetric operator via thick-restarted Lanczos.
///
/// Each cycle builds a Krylov basis of at most [`LanczosParams::basis_size`]
/// vectors with full reorthogonalisation, does a Rayleigh-Ritz extraction and
/// restarts from the best Ritz vectors plus the residual direction. The basis
/// stays bounded while the iteration keeps going, so memory is
/// `basis_size * n` f64 regardless of how many matvecs convergence takes.
///
/// The projected matrix is accumulated explicitly as `H[i][j] = <v_i, A v_j>`
/// rather than as a tridiagonal recurrence: after a thick restart the leading
/// basis vectors are Ritz vectors, so the projection is arrow-shaped.
/// Orthogonalisation is classical Gram-Schmidt with a DGKS-conditional second
/// pass. Convergence uses the Arnoldi residual `|beta_m| * |y[m-1]|`, tested
/// against `tol * ||A||`.
///
/// Every reduction runs over fixed row blocks, so a seed reproduces the run
/// bit for bit on any thread count, provided `matvec` does too.
///
/// ### Params
///
/// * `n` - Dimension of the operator.
/// * `n_components` - Number of eigenpairs to compute. Capped at `n`.
/// * `seed` - Seed for the starting vector.
/// * `params` - [`LanczosParams`] for basis size, restarts and tolerance.
/// * `matvec` - Symmetric operator, writes `A x` into `y`.
///
/// ### Returns
///
/// [`RitzPairs`] with at most `min(n_components, n)` pairs, fewer if the
/// iteration hit an invariant subspace first.
///
/// ### References
///
/// Wu & Simon, SIAM J. Matrix Anal. Appl., 2000
pub fn lanczos_thick_restart<M>(
    n: usize,
    n_components: usize,
    seed: u64,
    params: LanczosParams,
    mut matvec: M,
) -> Result<RitzPairs, ManifoldsError>
where
    M: FnMut(&[f64], &mut [f64]),
{
    if n_components == 0 || n == 0 {
        return Err(ManifoldsError::NoData);
    }

    let n_keep = n_components.min(n);
    let m = params.resolve_basis(n_keep, n);

    let mut basis = vec![0.0f64; m * n];
    let mut h = Mat::<f64>::zeros(m, m);
    let mut w = vec![0.0f64; n];
    let mut coeffs = vec![0.0f64; m];

    let mut rng = StdRng::seed_from_u64(seed);
    for x in basis[..n].iter_mut() {
        *x = rng.random::<f64>() - 0.5;
    }
    normalise(&mut basis[..n]);

    // basis vectors carried over from the previous cycle, zero on the first
    let mut locked = 0usize;
    let mut ritz_values: Vec<f64> = Vec::new();
    let mut ritz_coeffs = Mat::<f64>::zeros(0, 0);
    let mut m_final = 0usize;

    // lower bound on ||A||_2, refined as the iteration sees more of the matrix
    let mut norm_estimate = 0.0f64;
    let mut converged = false;
    let budget = params.max_restarts.max(1);

    for cycle in 0..budget {
        let mut m_eff = m;
        let mut last_beta = 0.0f64;

        for j in locked..m {
            matvec(&basis[j * n..(j + 1) * n], &mut w);
            let width = j + 1;

            // taken before any subtraction, so `h` is the exact projection;
            // they double as the first Gram-Schmidt coefficients
            basis_project(&basis, n, width, &w, &mut coeffs[..width]);
            for i in 0..width {
                h[(i, j)] = coeffs[i];
                h[(j, i)] = coeffs[i];
            }

            let before = norm(&w);
            norm_estimate = norm_estimate.max(before);

            basis_axpy(&basis, n, &coeffs[..width], -1.0, &mut w);
            last_beta = norm(&w);

            if last_beta < LANCZOS_ORTHO_REFINE_RATIO * before {
                basis_project(&basis, n, width, &w, &mut coeffs[..width]);
                basis_axpy(&basis, n, &coeffs[..width], -1.0, &mut w);
                last_beta = norm(&w);
            }

            // the basis spans an invariant subspace: its Ritz pairs are exact
            if last_beta <= LANCZOS_BREAKDOWN_RATIO * norm_estimate {
                m_eff = width;
                last_beta = 0.0;
                break;
            }

            if width < m {
                let inv = 1.0 / last_beta;
                let next = &mut basis[width * n..(width + 1) * n];
                for (dst, &src) in next.iter_mut().zip(w.iter()) {
                    *dst = src * inv;
                }
            }
        }

        let (values, vectors) = symmetric_eigen_descending(h.as_ref(), m_eff)?;

        let n_out = n_keep.min(m_eff);
        let residual = (0..n_out).fold(0.0f64, |acc, i| {
            acc.max(last_beta * vectors[(m_eff - 1, i)].abs())
        });
        norm_estimate = norm_estimate
            .max(values[0].abs())
            .max(values[m_eff - 1].abs());
        converged = residual <= params.tol * norm_estimate;

        ritz_values = values;
        ritz_coeffs = vectors;
        m_final = m_eff;

        if converged || m_eff < m || cycle + 1 == budget {
            break;
        }

        // Thick restart: keep the best Ritz vectors, then hand the residual
        // direction over as the next basis vector. Never retain more than half
        // the basis, or a cycle buys too few new Krylov vectors to pay for it.
        let keep = (n_keep + LANCZOS_RESTART_GUARD).min(m / 2).max(1);
        let mut restarted = vec![0.0f64; keep * n];
        basis_expand(&basis, n, m_eff, ritz_coeffs.as_ref(), keep, &mut restarted);

        let inv = 1.0 / last_beta;
        w.iter_mut().for_each(|x| *x *= inv);

        h.fill(0.0);
        basis[..keep * n].copy_from_slice(&restarted);
        for (i, &value) in ritz_values.iter().take(keep).enumerate() {
            h[(i, i)] = value;
        }
        basis[keep * n..(keep + 1) * n].copy_from_slice(&w);
        locked = keep;
    }

    let n_out = n_keep.min(m_final);
    let mut ritz = vec![0.0f64; n_out * n];
    basis_expand(&basis, n, m_final, ritz_coeffs.as_ref(), n_out, &mut ritz);
    drop(basis);

    for column in ritz.chunks_exact_mut(n) {
        let scale = norm(column);
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        column.iter_mut().for_each(|v| *v *= inv);
    }

    Ok(RitzPairs {
        values: ritz_values[..n_out].to_vec(),
        vectors: ritz,
        converged,
    })
}

/// Thick-restarted Lanczos on a sparse matrix, optionally negated.
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix. CSC input is transposed once.
/// * `n_components` - Number of eigenpairs to compute
/// * `seed` - For reproducibility
/// * `negate` - Run on `-A`, which turns the largest pairs into the smallest
///
/// ### Returns
///
/// The [`RitzPairs`] of `A` or `-A`
fn sparse_lanczos<T>(
    matrix: &CompressedSparseData<T>,
    n_components: usize,
    seed: u64,
    negate: bool,
) -> Result<RitzPairs, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = matrix.shape.0;
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }

    let csr_owned;
    let csr: &CompressedSparseData<T> = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix,
        CompressedSparseFormat::Csc => {
            csr_owned = matrix.transform();
            &csr_owned
        }
    };
    let (data, indices, indptr) = (
        csr.data.as_slice(),
        csr.indices.as_slice(),
        csr.indptr.as_slice(),
    );
    let sign = if negate { -1.0 } else { 1.0 };

    // `T -> f64` at point of use rather than as an up-front widened copy
    let matvec = |x: &[f64], y: &mut [f64]| {
        let kernel = |base: usize, y_chunk: &mut [f64]| {
            for (offset, y_i) in y_chunk.iter_mut().enumerate() {
                let i = base + offset;
                let mut acc = 0.0;
                for idx in indptr[i]..indptr[i + 1] {
                    acc += data[idx].to_f64().unwrap() * x[indices[idx]];
                }
                *y_i = sign * acc;
            }
        };

        if y.len() < PARALLEL_VEC_MIN {
            kernel(0, y);
        } else {
            y.par_chunks_mut(SPMV_ROW_CHUNK)
                .enumerate()
                .for_each(|(chunk, y_chunk)| kernel(chunk * SPMV_ROW_CHUNK, y_chunk));
        }
    };

    lanczos_thick_restart(n, n_components, seed, LanczosParams::default(), matvec)
}

/// Convert Ritz pairs into the `(eigenvalues, per-point eigenvectors)` layout.
///
/// ### Params
///
/// * `ritz` - Ritz pairs, `n` rows per vector
/// * `n` - Number of points
/// * `sign` - Factor applied to the eigenvalues (`-1` undoes a negation)
///
/// ### Returns
///
/// `(eigenvalues, eigenvectors)` where `eigenvectors[i][j]` is element `i` of
/// eigenvector `j`.
fn ritz_to_rows(ritz: &RitzPairs, n: usize, sign: f64) -> (Vec<f32>, Vec<Vec<f32>>) {
    let n_out = ritz.values.len();
    let evals = ritz.values.iter().map(|&v| (sign * v) as f32).collect();
    let evecs = (0..n)
        .map(|i| (0..n_out).map(|c| ritz.vectors[c * n + i] as f32).collect())
        .collect();
    (evals, evecs)
}

/// Compute smallest eigenvalues and eigenvectors using Lanczos
///
/// Thick-restarted Lanczos on `-A`, see [`lanczos_thick_restart`].
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix
/// * `n_components` - Number of eigenpairs to compute
/// * `seed` - For reproducibility
///
/// ### Returns
///
/// (eigenvalues, eigenvectors) sorted ascending, where `eigenvectors[i][j]` is
/// element i of eigenvector j.
///
/// Both are at most `n_components` wide, and can be SHORTER: an operator with
/// fewer distinct eigenvalues than requested (a clique, a rank-deficient
/// kernel) hits an invariant subspace early and yields only as many Ritz pairs
/// as its Krylov space admits. Callers must check the width rather than assume
/// `n_components`.
pub fn compute_smallest_eigenpairs_lanczos<T>(
    matrix: &CompressedSparseData<T>,
    n_components: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let ritz = sparse_lanczos(matrix, n_components, seed, true)?;
    Ok(ritz_to_rows(&ritz, matrix.shape.0, -1.0))
}

/// Compute largest eigenvalues and eigenvectors using Lanczos
///
/// Thick-restarted Lanczos, see [`lanczos_thick_restart`].
///
/// ### Params
///
/// * `matrix` - Symmetric sparse matrix
/// * `n_components` - Number of eigenpairs to compute
/// * `seed` - For reproducibility
///
/// ### Returns
///
/// (eigenvalues, eigenvectors) sorted descending, where `eigenvectors[i][j]` is
/// element i of eigenvector j.
///
/// Both are at most `n_components` wide, and can be SHORTER: an operator with
/// fewer distinct eigenvalues than requested (a clique, a rank-deficient
/// kernel) hits an invariant subspace early and yields only as many Ritz pairs
/// as its Krylov space admits. Callers must check the width rather than assume
/// `n_components`.
pub fn compute_largest_eigenpairs_lanczos<T>(
    matrix: &CompressedSparseData<T>,
    n_components: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let ritz = sparse_lanczos(matrix, n_components, seed, false)?;
    Ok(ritz_to_rows(&ritz, matrix.shape.0, 1.0))
}

/////////////////////////
// Von Neumann Entropy //
/////////////////////////

/// Calculate the von Neumann entropy of a matrix.
///
/// ### Params
///
/// * `mat`: The matrix for which to calculate the von Neumann entropy.
/// * `t_max`: The maximum number of powers to consider.
///
/// ### Returns
///
/// A vector of von Neumann entropies for each power.
pub fn landmark_von_neumann_entropy<T>(
    operator: &CompressedSparseData<T>,
    t_max: usize,
) -> Result<Vec<T>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let (n, _) = operator.shape;

    let eigenvalues: Vec<T> = if n < VNE_DENSE_THRESHOLD {
        let dense = operator.to_dense();
        let svd = dense.thin_svd().map_err(|_| ManifoldsError::FaerSvdError)?;
        svd.S().column_vector().iter().copied().collect()
    } else {
        let rank = VNE_SPARSE_RANK.min(n.saturating_sub(1));
        let svd = sparse_randomised_svd(operator, rank, 42, None, None)?;
        svd.s
    };

    let mut eigenvalues_t = eigenvalues.clone();
    let mut entropy = Vec::with_capacity(t_max);

    for _ in 0..t_max {
        let sum: T = eigenvalues_t.iter().copied().sum();

        if sum <= T::zero() {
            entropy.push(T::zero());
        } else {
            let h: T = eigenvalues_t
                .iter()
                .map(|&v| {
                    let prob = v / sum + T::epsilon();
                    -prob * prob.ln()
                })
                .sum();
            entropy.push(h);
        }

        for (vt, &v) in eigenvalues_t.iter_mut().zip(&eigenvalues) {
            *vt *= v;
        }
    }

    Ok(entropy)
}

///////////////
// FFT grids //
///////////////

/// Determine optimal grid size based on data spread.
///
/// Follows FIt-SNE's heuristic: number of boxes scales with spread,
/// then rounds up to FFT-friendly sizes. Allowed sizes match the C++
/// reference implementation.
///
/// ### Params
///
/// * `coord_min` - Minimum coordinate
/// * `coord_max` - Maximum coordinate
/// * `intervals_per_integer` - Target interval size (typically 1.0)
/// * `min_intervals` - Minimum number of intervals (typically 50)
///
/// ### Returns
///
/// Number of boxes per dimension
pub fn choose_grid_size(
    coord_min: f64,
    coord_max: f64,
    intervals_per_integer: f64,
    min_intervals: usize,
) -> usize {
    let spread = coord_max - coord_min;
    let n_boxes = ((spread / intervals_per_integer).max(min_intervals as f64)) as usize;

    const ALLOWED_SIZES: [usize; 20] = [
        25, 36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140, 150, 175, 200,
    ];

    if n_boxes < ALLOWED_SIZES[19] {
        for &size in &ALLOWED_SIZES {
            if size >= n_boxes {
                return size;
            }
        }
    }
    n_boxes
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_utils_math {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert_relative_eq!(result, 32.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dot_product_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];

        let result = dot(&a, &b);
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_norm() {
        let v = vec![3.0, 4.0];
        let result = norm(&v);
        assert_relative_eq!(result, 5.0, epsilon = 1e-6); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_normalise() {
        let mut v = vec![3.0, 4.0];
        normalise(&mut v);

        assert_relative_eq!(v[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(v[1], 0.8, epsilon = 1e-6);

        // Check that norm is now 1
        let new_norm = norm(&v);
        assert_relative_eq!(new_norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_symmetric_eigen_descending_simple() {
        // [2 1; 1 2] has eigenvalues 3 and 1
        let h = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });
        let (evals, _) = symmetric_eigen_descending(h.as_ref(), 2).unwrap();

        assert_eq!(evals.len(), 2);
        assert_relative_eq!(evals[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(evals[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_largest_eigenpairs_identity() {
        // Identity matrix has all eigenvalues = 1
        // But Lanczos is designed for graph Laplacians
        // For a proper test, use a small graph Laplacian instead

        // Simple path graph: 0-1-2
        // Adjacency matrix has 1s on off-diagonals
        let n = 3;
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];

        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));

        let (evals, evecs) = compute_smallest_eigenpairs_lanczos(&matrix, 2, 42).unwrap();

        assert_eq!(evals.len(), 2);
        assert_eq!(evecs.len(), n);
        assert_eq!(evecs[0].len(), 2);

        // For this Laplacian, largest eigenvalues should be positive
        for eval in evals {
            assert!(eval > 0.0);
        }
    }

    #[test]
    fn test_compute_smallest_eigenpairs_diagonal() {
        // Diagonal matrix with values [5, 4, 3, 2, 1]
        let n = 5;
        let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let indices = vec![0, 1, 2, 3, 4];
        let indptr = vec![0, 1, 2, 3, 4, 5];

        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));

        let (evals, evecs) = compute_smallest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();

        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.len(), n);

        // smallest eigenvalues should be approximately 1, 2, 3
        let mut sorted_evals = evals.clone();
        sorted_evals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(sorted_evals[0], 3.0, epsilon = 0.1);
        assert_relative_eq!(sorted_evals[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(sorted_evals[2], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_lanczos_reproducibility() {
        // n must stay well above SPMV_ROW_CHUNK: a smaller matrix fits in a
        // single parallel chunk and would leave every chunking bug undetected
        let n = 5000;

        // Create a tridiagonal matrix (more realistic for Lanczos)
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];

        for i in 0..n {
            if i > 0 {
                data.push(-1.0);
                indices.push(i - 1);
            }
            data.push(2.0);
            indices.push(i);
            if i < n - 1 {
                data.push(-1.0);
                indices.push(i + 1);
            }
            indptr.push(data.len());
        }

        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));

        let (evals1, evecs1) = compute_smallest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();
        let (evals2, evecs2) = compute_smallest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();

        assert_eq!(evals1, evals2);
        assert_eq!(evecs1, evecs2);
    }

    /// Build a symmetric tridiagonal (2 on the diagonal, -1 off) in CSR arrays.
    ///
    /// ### Params
    ///
    /// * `n` - Matrix dimension
    ///
    /// ### Returns
    ///
    /// `(data, indices, indptr)`
    fn tridiag_arrays(n: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        for i in 0..n {
            if i > 0 {
                data.push(-1.0);
                indices.push(i - 1);
            }
            data.push(2.0);
            indices.push(i);
            if i < n - 1 {
                data.push(-1.0);
                indices.push(i + 1);
            }
            indptr.push(data.len());
        }
        (data, indices, indptr)
    }

    #[test]
    fn test_lanczos_csc_input_matches_csr() {
        // The CSC arm of the borrow-or-own dispatch goes through `transform()`
        // and nothing else in the suite reaches it. The matrix is symmetric, so
        // both layouts hold identical content and must give identical spectra.
        let n = 200;
        let (data, indices, indptr) = tridiag_arrays(n);

        let csr = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));
        let csc = CompressedSparseData::new_csc(&data, &indices, &indptr, (n, n));

        let (csr_evals, csr_evecs) = compute_smallest_eigenpairs_lanczos(&csr, 3, 42).unwrap();
        let (csc_evals, csc_evecs) = compute_smallest_eigenpairs_lanczos(&csc, 3, 42).unwrap();

        assert_eq!(csr_evals.len(), csc_evals.len());
        assert_eq!(csr_evecs.len(), csc_evecs.len());
        for (a, b) in csr_evals.iter().zip(&csc_evals) {
            assert_relative_eq!(a, b, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_lanczos_eigenpairs_satisfy_residual() {
        // Checks ||Mv - lambda v||, which unlike a fixed bit pattern does not
        // depend on the SIMD level `dot_simd` dispatches to, so it holds across
        // machines. Catches a corrupted matrix or a botched back-transform,
        // neither of which a self-comparison test can see.
        //
        // The spectrum is well separated and the LARGEST pairs are requested so
        // that Lanczos converges. Asking for the smallest pairs of the 2/-1
        // tridiagonal would measure convergence instead: those eigenvalues are
        // too tightly clustered for the fixed iteration count to resolve.
        let n = 200;
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        for i in 0..n {
            if i > 0 {
                data.push(-1e-3);
                indices.push(i - 1);
            }
            data.push(1.0 / (i as f64 + 1.0));
            indices.push(i);
            if i < n - 1 {
                data.push(-1e-3);
                indices.push(i + 1);
            }
            indptr.push(data.len());
        }
        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));

        let (evals, evecs) = compute_largest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();
        assert_eq!(evals.len(), 3);

        for (k, &lambda) in evals.iter().enumerate() {
            let v: Vec<f64> = (0..n).map(|i| evecs[i][k] as f64).collect();
            let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                norm_v > 0.5,
                "eigenvector {k} is degenerate (norm {norm_v})"
            );

            let mut resid_sq = 0.0;
            for i in 0..n {
                let mut mv = 0.0;
                for idx in indptr[i]..indptr[i + 1] {
                    mv += data[idx] * v[indices[idx]];
                }
                let r = mv - lambda as f64 * v[i];
                resid_sq += r * r;
            }
            let rel = resid_sq.sqrt() / norm_v;
            assert!(rel < 1e-3, "residual {rel} too large for eigenpair {k}");
        }
    }

    #[test]
    #[should_panic]
    fn test_dot_product_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let _ = dot(&a, &b);
    }

    #[test]
    fn test_compute_largest_eigenpairs_diagonal() {
        let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let indices = vec![0, 1, 2, 3, 4];
        let indptr = vec![0, 1, 2, 3, 4, 5];
        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (5, 5));

        let (evals, evecs) = compute_largest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();

        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.len(), 5);
        assert_eq!(evecs[0].len(), 3);

        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(sorted[0], 5.0, epsilon = 0.1);
        assert_relative_eq!(sorted[1], 4.0, epsilon = 0.1);
        assert_relative_eq!(sorted[2], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_compute_largest_eigenpairs_path_laplacian() {
        // Path graph Laplacian, n=3: [[2,-1,0],[-1,2,-1],[0,-1,2]]
        // Eigenvalues: 2 - sqrt(2), 2, 2 + sqrt(2)
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];
        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let (evals, _) = compute_largest_eigenpairs_lanczos(&matrix, 2, 42).unwrap();

        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(sorted[0], 2.0 + 2.0_f32.sqrt(), epsilon = 0.05);
        assert_relative_eq!(sorted[1], 2.0, epsilon = 0.05);
    }

    #[test]
    fn test_largest_eigenpairs_reproducibility() {
        let n = 10;
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        for i in 0..n {
            if i > 0 {
                data.push(-1.0);
                indices.push(i - 1);
            }
            data.push(2.0);
            indices.push(i);
            if i < n - 1 {
                data.push(-1.0);
                indices.push(i + 1);
            }
            indptr.push(data.len());
        }
        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));

        let (e1, v1) = compute_largest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();
        let (e2, v2) = compute_largest_eigenpairs_lanczos(&matrix, 3, 42).unwrap();

        assert_eq!(e1, e2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_smallest_and_largest_agree_on_diagonal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0, 1, 2, 3, 4];
        let indptr = vec![0, 1, 2, 3, 4, 5];
        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (5, 5));

        let (small, _) = compute_smallest_eigenpairs_lanczos(&matrix, 2, 42).unwrap();
        let (large, _) = compute_largest_eigenpairs_lanczos(&matrix, 2, 42).unwrap();

        let mut s_sorted = small.clone();
        s_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut l_sorted = large.clone();
        l_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(s_sorted[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(s_sorted[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(l_sorted[0], 5.0, epsilon = 0.1);
        assert_relative_eq!(l_sorted[1], 4.0, epsilon = 0.1);
    }
}

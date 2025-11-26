use faer::traits::{ComplexField, RealField};
use faer::{Mat, MatRef};
use num_traits::{Float, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardNormal};
use rayon::iter::*;
use std::iter::Sum;
use std::ops::{Add, Mul};

use crate::assert_same_len;
use crate::data_struct::*;

////////////////////
// Randomised SVD //
////////////////////

/// Structure for random SVD results
///
/// ### Fields
///
/// * `u` - Matrix u of the SVD decomposition
/// * `v` - Matrix v of the SVD decomposition
/// * `s` - Eigen vectors of the SVD decomposition
#[derive(Clone, Debug)]
pub struct RandomSvdResults<T> {
    pub u: faer::Mat<T>,
    pub v: faer::Mat<T>,
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
) -> RandomSvdResults<T>
where
    T: Float + Send + Sync + Sum + ComplexField + RealField + ToPrimitive,
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
    let svd = b.thin_svd().unwrap();

    RandomSvdResults {
        u: q * svd.U(),
        v: svd.V().cloned(), // Use clone instead of manual copying
        s: svd.S().column_vector().iter().copied().collect(),
    }
}

/////////////////////////////////////
// Lanczos Eigenvalue calculations //
/////////////////////////////////////

/// Helper function for dot product of two vectors
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
    T: Float + Send + Sync + Sum,
{
    assert_same_len!(a, b);
    a.par_iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Helper function to normalise a vector
///
/// ### Params
///
/// * `v` - Initial vector
///
/// ### Returns
///
/// Normalised dot product of the vector `v`
fn norm<T>(v: &[T]) -> T
where
    T: Float + Send + Sync + Sum,
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
    T: Float + Send + Sync + Sum,
{
    let n = norm(v);
    v.par_iter_mut().for_each(|x| *x = *x / n);
}

/// Helper function to calculate eigenvalues
///
/// ### Params
///
/// * `alpha` - alpha vector
/// * `beta` - beta vector
///
/// ### Returns
///
/// Tuple of `(eigenvectors, eigenvalues)`
fn tridiag_eig<T>(alpha: &[T], beta: &[T]) -> (Vec<T::Real>, Mat<T>)
where
    T: ComplexField + Copy + RealField,
{
    let n = alpha.len();
    let mut t = Mat::<T>::zeros(n, n);

    for i in 0..n {
        t[(i, i)] = alpha[i];
        if i < n - 1 {
            t[(i, i + 1)] = beta[i];
            t[(i + 1, i)] = beta[i];
        }
    }

    let eig = t.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let evals = eig.S().column_vector().iter().copied().collect();
    let evecs = eig.U().to_owned();

    (evals, evecs)
}

/// Compute smallest eigenvalues and eigenvectors using Lanczos
///
/// This function returns the smallest eigenvalues, specifically designed
/// for spectral initialisations.
///
/// ### Params
///
/// * `matrix` - Sparse matrix in CSR format
/// * `n_components` - Number of eigenpairs to compute
/// * `seed` - For reproducibility
///
/// ### Returns
///
/// (eigenvalues, eigenvectors) where eigenvectors[i][j] is element j of
/// eigenvector i
pub fn compute_smallest_eigenpairs_lanczos<T>(
    matrix: &CompressedSparseData<T>,
    n_components: usize,
    seed: u64,
) -> (Vec<f32>, Vec<Vec<f32>>)
where
    T: Clone + Default + Into<f64> + Sync + Add + PartialEq + Mul,
{
    let n = matrix.shape.0;
    let n_iter = (n_components * 2 + 10).max(n_components).min(n);

    // Convert to CSR for efficient row access
    let csr = match matrix.cs_type {
        CompressedSparseFormat::Csr => matrix.clone(),
        CompressedSparseFormat::Csc => matrix.transform(),
    };

    let data_f64: Vec<f64> = csr.data.iter().map(|v| v.clone().into()).collect();

    let matvec = |x: &[f64], y: &mut [f64]| {
        y.fill(0.0);
        for i in 0..n {
            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let j = csr.indices[idx];
                y[i] += data_f64[idx] * x[j];
            }
        }
    };

    // Lanczos iteration
    let mut v = vec![0.0; n];
    let mut v_old = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut v_matrix = vec![vec![0.0; n]; n_iter];

    let mut rng = StdRng::seed_from_u64(seed);

    for i in 0..n {
        v[i] = rng.random::<f64>() - 0.5;
    }
    normalise(&mut v);

    let mut alpha = vec![0.0; n_iter];
    let mut beta = vec![0.0; n_iter];

    for j in 0..n_iter {
        v_matrix[j].copy_from_slice(&v);

        matvec(&v, &mut w);
        alpha[j] = dot(&w, &v);

        // w = w - alpha[j]*v - beta[j-1]*v_old
        for i in 0..n {
            w[i] -= alpha[j] * v[i];
            if j > 0 {
                w[i] -= beta[j - 1] * v_old[i];
            }
        }

        beta[j] = norm(&w);
        if beta[j] < 1e-12 {
            break;
        }

        v_old.copy_from_slice(&v);
        v.copy_from_slice(&w);
        normalise(&mut v);
    }

    let (evals, evecs) = tridiag_eig(&alpha[..n_iter], &beta[..n_iter - 1]);

    let mut indices: Vec<usize> = (0..evals.len()).collect();
    indices.sort_by(|&i, &j| evals[i].partial_cmp(&evals[j]).unwrap());

    let mut smallest_evals: Vec<f32> = Vec::with_capacity(n_components);
    let mut smallest_evecs: Vec<Vec<f32>> = Vec::with_capacity(n_components);

    for &idx in indices.iter().take(n_components) {
        // Transform eigenvector back to original space: v_original = V * v_tridiag
        let mut evec = vec![0.0; n];
        for i in 0..n {
            for j in 0..n_iter {
                evec[i] += v_matrix[j][i] * evecs[(j, idx)].to_f64().unwrap();
            }
        }

        // Normalise the transformed eigenvector... Really should do this...
        let norm: f64 = evec.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut evec {
            *x /= norm;
        }

        smallest_evals.push(evals[idx].to_f64().unwrap() as f32);
        smallest_evecs.push(evec.iter().map(|&x| x as f32).collect());
    }

    let mut transposed = vec![vec![0.0f32; n_components]; n];
    for comp_idx in 0..n_components {
        for point_idx in 0..n {
            transposed[point_idx][comp_idx] = smallest_evecs[comp_idx][point_idx];
        }
    }

    (smallest_evals, transposed)
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
    fn test_tridiag_eig_simple() {
        // Simple 2x2 tridiagonal matrix
        // [2  1]
        // [1  2]
        let alpha = vec![2.0, 2.0];
        let beta = vec![1.0];

        let (evals, _evecs) = tridiag_eig(&alpha, &beta);

        assert_eq!(evals.len(), 2);

        // Eigenvalues should be 1 and 3
        let mut sorted_evals = evals.clone();
        sorted_evals.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

        assert_relative_eq!(sorted_evals[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sorted_evals[1], 3.0, epsilon = 1e-5);
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

        let (evals, evecs) = compute_smallest_eigenpairs_lanczos(&matrix, 2, 42);

        assert_eq!(evals.len(), 2);
        assert_eq!(evecs.len(), n);
        assert_eq!(evecs[0].len(), 2);

        // For this Laplacian, largest eigenvalues should be positive
        for eval in evals {
            assert!(eval > 0.0);
        }
    }

    #[test]
    fn test_compute_largest_eigenpairs_diagonal() {
        // Diagonal matrix with values [5, 4, 3, 2, 1]
        let n = 5;
        let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let indices = vec![0, 1, 2, 3, 4];
        let indptr = vec![0, 1, 2, 3, 4, 5];

        let matrix = CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n));

        let (evals, evecs) = compute_smallest_eigenpairs_lanczos(&matrix, 3, 42);

        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.len(), n);

        // Largest eigenvalues should be approximately 5, 4, 3
        let mut sorted_evals = evals.clone();
        sorted_evals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert_relative_eq!(sorted_evals[0], 5.0, epsilon = 0.1);
        assert_relative_eq!(sorted_evals[1], 4.0, epsilon = 0.1);
        assert_relative_eq!(sorted_evals[2], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_lanczos_reproducibility() {
        // Use a proper sparse matrix, not identity
        let n = 10;

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

        let (evals1, evecs1) = compute_smallest_eigenpairs_lanczos(&matrix, 3, 42);
        let (evals2, evecs2) = compute_smallest_eigenpairs_lanczos(&matrix, 3, 42);

        assert_eq!(evals1, evals2);
        assert_eq!(evecs1, evecs2);
    }

    #[test]
    #[should_panic]
    fn test_dot_product_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let _ = dot(&a, &b);
    }
}

#![allow(clippy::needless_range_loop)]

use faer::Mat;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Create a synthetic dataset with well-separated clusters
///
/// ### Params
///
/// * `n_per_cluster`: Number of points per cluster
/// * `n_dim`: Number of dimensions
/// * `seed`: Seed for the random number generator
///
/// ### Returns
///
/// Tuple containing the data matrix and the labels vector
pub fn create_diagnostic_data(
    n_per_cluster: usize,
    n_dim: usize,
    seed: u64,
) -> (Mat<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_total = n_per_cluster * 5;

    let mut data_vec = Vec::with_capacity(n_total * n_dim);
    let mut labels = Vec::with_capacity(n_total);

    let centres = [
        vec![0.0; n_dim],
        (0..n_dim)
            .map(|i| if i == 0 { 5.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        (0..n_dim)
            .map(|i| if i == 1 { 5.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        (0..n_dim)
            .map(|i| if i == 2 { 5.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        vec![10.0; n_dim],
    ];

    for (cluster_id, centre) in centres.iter().enumerate() {
        for _ in 0..n_per_cluster {
            #[allow(clippy::needless_range_loop)]
            for dim in 0..n_dim {
                let noise: f64 = rng.random::<f64>() * 3.0 - 1.5;
                data_vec.push(centre[dim] + noise);
            }
            labels.push(cluster_id);
        }
    }

    let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
    (data, labels)
}

/// Transform the type to f32
///
/// ### Params
///
/// * `input` - The original matrix as `f64`
///
/// ### Returns
///
/// The matrix as `f32`
#[allow(unused)]
pub fn mat_to_f32(input: Mat<f64>) -> Mat<f32> {
    Mat::from_fn(input.nrows(), input.ncols(), |i, j| input[(i, j)] as f32)
}

/// Clusters with deliberately different spreads, as `f32` for the GPU path.
#[allow(unused)]
pub fn variable_density_clusters_f32(
    n_per_cluster: usize,
    n_dim: usize,
    stds: &[f64],
    seed: u64,
) -> (Mat<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = stds.len();
    let n_total = n_per_cluster * n_clusters;

    let mut data_vec = Vec::with_capacity(n_total * n_dim);
    let mut labels = Vec::with_capacity(n_total);

    for (cluster_id, &std) in stds.iter().enumerate() {
        let mut centre = vec![0.0; n_dim];
        centre[cluster_id % n_dim] = 40.0 * (cluster_id + 1) as f64;

        for _ in 0..n_per_cluster {
            for dim in 0..n_dim {
                let noise: f64 = rng.random::<f64>() * 2.0 - 1.0;
                data_vec.push((centre[dim] + noise * std) as f32);
            }
            labels.push(cluster_id);
        }
    }

    let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
    (data, labels)
}

/// Mean distance of each cluster's points from its own centroid in the
/// embedding.
#[allow(unused)]
pub fn cluster_radii(embd: &[Vec<f32>], labels: &[usize], n_clusters: usize) -> Vec<f64> {
    let n = labels.len();

    let mut cx = vec![0.0f64; n_clusters];
    let mut cy = vec![0.0f64; n_clusters];
    let mut counts = vec![0.0f64; n_clusters];

    for i in 0..n {
        cx[labels[i]] += embd[0][i] as f64;
        cy[labels[i]] += embd[1][i] as f64;
        counts[labels[i]] += 1.0;
    }
    for c in 0..n_clusters {
        cx[c] /= counts[c];
        cy[c] /= counts[c];
    }

    let mut radii = vec![0.0f64; n_clusters];
    for i in 0..n {
        let c = labels[i];
        let dx = embd[0][i] as f64 - cx[c];
        let dy = embd[1][i] as f64 - cy[c];
        radii[c] += (dx * dx + dy * dy).sqrt();
    }
    for c in 0..n_clusters {
        radii[c] /= counts[c];
    }

    radii
}

/// Mean distance of each cluster's points from its centroid in the input space.
#[allow(unused)]
pub fn original_cluster_radii(data: &Mat<f32>, labels: &[usize], n_clusters: usize) -> Vec<f64> {
    let n = data.nrows();
    let dim = data.ncols();

    let mut centroids = vec![vec![0.0f64; dim]; n_clusters];
    let mut counts = vec![0.0f64; n_clusters];

    for i in 0..n {
        for d in 0..dim {
            centroids[labels[i]][d] += data[(i, d)] as f64;
        }
        counts[labels[i]] += 1.0;
    }
    for c in 0..n_clusters {
        for d in 0..dim {
            centroids[c][d] /= counts[c];
        }
    }

    let mut radii = vec![0.0f64; n_clusters];
    for i in 0..n {
        let c = labels[i];
        let mut acc = 0.0;
        for d in 0..dim {
            let diff = data[(i, d)] as f64 - centroids[c][d];
            acc += diff * diff;
        }
        radii[c] += acc.sqrt();
    }
    for c in 0..n_clusters {
        radii[c] /= counts[c];
    }

    radii
}

/// Pearson correlation between two equal-length vectors.
#[allow(unused)]
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

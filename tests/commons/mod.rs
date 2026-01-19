use faer::Mat;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Create a synthetic dataset with well-separated clusters
/// (Same as UMAP tests - could do DRY)
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
            .map(|i| if i == 0 { 20.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        (0..n_dim)
            .map(|i| if i == 1 { 20.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        (0..n_dim)
            .map(|i| if i == 2 { 20.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        vec![10.0; n_dim],
    ];

    for (cluster_id, centre) in centres.iter().enumerate() {
        for _ in 0..n_per_cluster {
            for dim in 0..n_dim {
                let noise: f64 = rng.random::<f64>() * 0.5 - 0.25;
                data_vec.push(centre[dim] + noise);
            }
            labels.push(cluster_id);
        }
    }

    let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
    (data, labels)
}

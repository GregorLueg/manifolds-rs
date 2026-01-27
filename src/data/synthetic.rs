use faer::Mat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Generate Swiss Roll dataset
///
/// Creates a 2D manifold embedded in 3D space in the shape of a Swiss roll.
/// Standard benchmark for testing manifold unrolling capabilities.
///
/// ### Params
///
/// * `n_samples` - Number of points
/// * `noise` - Standard deviation of Gaussian noise added to the data
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, 3)
pub fn generate_swiss_roll(n_samples: usize, noise: f64, seed: u64) -> Mat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<f64>::zeros(n_samples, 3);

    for i in 0..n_samples {
        // Parameter t controls position along the roll
        let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * rng.random::<f64>());

        // Height along the roll
        let height = 21.0 * rng.random::<f64>();

        // Generate noise
        let noise_x = rng.random_range(-noise..noise);
        let noise_y = rng.random_range(-noise..noise);
        let noise_z = rng.random_range(-noise..noise);

        // Swiss roll coordinates
        data[(i, 0)] = t * t.cos() + noise_x;
        data[(i, 1)] = height + noise_y;
        data[(i, 2)] = t * t.sin() + noise_z;
    }

    data
}

/// Generate synthetic single-cell-like data with cluster structure
///
/// Creates data with multiple Gaussian clusters to simulate clusters (for
/// example cell types) in the data.
///
/// ### Params
///
/// * n_samples - Number of samples
/// * dim - Embedding dimensionality
/// * n_clusters - Number of distinct clusters
/// * cluster_std - Standard deviation within clusters
/// * seed - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim)
pub fn generate_clustered_data(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> (Mat<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<f64>::zeros(n_samples, dim);
    // variable cluster sizes and std deviations
    let mut centres = Vec::with_capacity(n_clusters);
    let mut cluster_stds = Vec::new();
    for _ in 0..n_clusters {
        let centre: Vec<f64> = (0..dim).map(|_| rng.random_range(-7.5..7.5)).collect();
        centres.push(centre);
        cluster_stds.push(rng.random_range(0.5..2.5));
    }
    // assign samples with variable cluster sizes
    // with some clusters bigger than others
    let mut cluster_assignments = Vec::new();
    for cluster_idx in 0..n_clusters {
        let weight = rng.random_range(0.5..2.5);
        let n_in_cluster = ((n_samples as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        cluster_assignments.extend(vec![cluster_idx; n_in_cluster]);
    }
    // fill remaining
    while cluster_assignments.len() < n_samples {
        cluster_assignments.push(rng.random_range(0..n_clusters));
    }
    cluster_assignments.shuffle(&mut rng);
    cluster_assignments.truncate(n_samples);
    // generate with variable noise
    for (i, &cluster_idx) in cluster_assignments.iter().enumerate() {
        let centre = &centres[cluster_idx];
        let std = cluster_stds[cluster_idx];
        for j in 0..dim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data[(i, j)] = centre[j] + noise * std;
        }
    }
    (data, cluster_assignments)
}

/// Generate tree-like branching structure dataset
///
/// Creates data with a trunk and multiple branches embedded in high-dimensional
/// space. Useful for testing algorithms like PHATE that preserve trajectory
/// structure.
///
/// ### Params
///
/// * `n_samples` - Total number of points
/// * `n_branches` - Number of branches (excluding trunk)
/// * `dim` - Dimensionality of ambient space
/// * `noise` - Standard deviation of Gaussian noise perpendicular to branches
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim)
pub fn generate_tree_structure(
    n_samples: usize,
    n_branches: usize,
    dim: usize,
    noise: f64,
    seed: u64,
) -> Mat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<f64>::zeros(n_samples, dim);

    // distribute samples: trunk gets 30%, rest split among branches
    let n_trunk = (n_samples as f64 * 0.3) as usize;
    let n_per_branch = (n_samples - n_trunk) / n_branches;

    // generate trunk direction (random unit vector)
    let trunk_dir: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let trunk_norm = trunk_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    let trunk_dir: Vec<f64> = trunk_dir.iter().map(|x| x / trunk_norm).collect();

    let mut idx = 0;

    // generate trunk points
    for i in 0..n_trunk {
        let t = (i as f64) / (n_trunk as f64) * 5.0;

        for j in 0..dim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let gauss_noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            data[(idx, j)] = t * trunk_dir[j] + noise * gauss_noise;
        }
        idx += 1;
    }

    // generate branch points
    let branch_start = 2.5; // Branches start halfway along trunk

    for _ in 0..n_branches {
        // random branch direction (orthogonalised to trunk)
        let mut branch_dir: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

        // orthogonalise to trunk using Gram-Schmidt
        let dot: f64 = branch_dir.iter().zip(&trunk_dir).map(|(a, b)| a * b).sum();
        for j in 0..dim {
            branch_dir[j] -= dot * trunk_dir[j];
        }

        let branch_norm = branch_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        let branch_dir: Vec<f64> = branch_dir.iter().map(|x| x / branch_norm).collect();

        // starting point on trunk
        let branch_start_point: Vec<f64> = trunk_dir.iter().map(|x| x * branch_start).collect();

        for i in 0..n_per_branch {
            if idx >= n_samples {
                break;
            }

            let t = (i as f64) / (n_per_branch as f64) * 5.0;

            for j in 0..dim {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let gauss_noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

                data[(idx, j)] = branch_start_point[j] + t * branch_dir[j] + noise * gauss_noise;
            }
            idx += 1;
        }
    }

    data
}

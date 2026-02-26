use faer::Mat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

///////////////
// SwissRole //
///////////////

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
        // parameter t controls position along the roll
        let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * rng.random::<f64>());

        // height along the roll
        let height = 21.0 * rng.random::<f64>();

        // generate noise
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

/////////////
// Cluster //
/////////////

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

////////////////
// Trajectory //
////////////////

/// Structure for branch specification
///
/// ### Fields
///
/// * `parent` - The optional parent of this cell type
/// * `split_at` - Fraction along the parent where this branch starts (0.0 or
///   1.0)
/// * `length` - The length of this branch
pub struct BranchSpec {
    pub parent: Option<usize>,
    pub split_at: f64, // fraction along parent where this branch starts (0.0–1.0)
    pub length: f64,
}

/// Generates an example haematopoiesis trajectory with 9 branches,
/// mimicking the differentiation hierarchy from HSC through myeloid
/// (CMP → Monocyte, Granulocyte, Erythroid) and lymphoid
/// (CLP → B cell, T cell, NK cell) lineages.
///
/// All branches split at the tip of their parent (`split_at: 1.0`).
/// Branch lengths are scaled loosely to reflect biological distance
/// from the progenitor state.
///
/// ### Returns
///
/// A vector of 9 [`BranchSpec`], ordered so that each parent index
/// refers to an earlier entry in the vector, as required by
/// [`generate_trajectory`].
pub fn generate_example_branches() -> Vec<BranchSpec> {
    let branches = vec![
        BranchSpec {
            parent: None,
            split_at: 0.0,
            length: 2.0,
        }, // 0: HSC
        BranchSpec {
            parent: Some(0),
            split_at: 1.0,
            length: 3.0,
        }, // 1: CMP
        BranchSpec {
            parent: Some(0),
            split_at: 1.0,
            length: 2.0,
        }, // 2: CLP
        BranchSpec {
            parent: Some(1),
            split_at: 1.0,
            length: 3.0,
        }, // 3: Monocyte
        BranchSpec {
            parent: Some(1),
            split_at: 1.0,
            length: 4.0,
        }, // 4: Granulocyte
        BranchSpec {
            parent: Some(1),
            split_at: 1.0,
            length: 4.0,
        }, // 5: Erythroid
        BranchSpec {
            parent: Some(2),
            split_at: 1.0,
            length: 5.0,
        }, // 6: B cell
        BranchSpec {
            parent: Some(2),
            split_at: 1.0,
            length: 4.0,
        }, // 7: T cell
        BranchSpec {
            parent: Some(2),
            split_at: 1.0,
            length: 3.0,
        }, // 8: NK cell
    ];

    branches
}

/// Generate Gaussian independent variables
///
/// ### Params
///
/// * `rng` - The Rng
///
/// ### Returns
///
/// The random value
fn box_muller(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate a synthetic single-cell differentiation trajectory in
/// high-dimensional space.
///
/// Points are sampled along branches with pseudotime biased toward the branch
/// origin, mimicking progenitor accumulation. Branch directions are partially
/// rotated toward their parent direction so that related lineages share
/// variance rather than being fully orthogonal. Cells near a bifurcation are
/// blended back toward the parent trajectory over a short transition window.
/// Noise amplitude grows with pseudotime, reflecting increased transcriptional
/// heterogeneity in mature cell types. A small shared low-rank background
/// (3 components) is added to every cell to simulate global variation such as
/// cell cycle or stress response.
///
/// Branch directions are sequentially orthogonalised against all previously
/// generated directions before the parent-mixing step, so the base geometry
/// remains well-conditioned.
///
/// ### Params
///
/// * `n_samples` - Total number of points, distributed evenly across branches
/// * `branches` - Slice of [`BranchSpec`] defining the tree topology. Branch
///   `i` may reference any `j < i` as its parent; forward references are not
///   allowed. The first entry must have `parent: None`.
/// * `dim` - Dimensionality of the ambient space. Must be >= `branches.len()`
/// * `noise` - Base standard deviation of Gaussian noise; scales up along
///   pseudotime as `noise * (1.0 + t / branch_length)`
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Tuple of (data matrix of shape `(n_samples, dim)`, branch assignments as
/// indices into `branches`)
pub fn generate_trajectory(
    n_samples: usize,
    branches: &[BranchSpec],
    dim: usize,
    noise: f64,
    seed: u64,
) -> (Mat<f64>, Vec<usize>) {
    assert!(dim >= branches.len(), "dim must be >= number of branches");

    let mut rng = StdRng::seed_from_u64(seed);
    let n_branches = branches.len();
    let n_per_branch = n_samples / n_branches;

    let mut dirs: Vec<Vec<f64>> = Vec::with_capacity(n_branches);
    let mut starts: Vec<Vec<f64>> = Vec::with_capacity(n_branches);

    for spec in branches.iter() {
        // random direction, orthogonalised against ALL previous directions
        let mut dir: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        for prev in &dirs {
            let dot: f64 = dir.iter().zip(prev).map(|(a, b)| a * b).sum();
            for j in 0..dim {
                dir[j] -= dot * prev[j];
            }
        }
        let norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dir: Vec<f64> = dir.iter().map(|x| x / norm).collect();

        let start = match spec.parent {
            None => vec![0.0; dim],
            Some(p) => starts[p]
                .iter()
                .zip(&dirs[p])
                .map(|(s, d)| s + spec.split_at * branches[p].length * d)
                .collect(),
        };

        dirs.push(dir);
        starts.push(start);
    }

    let mut data = Mat::<f64>::zeros(n_samples, dim);
    let mut assignments = Vec::with_capacity(n_samples);
    let mut idx = 0;

    for (b, spec) in branches.iter().enumerate() {
        let count = if b == n_branches - 1 {
            n_samples - idx
        } else {
            n_per_branch
        };
        for i in 0..count {
            let t = (i as f64) / (count as f64) * spec.length;
            for j in 0..dim {
                let noise_val = noise * box_muller(&mut rng);
                // project noise onto the plane perpendicular to dirs[b]
                data[(idx, j)] = starts[b][j] + t * dirs[b][j] + noise_val;
            }
            // subtract the noise component along the branch direction
            let dot: f64 = (0..dim)
                .map(|j| data[(idx, j)] - starts[b][j] - t * dirs[b][j])
                .zip(&dirs[b])
                .map(|(n, d)| n * d)
                .sum();
            for j in 0..dim {
                data[(idx, j)] -= dot * dirs[b][j];
            }
            assignments.push(b);
            idx += 1;
        }
    }

    (data, assignments)
}

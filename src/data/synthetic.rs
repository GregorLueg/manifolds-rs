//! Synthetic data generation to understand caveats of different embedding
//! methods and do (assumption) testing

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

/// Defines the biological topology for the synthetic trajectory.
#[derive(Default, Clone, Debug)]
pub enum TrajectoryTopology {
    #[default]
    /// A hierarchical tree with cascading lineage commitments.
    DeepBifurcation,
    /// A single continuous lineage without bifurcations.
    Linear,
    /// A main progenitor backbone with mature cell types splitting off mid-way.
    Comb,
}

/// Parse the toplogy
///
/// ### Params
///
/// * `s` - Topology to create
///
/// ### Returns
///
/// The option of the `TrajectoryTopology`
pub fn parse_topology(s: &str) -> Option<TrajectoryTopology> {
    match s.to_lowercase().as_str() {
        "bifurcation" => Some(TrajectoryTopology::DeepBifurcation),
        "linear" => Some(TrajectoryTopology::Linear),
        "combination" => Some(TrajectoryTopology::Comb),
        _ => None,
    }
}

/// Structure for branch specification
#[derive(Clone, Debug)]
pub struct BranchSpec {
    /// The optional parent of this cell type
    pub parent: Option<usize>,
    /// Fraction along the parent where this branch starts (0.0 or 1.0)
    pub split_at: f64,
    /// The length of this branch
    pub length: f64,
}

/// Generates a trajectory of differentiation example
///
/// ### Params
///
/// * `topology` - The desired Topology
///
/// ### Returns
///
/// A vector of `BranchSpec`s for the desired topology
pub fn generate_example_branches(topology: &TrajectoryTopology) -> Vec<BranchSpec> {
    match topology {
        TrajectoryTopology::DeepBifurcation => vec![
            BranchSpec {
                parent: None,
                split_at: 0.0,
                length: 0.75,
            }, // 0: Stem
            BranchSpec {
                parent: Some(0),
                split_at: 1.0,
                length: 0.5,
            }, // 1: Progenitor
            BranchSpec {
                parent: Some(1),
                split_at: 1.0,
                length: 3.0,
            }, // 2: Type A
            BranchSpec {
                parent: Some(1),
                split_at: 1.0,
                length: 1.0,
            }, // 3: Type B
            BranchSpec {
                parent: Some(3),
                split_at: 1.0,
                length: 1.5,
            }, // 4: Type C
            BranchSpec {
                parent: Some(3),
                split_at: 1.0,
                length: 2.5,
            }, // 5: Type D
        ],
        TrajectoryTopology::Linear => vec![
            BranchSpec {
                parent: None,
                split_at: 0.0,
                length: 1.0,
            }, // 0: Early
            BranchSpec {
                parent: Some(0),
                split_at: 1.0,
                length: 2.0,
            }, // 1: Intermediate
            BranchSpec {
                parent: Some(1),
                split_at: 1.0,
                length: 3.0,
            }, // 2: Late
        ],
        TrajectoryTopology::Comb => vec![
            BranchSpec {
                parent: None,
                split_at: 0.0,
                length: 5.0,
            }, // 0: Main Stem Backbone
            // Notice split_at < 1.0: peeling off along the backbone!
            BranchSpec {
                parent: Some(0),
                split_at: 0.2,
                length: 2.0,
            }, // 1: Early exit fate
            BranchSpec {
                parent: Some(0),
                split_at: 0.5,
                length: 2.5,
            }, // 2: Mid exit fate
            BranchSpec {
                parent: Some(0),
                split_at: 0.8,
                length: 3.0,
            }, // 3: Late exit fate
        ],
    }
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
/// high-dimensional space with non-linear manifolds.
///
/// Points are sampled along branches with pseudotime heavily biased toward the
/// branch origin, mimicking progenitor accumulation (critical slowing down).
///
/// **Geometry & Curvature:**
///
/// Branch directions are partially rotated toward  their parent direction so
/// related lineages share variance. Furthermore, each branch exhibits
/// non-linear curvature via a sinusoidal drift along a secondary, orthogonal
/// axis. This creates complex, non-linear manifolds to test embedding
/// algorithms.
///
/// **Bifurcations & Noise:**
///
/// Cells near a bifurcation are smoothly blended back toward the parent's curved
/// trajectory over a transition window. Noise amplitude grows heteroskedastically
/// with pseudotime, reflecting increased transcriptional heterogeneity in mature
/// cell types. A shared low-rank background (3 components) is added to every
/// cell to simulate global variation such as cell cycle or stress response.
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

    // Shared low-rank background (cell cycle, stress response, etc.)
    let n_bg = 3usize;
    let bg_weight = 0.15;
    let bg_dirs: Vec<Vec<f64>> = (0..n_bg)
        .map(|_| {
            let mut v: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect();

    let mut dirs: Vec<Vec<f64>> = Vec::with_capacity(n_branches);
    let mut curve_dirs: Vec<Vec<f64>> = Vec::with_capacity(n_branches);
    let mut starts: Vec<Vec<f64>> = Vec::with_capacity(n_branches);

    for spec in branches.iter() {
        // 1. Generate primary branch direction
        let mut dir: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        for prev in &dirs {
            let dot: f64 = dir.iter().zip(prev).map(|(a, b)| a * b).sum();
            for j in 0..dim {
                dir[j] -= dot * prev[j];
            }
        }
        let norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut dir: Vec<f64> = dir.iter().map(|x| x / norm).collect();

        // rotate child direction partly toward parent to share biological variance
        if let Some(p) = spec.parent {
            let alpha = 0.4f64;
            for j in 0..dim {
                dir[j] = alpha * dirs[p][j] + (1.0 - alpha) * dir[j];
            }
            let norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
            dir = dir.iter().map(|x| x / norm).collect();
        }

        // generate a curvature direction strictly orthogonal to the primary direction
        let mut c_dir: Vec<f64> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let dot_c: f64 = c_dir.iter().zip(&dir).map(|(c, d)| c * d).sum();
        for j in 0..dim {
            c_dir[j] -= dot_c * dir[j];
        }
        let norm_c = c_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        let c_dir: Vec<f64> = c_dir.iter().map(|x| x / norm_c).collect();

        // calculate starting position (accounting for parent's curvature)
        let start = match spec.parent {
            None => vec![0.0; dim],
            Some(p) => {
                let t_parent = spec.split_at * branches[p].length;
                let parent_curve_amt = (t_parent / branches[p].length * std::f64::consts::PI).sin()
                    * (branches[p].length * 0.3);

                (0..dim)
                    .map(|j| {
                        starts[p][j] + t_parent * dirs[p][j] + parent_curve_amt * curve_dirs[p][j]
                    })
                    .collect()
            }
        };

        dirs.push(dir);
        curve_dirs.push(c_dir);
        starts.push(start);
    }

    let mut data = Mat::<f64>::zeros(n_samples, dim);
    let mut assignments = Vec::with_capacity(n_samples);
    let mut idx = 0;
    let transition_window = 0.3f64;

    for (b, spec) in branches.iter().enumerate() {
        let count = if b == n_branches - 1 {
            n_samples - idx
        } else {
            n_per_branch
        };

        for _ in 0..count {
            // power > 1.0 creates heavy-tailed accumulation at the root (progenitors)
            let u: f64 = rng.random();
            let t = spec.length * u.powf(2.5);

            // calculate smooth blending weight near the bifurcation
            let blend = if spec.parent.is_some() {
                let frac = t / spec.length;
                if frac < transition_window {
                    (1.0 - frac / transition_window).powi(2)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // calculate clean, noiseless manifold position with non-linear
            // curvature
            let mut clean_pos = vec![0.0; dim];
            let curve_amt = (t / spec.length * std::f64::consts::PI).sin() * (spec.length * 0.3);

            for j in 0..dim {
                clean_pos[j] = starts[b][j] + t * dirs[b][j] + curve_amt * curve_dirs[b][j];
            }

            // blend back toward the parent's true curved position
            if blend > 0.0 {
                if let Some(p) = spec.parent {
                    let t_parent = spec.split_at * branches[p].length;
                    let parent_curve_amt = (t_parent / branches[p].length * std::f64::consts::PI)
                        .sin()
                        * (branches[p].length * 0.3);

                    for j in 0..dim {
                        let parent_pos = starts[p][j]
                            + t_parent * dirs[p][j]
                            + parent_curve_amt * curve_dirs[p][j];
                        clean_pos[j] = blend * parent_pos + (1.0 - blend) * clean_pos[j];
                    }
                }
            }

            // add heteroskedastic noise
            let local_noise = noise * (1.0 + t / spec.length);
            for j in 0..dim {
                let noise_val = local_noise * box_muller(&mut rng);
                data[(idx, j)] = clean_pos[j] + noise_val;
            }

            // flatten noise along the base directional axis to maintain
            // trajectory shape
            let dot: f64 = (0..dim)
                .map(|j| (data[(idx, j)] - clean_pos[j]) * dirs[b][j])
                .sum();
            for j in 0..dim {
                data[(idx, j)] -= dot * dirs[b][j];
            }

            // 5. Add shared low-rank background variation (e.g., cell cycle)
            for bg in &bg_dirs {
                let coeff = bg_weight * box_muller(&mut rng);
                for j in 0..dim {
                    data[(idx, j)] += coeff * bg[j];
                }
            }

            assignments.push(b);
            idx += 1;
        }
    }

    (data, assignments)
}

///////////////////////////
// Hierarchical clusters //
///////////////////////////

/// Generate hierarchical cluster data with supergroups and subclusts.
///
/// Creates data with a two-level cluster hierarchy: `n_supergroups` groups
/// each containing `n_subclusts` tight subclusters. Supergroup centres are
/// spread far apart; subcluster centres are spread tightly around their
/// supergroup centre. This structure tests whether an embedding preserves
/// global relational structure (supergroup distances) while also resolving
/// local separation (subclusters within a supergroup).
///
/// UMAP/tSNE tends to collapse supergroup distances arbitrarily and
/// over-separate subclusters. PaCMAP should keep supergroups at meaningful
/// relative distances while still resolving the subcluster structure.
///
/// ### Params
///
/// * `n_samples` - Total number of points, distributed evenly across all
///   subclusters.
/// * `dim` - Dimensionality of the ambient space.
/// * `n_supergroups` - Number of top-level groups. Default `3`.
/// * `n_subclusts` - Number of subclusters per supergroup. Default `3`.
/// * `supergroup_spread` - Spread of supergroup centres. Default `15.0`.
/// * `subcluster_spread` - Spread of subcluster centres around their
///   supergroup centre. Default 2.0.
/// * `point_std` - Within-subcluster Gaussian noise. Default `0.4`.
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// Tuple of:
/// - `Mat<f64>` of shape `(n_samples, dim)`
/// - `Vec<usize>` supergroup label per sample (0..n_supergroups)
/// - `Vec<usize>` subcluster label per sample (0..n_supergroups * n_subclusts)
#[allow(clippy::too_many_arguments)]
pub fn generate_hierarchical_clusters(
    n_samples: usize,
    dim: usize,
    n_supergroups: usize,
    n_subclusts: usize,
    supergroup_spread: f64,
    subcluster_spread: f64,
    point_std: f64,
    seed: u64,
) -> (Mat<f64>, Vec<usize>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let total_subclusters = n_supergroups * n_subclusts;
    let n_per_subcluster = n_samples / total_subclusters;

    // supergroup centres — spread far apart
    let supergroup_centres: Vec<Vec<f64>> = (0..n_supergroups)
        .map(|_| {
            (0..dim)
                .map(|_| rng.random_range(-supergroup_spread..supergroup_spread))
                .collect()
        })
        .collect();

    // subcluster centres — tight around their supergroup centre
    let mut subcluster_centres: Vec<(usize, Vec<f64>)> = Vec::with_capacity(total_subclusters);
    for sg in 0..n_supergroups {
        let sc = &supergroup_centres[sg];
        for _ in 0..n_subclusts {
            let centre = sc
                .iter()
                .map(|&c| c + rng.random_range(-subcluster_spread..subcluster_spread))
                .collect();
            subcluster_centres.push((sg, centre));
        }
    }

    let actual_n = n_per_subcluster * total_subclusters;
    let mut data = Mat::<f64>::zeros(actual_n, dim);
    let mut supergroup_labels = Vec::with_capacity(actual_n);
    let mut subcluster_labels = Vec::with_capacity(actual_n);

    let mut idx = 0;
    for (sc_idx, (sg_idx, centre)) in subcluster_centres.iter().enumerate() {
        for _ in 0..n_per_subcluster {
            for j in 0..dim {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                data[(idx, j)] = centre[j] + noise * point_std;
            }
            supergroup_labels.push(*sg_idx);
            subcluster_labels.push(sc_idx);
            idx += 1;
        }
    }

    (data, supergroup_labels, subcluster_labels)
}

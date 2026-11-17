//! Timing and recall harness for the CPU NN-Descent and IVF kNN graphs.
//!
//! Sweeps dataset size, `k` and input shape. IVF runs with both k-means
//! algorithms (e-means and Lloyd's) at the default `nprobe` and at twice that,
//! since e-means' more even partitions may want more probes. Recall is against
//! an exact f64 scan of a fixed row sample, so the ground truth stays cheap at
//! any `n`. Gated behind `large_scale_diagnostics`.
//!
//! `KNN_BENCH_RAW` swaps the synthetic shapes for a real matrix: raw
//! little-endian `f64`, row-major, `KNN_BENCH_DIM` columns. Each `n` takes the
//! first `n` rows, so shuffle the rows when writing the file.
//!
//! ```sh
//! KNN_BENCH_N=20000,100000 KNN_BENCH_REPS=3 \
//!     cargo bench --features large_scale_diagnostics --bench knn
//! ```

use faer::Mat;
use manifolds_rs::data::synthetic::generate_clustered_data;
use manifolds_rs::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

/// Dataset sizes when `KNN_BENCH_N` is unset. Small on purpose: scale up
/// explicitly once the small case looks sane.
const DEFAULT_SAMPLE_SIZES: [usize; 1] = [20_000];

/// Timed repeats per configuration when `KNN_BENCH_REPS` is unset; the median
/// is reported.
const DEFAULT_REPS: usize = 3;

/// Neighbour counts: UMAP default, a common larger choice, tSNE at perplexity
/// 30, and a wide graph.
const K_VALUES: [usize; 4] = [15, 30, 90, 150];

/// Rows whose exact neighbours define recall.
const N_RECALL_ROWS: usize = 2_000;

/// Number of clusters in the synthetic data.
const N_CLUSTERS: usize = 20;

/// Seed for the synthetic data, the recall rows and the ANN search.
const SEED: usize = 42;

/////////////
// Helpers //
/////////////

/// One input shape in the sweep.
struct Dataset {
    /// Label printed in the table
    name: &'static str,
    /// Ambient dimensionality
    dim: usize,
    /// Scale column `j` by `1 / sqrt(j + 1)`, a decaying spectrum like a PCA
    /// input rather than isotropic noise
    decay: bool,
}

/// The input shapes to sweep.
const DATASETS: [Dataset; 3] = [
    Dataset {
        name: "iso_50d",
        dim: 50,
        decay: false,
    },
    Dataset {
        name: "decay_50d",
        dim: 50,
        decay: true,
    },
    Dataset {
        name: "iso_15d",
        dim: 15,
        decay: false,
    },
];

/// Read a `usize` list from a comma-separated environment variable.
///
/// ### Params
///
/// * `key` - Variable name
///
/// ### Returns
///
/// The parsed list, or `None` if unset.
fn env_list(key: &str) -> Option<Vec<usize>> {
    std::env::var(key).ok().map(|v| {
        v.split(',')
            .map(|s| s.trim().parse().expect("comma-separated integers"))
            .collect()
    })
}

/// Generate one dataset in `f32`, plus the `f64` original for the ground truth.
///
/// ### Params
///
/// * `ds` - Input shape
/// * `n` - Number of samples
///
/// ### Returns
///
/// `(f64 matrix, f32 matrix)`, samples by features.
fn make_data(ds: &Dataset, n: usize) -> (Mat<f64>, Mat<f32>) {
    let (mut data, _) = generate_clustered_data(n, ds.dim, N_CLUSTERS, SEED as u64);
    if ds.decay {
        for j in 0..ds.dim {
            let s = 1.0 / ((j + 1) as f64).sqrt();
            for i in 0..n {
                data[(i, j)] *= s;
            }
        }
    }
    let data_f32 = Mat::from_fn(n, ds.dim, |i, j| data[(i, j)] as f32);
    (data, data_f32)
}

/// Read the first `n` rows of a raw row-major `f64` matrix.
///
/// ### Params
///
/// * `path` - File of little-endian `f64`
/// * `dim` - Columns
/// * `n` - Rows to keep
///
/// ### Returns
///
/// `(f64 matrix, f32 matrix)`, samples by features.
fn read_raw(path: &str, dim: usize, n: usize) -> (Mat<f64>, Mat<f32>) {
    let bytes = std::fs::read(path).expect("readable KNN_BENCH_RAW file");
    let vals: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .take(n * dim)
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    assert_eq!(vals.len(), n * dim, "file holds fewer than {n} rows");
    let data = Mat::from_fn(n, dim, |i, j| vals[i * dim + j]);
    let data_f32 = Mat::from_fn(n, dim, |i, j| vals[i * dim + j] as f32);
    (data, data_f32)
}

/// Exact `k_max` nearest neighbours of a set of rows, self excluded, in `f64`.
///
/// ### Params
///
/// * `data` - Samples by features
/// * `rows` - Rows to query
/// * `k_max` - Neighbours per row
///
/// ### Returns
///
/// One row of neighbour indices per query, nearest first.
fn exact_knn(data: &Mat<f64>, rows: &[usize], k_max: usize) -> Vec<Vec<usize>> {
    let (n, dim) = (data.nrows(), data.ncols());
    rows.par_iter()
        .map(|&i| {
            let mut d: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let s = (0..dim)
                        .map(|c| (data[(i, c)] - data[(j, c)]).powi(2))
                        .sum::<f64>();
                    (s, j)
                })
                .collect();
            d.select_nth_unstable_by(k_max - 1, |a, b| a.0.total_cmp(&b.0));
            d.truncate(k_max);
            d.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            d.into_iter().map(|(_, j)| j).collect()
        })
        .collect()
}

/// Mean recall@k over the sampled rows.
///
/// ### Params
///
/// * `found` - Full graph from the search, one row per sample
/// * `rows` - Sampled rows
/// * `truth` - Exact neighbours of the sampled rows, at least `k` wide
/// * `k` - Neighbours per row
///
/// ### Returns
///
/// Fraction of exact neighbours recovered.
fn recall(found: &[Vec<usize>], rows: &[usize], truth: &[Vec<usize>], k: usize) -> f64 {
    let hits: usize = rows
        .iter()
        .zip(truth)
        .map(|(&i, t)| {
            let t: HashSet<_> = t[..k].iter().collect();
            found[i].iter().take(k).filter(|j| t.contains(j)).count()
        })
        .sum();
    hits as f64 / (rows.len() * k) as f64
}

/// Time one search configuration and score its recall.
///
/// ### Params
///
/// * `data` - Input in `f32`
/// * `k` - Neighbours per point
/// * `ann_type` - Backend key
/// * `params` - Backend parameters
/// * `reps` - Timed repeats
/// * `rows` - Sampled rows for recall
/// * `truth` - Exact neighbours of the sampled rows
///
/// ### Returns
///
/// `(median seconds, recall)`.
fn run(
    data: &Mat<f32>,
    k: usize,
    ann_type: &str,
    params: &NearestNeighbourParams<f32>,
    reps: usize,
    rows: &[usize],
    truth: &[Vec<usize>],
) -> (f64, f64) {
    let mut times = Vec::with_capacity(reps);
    let mut graph = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let (idx, _) =
            run_ann_search(data.as_ref(), k, ann_type.to_string(), params, SEED, 0).unwrap();
        times.push(t0.elapsed().as_secs_f64());
        graph = idx;
    }
    times.sort_by(f64::total_cmp);
    (times[reps / 2], recall(&graph, rows, truth, k))
}

//////////
// Main //
//////////

fn main() {
    let sizes = env_list("KNN_BENCH_N").unwrap_or(DEFAULT_SAMPLE_SIZES.to_vec());
    let reps = env_list("KNN_BENCH_REPS")
        .and_then(|v| v.first().copied())
        .unwrap_or(DEFAULT_REPS)
        .max(1);
    let k_max = *K_VALUES.iter().max().unwrap();
    let raw = std::env::var("KNN_BENCH_RAW").ok();

    println!("dataset,n,k,method,nprobe,seconds,recall");
    for &n in &sizes {
        let inputs: Vec<(&str, Mat<f64>, Mat<f32>)> = match &raw {
            Some(path) => {
                let dim = env_list("KNN_BENCH_DIM")
                    .and_then(|v| v.first().copied())
                    .expect("KNN_BENCH_DIM with KNN_BENCH_RAW");
                let (d, f) = read_raw(path, dim, n);
                vec![("raw", d, f)]
            }
            None => DATASETS
                .iter()
                .map(|ds| {
                    let (d, f) = make_data(ds, n);
                    (ds.name, d, f)
                })
                .collect(),
        };
        for (name, data, data_f32) in inputs {
            let step = (n / N_RECALL_ROWS).max(1);
            let rows: Vec<usize> = (0..n).step_by(step).take(N_RECALL_ROWS).collect();
            let truth = exact_knn(&data, &rows, k_max);

            // ann-search-rs defaults: nlist = sqrt(n), nprobe = sqrt(nlist)
            let nlist = ((n as f32).sqrt() as usize).max(1);
            let nprobe = ((nlist as f64).sqrt() as usize).max(1);

            for &k in &K_VALUES {
                let base = NearestNeighbourParams::<f32>::default();
                let (s, r) = run(&data_f32, k, "nndescent", &base, reps, &rows, &truth);
                println!("{name},{n},{k},nndescent,-,{s:.3},{r:.4}");

                for k_means in ["emeans", "lloyd"] {
                    for probes in [nprobe, 2 * nprobe] {
                        let params = NearestNeighbourParams {
                            k_means: k_means.to_string(),
                            n_probes: Some(probes),
                            ..base.clone()
                        };
                        let (s, r) = run(&data_f32, k, "ivf", &params, reps, &rows, &truth);
                        println!("{name},{n},{k},ivf_{k_means},{probes},{s:.3},{r:.4}");
                    }
                }
            }
        }
    }
}

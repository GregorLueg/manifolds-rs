//! Correctness and determinism tests for spectral initialisation, plus a timing
//! harness behind `large_scale_diagnostics`.
//!
//! These are the only coverage of the parallel code paths in `spectral_layout`,
//! so the graphs are sized deliberately rather than generously. Every graph
//! clears both random-init fallback guards (`n <= n_comp + 1` in
//! `single_component_spectral_raw`, `component.len() < 2 * n_comp` in
//! `multi_component_init`), and every *component* clears `PARALLEL_VEC_MIN` so
//! the parallel matvec actually runs. A component below that threshold takes the
//! serial path and tests nothing. Keep them cheap; CI runners are slow.

use manifolds_rs::data::init::spectral_layout;
use manifolds_rs::prelude::CoordinateList;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(feature = "large_scale_diagnostics")]
use std::time::Instant;

/// Neighbours per point in the synthetic graphs, matching the UMAP default.
const K: usize = 15;

/// Smallest component size that still exercises the parallel Lanczos paths.
///
/// Must stay above `PARALLEL_VEC_MIN` (4096) in `src/utils/math.rs`, and high
/// enough above `SPMV_ROW_CHUNK` (512) that the row partition spans many chunks.
/// Raising those constants means raising this one.
const PARALLEL_COMPONENT_SIZE: usize = 6_000;

/////////////
// Helpers //
/////////////

/// Build a synthetic kNN-shaped graph with cluster block structure.
///
/// Mirrors the shape a real UMAP graph has after `symmetrise_graph`: symmetric,
/// sorted by row and by column within each row, with neighbours drawn from
/// within a contiguous cluster block so the column-gather pattern in the sparse
/// matvec is realistic rather than artificially local.
///
/// ### Params
///
/// * `n` - Number of vertices
/// * `n_clusters` - Number of contiguous index blocks to draw neighbours from
/// * `connect` - Whether to chain the clusters into one connected component
/// * `seed` - Random seed
///
/// ### Returns
///
/// Symmetric weighted graph in COO format.
fn synthetic_knn_graph(
    n: usize,
    n_clusters: usize,
    connect: bool,
    seed: u64,
) -> CoordinateList<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let block = n.div_ceil(n_clusters);

    // undirected edge set, keyed (min, max) so both directions get one weight
    let mut pairs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    let push = |pairs: &mut Vec<Vec<(usize, f64)>>, a: usize, b: usize, w: f64| {
        pairs[a].push((b, w));
        pairs[b].push((a, w));
    };

    for i in 0..n {
        let c = i / block;
        let lo = c * block;
        let hi = ((c + 1) * block).min(n);
        if hi - lo < 2 {
            continue;
        }
        for _ in 0..K {
            let j = rng.random_range(lo..hi);
            if j != i {
                push(&mut pairs, i, j, rng.random::<f64>() * 0.9 + 0.1);
            }
        }
    }

    // chain cluster c to cluster c-1 so the whole graph is one component
    if connect {
        for c in 1..n_clusters {
            let a = (c - 1) * block;
            let b = (c * block).min(n - 1);
            if a != b {
                push(&mut pairs, a, b, 0.5);
            }
        }
    }

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for (i, neighbours) in pairs.iter_mut().enumerate() {
        neighbours.sort_unstable_by_key(|&(j, _)| j);
        neighbours.dedup_by_key(|&mut (j, _)| j);
        for &(j, w) in neighbours.iter() {
            row_indices.push(i);
            col_indices.push(j);
            values.push(w);
        }
    }

    CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    }
}

/// FNV-1a over the raw bits of an embedding.
///
/// Used as a compact stand-in for full output comparison so a run's exact
/// numerical result can be recorded and diffed across refactors.
///
/// ### Params
///
/// * `embd` - Embedding coordinates
///
/// ### Returns
///
/// 64-bit digest of the embedding's bit pattern.
#[cfg(feature = "large_scale_diagnostics")]
fn digest(embd: &[Vec<f64>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for row in embd {
        for &v in row {
            h ^= v.to_bits();
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Time `spectral_layout` over five runs and print the minimum.
///
/// ### Params
///
/// * `label` - Description printed alongside the timing
/// * `graph` - Graph to embed
#[cfg(feature = "large_scale_diagnostics")]
fn time_layout(label: &str, graph: &CoordinateList<f64>) {
    /// Repetitions per measurement; the minimum is reported.
    const N_REPS: usize = 5;

    let mut best = f64::INFINITY;
    let mut d = 0u64;
    for _ in 0..N_REPS {
        let start = Instant::now();
        let embd = spectral_layout(graph, 2, 42, None, None).unwrap();
        let secs = start.elapsed().as_secs_f64();
        best = best.min(secs);
        d = digest(&embd);
    }
    println!(
        "{label:<34} nnz={:>10}  min={:>8.1} ms  digest={d:#018x}",
        graph.row_indices.len(),
        best * 1e3
    );
}

//////////////////
// Determinism  //
//////////////////

#[test]
fn spectral_determinism_single_component() {
    // one connected component of PARALLEL_COMPONENT_SIZE vertices, so the
    // parallel matvec runs and spans roughly a dozen row chunks
    let graph = synthetic_knn_graph(PARALLEL_COMPONENT_SIZE, 4, true, 7);
    let a = spectral_layout(&graph, 2, 42, None, None).unwrap();
    let b = spectral_layout(&graph, 2, 42, None, None).unwrap();
    assert_eq!(a, b, "spectral layout is not run-to-run deterministic");
}

#[test]
fn spectral_determinism_multi_component() {
    // Two disconnected components of PARALLEL_COMPONENT_SIZE each. Both the
    // count and the size matter: more, smaller components would drop every
    // per-component solve onto the serial path and cover nothing.
    let graph = synthetic_knn_graph(2 * PARALLEL_COMPONENT_SIZE, 2, false, 11);
    let a = spectral_layout(&graph, 2, 42, None, None).unwrap();
    let b = spectral_layout(&graph, 2, 42, None, None).unwrap();
    assert_eq!(a, b, "multi-component spectral layout is not deterministic");
}

#[test]
fn spectral_is_independent_of_thread_count() {
    // Catches a float reduction whose order follows the thread count, such as
    // swapping a serial SIMD `dot` for `par_iter().sum()`. That stays
    // reproducible on one machine, so an ordinary determinism test misses it,
    // and then hands a user different embeddings on different hardware. Three
    // threads is deliberately an awkward divisor.
    //
    // It does not catch a change of chunk size in the row-parallel matvec, and
    // cannot: each output element is an independent fixed-order sum, so the
    // result is chunk-agnostic by construction.
    let graph = synthetic_knn_graph(PARALLEL_COMPONENT_SIZE, 4, true, 7);

    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| spectral_layout(&graph, 2, 42, None, None).unwrap())
    };

    let (one, three, eight) = (run(1), run(3), run(8));
    assert_eq!(one, three, "spectral layout varies with the thread count");
    assert_eq!(three, eight, "spectral layout varies with the thread count");
}

/// Build a uniform-weight clique on `n` vertices.
///
/// The normalised adjacency of a clique has exactly two distinct eigenvalues,
/// so Lanczos exhausts its Krylov space after two iterations. That makes it the
/// minimal reproduction of an early breakdown, which is otherwise hard to
/// trigger and easy to leave untested.
///
/// ### Params
///
/// * `n` - Number of vertices
///
/// ### Returns
///
/// Complete graph in COO format, all weights 1.
fn clique(n: usize) -> CoordinateList<f64> {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                row_indices.push(i);
                col_indices.push(j);
                values.push(1.0);
            }
        }
    }
    CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    }
}

#[test]
fn spectral_survives_early_lanczos_breakdown() {
    // A cluster of identical points is routine in real data and forms exactly
    // this graph. Lanczos breaks down at j = 1, yielding fewer Ritz values than
    // the n_comp + 1 eigenpairs requested, and the solver must degrade rather
    // than index off the end of the selected set.
    for n in [5usize, 50, 500] {
        let embd = spectral_layout(&clique(n), 2, 42, None, None)
            .unwrap_or_else(|e| panic!("clique({n}) failed: {e}"));
        assert_eq!(embd.len(), n);
        assert_eq!(embd[0].len(), 2);
        for row in &embd {
            for &v in row {
                assert!(v.is_finite(), "clique({n}) produced a non-finite value");
            }
        }
        // Neither dimension may be dead. The threshold has to clear the 1e-4
        // Gaussian noise `finalise_spectral_embedding` sprinkles on at the end,
        // otherwise an all-zero column reads as alive on the noise alone; a
        // genuinely populated dimension spans a good fraction of SPECTRAL_RANGE.
        for d in 0..2 {
            let lo = embd.iter().map(|r| r[d]).fold(f64::INFINITY, f64::min);
            let hi = embd.iter().map(|r| r[d]).fold(f64::NEG_INFINITY, f64::max);
            assert!(
                hi - lo > 1.0,
                "clique({n}) dimension {d} collapsed (span {})",
                hi - lo
            );
        }
    }
}

#[test]
fn spectral_reaches_lanczos_and_is_finite() {
    // guards against the whole suite silently testing only the random fallback
    let graph = synthetic_knn_graph(PARALLEL_COMPONENT_SIZE, 4, true, 3);
    let embd = spectral_layout(&graph, 2, 42, None, None).unwrap();

    assert_eq!(embd.len(), PARALLEL_COMPONENT_SIZE);
    for row in &embd {
        for &v in row {
            assert!(v.is_finite(), "spectral layout produced a non-finite value");
        }
    }

    // No dimension may collapse. As above, the threshold must clear the 1e-4
    // noise floor rather than merely be non-zero.
    for d in 0..2 {
        let lo = embd.iter().map(|r| r[d]).fold(f64::INFINITY, f64::min);
        let hi = embd.iter().map(|r| r[d]).fold(f64::NEG_INFINITY, f64::max);
        assert!(
            hi - lo > 1.0,
            "embedding dimension {d} collapsed (span {})",
            hi - lo
        );
    }
}

#[test]
fn spectral_preserves_structure_within_each_component() {
    // Determinism and finiteness are not enough: corrupting the local indices
    // in `extract_all_subgraphs` leaves both intact while destroying the
    // subgraph, because a wrecked Laplacian just falls back to a random fill
    // that is still finite, still reproducible, and still spans the range.
    //
    // So assert the thing that actually depends on the subgraph being right.
    // Each component is two dense blobs joined by a single weak edge; the
    // Fiedler vector of that component must separate them. It cannot, if the
    // edges were bucketed under the wrong endpoints.
    let (blob, n_components) = (100usize, 2usize);
    let per_component = 2 * blob;
    let n = n_components * per_component;

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut rng = StdRng::seed_from_u64(31);

    let edge = |r: &mut Vec<usize>, c: &mut Vec<usize>, v: &mut Vec<f64>, a, b, w| {
        r.push(a);
        c.push(b);
        v.push(w);
        r.push(b);
        c.push(a);
        v.push(w);
    };

    for comp in 0..n_components {
        let base = comp * per_component;
        // dense within each blob
        for half in 0..2 {
            let lo = base + half * blob;
            for i in lo..(lo + blob) {
                for _ in 0..K {
                    let j = rng.random_range(lo..(lo + blob));
                    if i != j {
                        edge(&mut row_indices, &mut col_indices, &mut values, i, j, 1.0);
                    }
                }
            }
        }
        // one weak bridge between the two blobs of this component
        edge(
            &mut row_indices,
            &mut col_indices,
            &mut values,
            base,
            base + blob,
            0.01,
        );
    }

    let graph = CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    };
    let embd = spectral_layout(&graph, 2, 42, None, None).unwrap();
    assert_eq!(embd.len(), n);

    for comp in 0..n_components {
        let base = comp * per_component;
        let mean = |lo: usize, hi: usize| {
            let s: f64 = embd[lo..hi].iter().map(|r| r[0]).sum();
            s / (hi - lo) as f64
        };
        let spread = |lo: usize, hi: usize| {
            let m = mean(lo, hi);
            (embd[lo..hi].iter().map(|r| (r[0] - m).powi(2)).sum::<f64>() / (hi - lo) as f64).sqrt()
        };

        let (m1, m2) = (mean(base, base + blob), mean(base + blob, base + 2 * blob));
        let within = spread(base, base + blob).max(spread(base + blob, base + 2 * blob));

        // the bridge is 100x weaker than the intra-blob edges, so the split
        // must dominate the scatter inside either blob
        assert!(
            (m1 - m2).abs() > 2.0 * within,
            "component {comp}: blobs not separated (gap {}, within-blob spread {within})",
            (m1 - m2).abs()
        );
    }
}

#[test]
fn spectral_handles_unsorted_and_duplicate_edges() {
    // Every other test graph is emitted row-major with unique ascending
    // columns, which is exactly the input for which the column-sorting radix
    // pass is a no-op. This one draws neighbours across the whole index range
    // (so columns arrive unsorted) and repeats an edge (so `(i, j)` duplicates
    // exist), which is what a kNN backend returning a repeated neighbour looks
    // like. See `dummy_knn` in src/data/pacmap_pairs.rs.
    let n = 600;
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut rng = StdRng::seed_from_u64(19);

    for i in 0..n {
        for _ in 0..K {
            let j = rng.random_range(0..n);
            if j != i {
                row_indices.push(i);
                col_indices.push(j);
                values.push(rng.random::<f64>() * 0.9 + 0.1);
            }
        }
        // a deliberately duplicated (i, j) pair with differing weights
        for w in [0.25, 0.75] {
            row_indices.push(i);
            col_indices.push((i + 1) % n);
            values.push(w);
        }
    }

    let graph = CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    };

    let a = spectral_layout(&graph, 2, 42, None, None).unwrap();
    let b = spectral_layout(&graph, 2, 42, None, None).unwrap();
    assert_eq!(a, b, "unsorted/duplicate graph is not deterministic");
    assert_eq!(a.len(), n);
    for row in &a {
        for &v in row {
            assert!(
                v.is_finite(),
                "unsorted/duplicate graph gave a non-finite value"
            );
        }
    }
}

#[test]
fn spectral_handles_directed_cross_component_edges() {
    // A purely directed chain `i -> i-1`. BFS follows row-to-column only, so
    // every vertex becomes its own component and every edge straddles two
    // labels. That is the branch in `extract_all_subgraphs` that drops
    // cross-label edges, which no symmetric graph can reach.
    let n = 20;
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for i in 1..n {
        row_indices.push(i);
        col_indices.push(i - 1);
        values.push(1.0);
    }

    let graph = CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    };

    let a = spectral_layout(&graph, 2, 42, None, None).unwrap();
    let b = spectral_layout(&graph, 2, 42, None, None).unwrap();
    assert_eq!(a, b, "directed graph is not deterministic");
    assert_eq!(a.len(), n);
    for row in &a {
        for &v in row {
            assert!(v.is_finite(), "directed graph gave a non-finite value");
        }
    }
}

#[test]
fn spectral_meta_layout_uses_data_when_many_components() {
    // The only test that reaches `component_spectral_meta`: it needs both
    // `n_components > 2 * n_comp` AND a data matrix. Every other test passes
    // `None`, so it takes the simplex placement and this path stayed dark,
    // even though production always arrives here with `Some(data)`.
    let (n_per, n_clusters) = (60usize, 6usize);
    let n = n_per * n_clusters;
    let graph = synthetic_knn_graph(n, n_clusters, false, 23);

    // clusters far apart in feature space, so the meta-layout has real signal
    let data = faer::Mat::from_fn(n, 3, |i, j| {
        let c = (i / n_per) as f64;
        match j {
            0 => c * 20.0,
            1 => (c * 2.0).sin() * 20.0,
            _ => (i % n_per) as f64 * 0.01,
        }
    });

    let a = spectral_layout(&graph, 2, 42, None, Some(data.as_ref())).unwrap();
    let b = spectral_layout(&graph, 2, 42, None, Some(data.as_ref())).unwrap();
    assert_eq!(a, b, "spectral meta-layout is not deterministic");

    // the components must not be piled on top of each other
    let centroid = |c: usize| {
        let (lo, hi) = (c * n_per, (c + 1) * n_per);
        let mut acc = [0.0f64; 2];
        for row in a.iter().take(hi).skip(lo) {
            acc[0] += row[0];
            acc[1] += row[1];
        }
        [acc[0] / n_per as f64, acc[1] / n_per as f64]
    };
    let mut min_sep = f64::INFINITY;
    for x in 0..n_clusters {
        for y in (x + 1)..n_clusters {
            let (cx, cy) = (centroid(x), centroid(y));
            let d = ((cx[0] - cy[0]).powi(2) + (cx[1] - cy[1]).powi(2)).sqrt();
            min_sep = min_sep.min(d);
        }
    }
    assert!(
        min_sep > 1e-3,
        "component centroids collapsed (min separation {min_sep})"
    );
}

/////////////
// Timing  //
/////////////

#[test]
#[cfg(feature = "large_scale_diagnostics")]
fn bench_spectral_single_component() {
    for &n in &[20_000usize, 100_000, 250_000] {
        let graph = synthetic_knn_graph(n, 8, true, 7);
        time_layout(&format!("single-component n={n}"), &graph);
    }
}

#[test]
#[cfg(feature = "large_scale_diagnostics")]
fn bench_spectral_multi_component() {
    for &n_clusters in &[8usize, 64, 256] {
        let graph = synthetic_knn_graph(100_000, n_clusters, false, 11);
        time_layout(&format!("multi-component c={n_clusters}"), &graph);
    }
}

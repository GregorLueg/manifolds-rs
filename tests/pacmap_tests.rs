#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::data::pacmap_pairs::construct_pacmap_pairs;
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rustc_hash::FxHashMap;

/////////////
// Helpers //
/////////////

fn cluster_separation(embd: &[Vec<f64>], labels: &[usize]) -> f64 {
    let mut centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();
    for (i, &label) in labels.iter().enumerate() {
        let e = centres.entry(label).or_insert((0.0, 0.0, 0));
        e.0 += embd[0][i];
        e.1 += embd[1][i];
        e.2 += 1;
    }
    let centroids: Vec<(usize, f64, f64)> = centres
        .iter()
        .map(|(&l, &(sx, sy, c))| (l, sx / c as f64, sy / c as f64))
        .collect();

    let min_inter = centroids
        .iter()
        .enumerate()
        .flat_map(|(i, ci)| {
            centroids[i + 1..]
                .iter()
                .map(move |cj| ((ci.1 - cj.1).powi(2) + (ci.2 - cj.2).powi(2)).sqrt())
        })
        .fold(f64::INFINITY, f64::min);

    let avg_intra = centroids
        .iter()
        .map(|(label, cx, cy)| {
            let pts: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == *label)
                .map(|(i, _)| i)
                .collect();
            pts.iter()
                .map(|&i| ((embd[0][i] - cx).powi(2) + (embd[1][i] - cy).powi(2)).sqrt())
                .sum::<f64>()
                / pts.len() as f64
        })
        .sum::<f64>()
        / centroids.len() as f64;

    min_inter / avg_intra
}

///////////
// Tests //
///////////

#[test]
fn pacmap_integration_01_knn_correctness() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let k = 50;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PaCMAP DIAGNOSTIC 1: kNN Search ===");
    println!("Points per cluster: 100, k = {} neighbours", k);

    // Self must not appear in neighbours
    let self_count = knn_indices
        .iter()
        .enumerate()
        .filter(|(i, nbrs)| nbrs.contains(i))
        .count();
    assert_eq!(self_count, 0, "self-loops found in kNN");
    println!("No self-loops in kNN");

    // With tightly clustered data and k=50, almost all neighbours should be
    // intra-cluster (99 same-cluster points available)
    let intra_ratio: f64 = knn_indices
        .iter()
        .enumerate()
        .map(|(i, nbrs)| {
            nbrs.iter().filter(|&&j| labels[j] == labels[i]).count() as f64 / nbrs.len() as f64
        })
        .sum::<f64>()
        / knn_indices.len() as f64;

    let max_possible = (99_usize).min(k) as f64 / k as f64;
    println!(
        "Intra-cluster ratio: {:.1}% (max possible {:.1}%)",
        intra_ratio * 100.0,
        max_possible * 100.0
    );
    assert!(
        intra_ratio > max_possible * 0.9,
        "expected intra-cluster ratio > {:.1}%, got {:.1}%",
        max_possible * 90.0,
        intra_ratio * 100.0
    );

    let all_dists: Vec<f64> = knn_dist.iter().flatten().copied().collect();
    let min_d = all_dists.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(min_d > 0.0, "min distance must be > 0 (no self)");
}

#[test]
fn pacmap_integration_02_pair_counts() {
    let (data, _) = create_diagnostic_data(100, 10, 42);
    let n = data.nrows();
    let k = 50;
    let n_mid_near = 2;
    let n_further = 2;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, _) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let pairs = construct_pacmap_pairs(&knn_indices, n_mid_near, n_further, 4, 50, 42);

    println!("\n=== PaCMAP DIAGNOSTIC 2: Pair Counts ===");
    println!(
        "near={}, mid_near={}, further={}",
        pairs.near.len(),
        pairs.mid_near.len(),
        pairs.further.len()
    );

    assert_eq!(pairs.near.len(), n * k, "near pair count");
    assert_eq!(pairs.mid_near.len(), n * n_mid_near, "mid-near pair count");
    assert_eq!(pairs.further.len(), n * n_further, "further pair count");
}

#[test]
fn pacmap_integration_03_no_self_pairs() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let k = 50;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, _) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let pairs = construct_pacmap_pairs(&knn_indices, 2, 2, 4, 50, 42);

    println!("\n=== PaCMAP DIAGNOSTIC 3: No Self-Pairs ===");

    for &(i, j) in &pairs.near {
        assert_ne!(i, j, "self-pair in near at {}", i);
    }
    for &(i, j) in &pairs.mid_near {
        assert_ne!(i, j, "self-pair in mid-near at {}", i);
    }
    for &(i, j) in &pairs.further {
        assert_ne!(i, j, "self-pair in further at {}", i);
    }
    println!("No self-pairs in any pair type");
}

#[test]
fn pacmap_integration_04_mid_near_candidate_window() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let k = 50;
    let candidate_start = 4;
    let candidate_end = 50;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, _) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let pairs = construct_pacmap_pairs(&knn_indices, 2, 2, candidate_start, candidate_end, 42);

    println!("\n=== PaCMAP DIAGNOSTIC 4: Mid-Near Candidate Window ===");

    for &(i, j) in &pairs.mid_near {
        let window_end = candidate_end.min(knn_indices[i].len());
        let window = &knn_indices[i][candidate_start..window_end];
        assert!(
            window.contains(&j),
            "mid-near pair ({},{}) not from candidate window [{},{})",
            i,
            j,
            candidate_start,
            window_end
        );
    }
    println!("All mid-near pairs drawn from candidate window");
}

#[test]
fn pacmap_integration_05_output_shape() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let n = data.nrows();
    let n_dim = 2;

    let params = PacmapParams::new(
        Some(n_dim),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let embd = pacmap(data.as_ref(), None, &params, 42, false).unwrap();

    println!("\n=== PaCMAP DIAGNOSTIC 5: Output Shape ===");
    println!("Output shape: [{}][{}]", embd.len(), embd[0].len());

    assert_eq!(embd.len(), n_dim, "wrong number of output dimensions");
    assert_eq!(embd[0].len(), n, "wrong number of output samples");
}

#[test]
fn pacmap_integration_06_all_finite() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    let params = PacmapParams::<f64>::default();
    let embd = pacmap(data.as_ref(), None, &params, 42, false).unwrap();

    println!("\n=== PaCMAP DIAGNOSTIC 6: All Finite ===");

    let nan_count = embd.iter().flatten().filter(|x| x.is_nan()).count();
    let inf_count = embd.iter().flatten().filter(|x| x.is_infinite()).count();

    println!("NaN: {}, Inf: {}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "NaN values in embedding");
    assert_eq!(inf_count, 0, "Inf values in embedding");
}

#[test]
fn pacmap_integration_07_reproducibility() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    let params = PacmapParams::new(
        Some(2),
        None,
        Some("balltree".to_string()),
        Some("adam".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let embd1 = pacmap(data.as_ref(), None, &params, 42, false).unwrap();
    let embd2 = pacmap(data.as_ref(), None, &params, 42, false).unwrap();

    println!("\n=== PaCMAP DIAGNOSTIC 7: Reproducibility ===");

    let max_diff = embd1
        .iter()
        .zip(&embd2)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| (x - y).abs()))
        .fold(0.0_f64, f64::max);

    println!("Max coordinate difference: {:.10}", max_diff);
    assert!(
        max_diff < 1e-10,
        "PaCMAP not reproducible: diff = {}",
        max_diff
    );
}

#[test]
fn pacmap_integration_08_different_seeds_differ() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    let params = PacmapParams::<f64>::default();
    let embd1 = pacmap(data.as_ref(), None, &params, 42, false).unwrap();
    let embd2 = pacmap(data.as_ref(), None, &params, 123, false).unwrap();

    println!("\n=== PaCMAP DIAGNOSTIC 8: Different Seeds ===");

    let max_diff = embd1
        .iter()
        .zip(&embd2)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| (x - y).abs()))
        .fold(0.0_f64, f64::max);

    println!("Max coordinate difference: {:.4}", max_diff);
    assert!(
        max_diff > 0.01,
        "different seeds produced identical results"
    );
}

#[test]
fn pacmap_integration_09_cluster_separation() {
    // 50 per cluster = 250 total. Using 100+ points with k=50 means near pairs
    // dominate heavily (10% of dataset as neighbours), which makes it hard for
    // further pairs to push 5 clusters apart in 2D within 450 epochs.
    let (data, labels) = create_diagnostic_data(50, 10, 42);

    println!("\n=== PaCMAP DIAGNOSTIC 9: Cluster Separation ===");

    let params = PacmapParams::new(
        Some(2),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some("pca".to_string()),
        None,
        Some(PacmapOptimParams::new(
            Some(450),
            None,
            None,
            None,
            None,
            None,
            None,
        )),
    );

    let embd = pacmap(data.as_ref(), None, &params, 42, true).unwrap();

    let all_coords: Vec<f64> = embd.iter().flatten().copied().collect();
    let range = all_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - all_coords.iter().copied().fold(f64::INFINITY, f64::min);

    println!("Coordinate range: {:.2}", range);
    assert!(range > 1.0, "embedding collapsed");
    assert!(range < 5000.0, "embedding exploded");

    let sep = cluster_separation(&embd, &labels);
    println!("Separation ratio: {:.2}", sep);
    assert!(sep > 1.0, "clusters not separated: ratio = {:.2}", sep);
    println!("Clusters are well separated");
}

#[test]
fn pacmap_integration_10_precomputed_knn() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let k = 50;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) = run_ann_search(
        data.as_ref(),
        k,
        "nndescent".to_string(),
        &nn_params,
        42,
        false,
    );

    // Use sequential Adam — parallel has non-deterministic FP accumulation order
    // so precomputed vs internal may diverge even with identical kNN graphs.
    let params = PacmapParams::new(
        Some(2),
        None,
        Some("nndescent".to_string()),
        Some("adam".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let embd_pre = pacmap(
        data.as_ref(),
        Some((knn_indices.clone(), knn_dist.clone())),
        &params,
        42,
        false,
    )
    .unwrap();
    let embd_int = pacmap(data.as_ref(), None, &params, 42, false).unwrap();

    println!("\n=== PaCMAP DIAGNOSTIC 10: Precomputed kNN ===");

    let max_diff = embd_pre
        .iter()
        .zip(&embd_int)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| (x - y).abs()))
        .fold(0.0_f64, f64::max);

    println!("Max diff (precomputed vs internal): {:.10}", max_diff);
    assert!(
        max_diff < 1e-10,
        "precomputed kNN produced different results: diff = {}",
        max_diff
    );
}

#[test]
fn pacmap_integration_11_sequential_parallel_identical() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    let params_seq = PacmapParams::new(
        Some(2),
        None,
        None,
        Some("adam".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let params_par = PacmapParams::new(
        Some(2),
        None,
        None,
        Some("adam_parallel".to_string()),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let embd_seq = pacmap(data.as_ref(), None, &params_seq, 42, false).unwrap();
    let embd_par = pacmap(data.as_ref(), None, &params_par, 42, false).unwrap();

    println!("\n=== PaCMAP DIAGNOSTIC 11: Sequential vs Parallel ===");

    let max_diff = embd_seq
        .iter()
        .zip(&embd_par)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| (x - y).abs()))
        .fold(0.0_f64, f64::max);

    println!("Max diff (sequential vs parallel): {:.10}", max_diff);
    assert!(
        max_diff < 1e-2,
        "sequential and parallel diverged: diff = {}",
        max_diff
    );
}

#[test]
fn pacmap_integration_12_further_pairs_influence() {
    use manifolds_rs::data::pacmap_pairs::PacmapPairs;
    use manifolds_rs::training::pacmap_optimiser::*;

    // Build a small embedding and pairs; compare run with/without further pairs
    let n = 20;
    let knn: Vec<Vec<usize>> = (0..n)
        .map(|i| (1..=10).map(|o| (i + o) % n).collect())
        .collect();

    let pairs_full = construct_pacmap_pairs(&knn, 2, 2, 4, 10, 42);
    let pairs_no_fp = PacmapPairs {
        near: pairs_full.near.clone(),
        mid_near: pairs_full.mid_near.clone(),
        further: vec![],
    };

    let make_embd = || -> Vec<Vec<f64>> { (0..n).map(|i| vec![i as f64 * 0.1, 0.0]).collect() };

    let optim_params = PacmapOptimParams::new(Some(50), None, None, None, None, None, None);

    let mut embd_full = make_embd();
    let mut embd_no_fp = make_embd();
    optimise_pacmap(&mut embd_full, &pairs_full, &optim_params, false);
    optimise_pacmap(&mut embd_no_fp, &pairs_no_fp, &optim_params, false);

    println!("\n=== PaCMAP DIAGNOSTIC 12: Further Pairs Influence ===");

    let diff: f64 = embd_full
        .iter()
        .zip(&embd_no_fp)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| (x - y).abs()))
        .sum();

    println!(
        "Total coord diff (with vs without further pairs): {:.4}",
        diff
    );
    assert!(diff > 1e-6, "further pairs had no effect on the embedding");
}

#[test]
fn pacmap_integration_13_pca_init_range() {
    let (data, _) = create_diagnostic_data(100, 10, 42);

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, _) =
        run_ann_search(data.as_ref(), 50, "hnsw".to_string(), &nn_params, 42, false);

    let dummy_graph = manifolds_rs::data::pacmap_pairs::knn_to_coo_unweighted::<f64>(&knn_indices);

    println!("\n=== PaCMAP DIAGNOSTIC 13: PCA vs Random Init ===");

    for (name, init_type) in &[
        ("pca", parse_initilisation("pca", true, None).unwrap()),
        (
            "random",
            parse_initilisation("random", true, Some(1e-4)).unwrap(),
        ),
    ] {
        let embd = initialise_embedding(init_type, 2, 42, &dummy_graph, data.as_ref()).unwrap();
        let coords: Vec<f64> = embd.iter().flatten().copied().collect();
        let mean = coords.iter().sum::<f64>() / coords.len() as f64;
        let std =
            (coords.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / coords.len() as f64).sqrt();
        println!("{} init std: {:.6}", name, std);
        assert!(
            coords.iter().all(|x| x.is_finite()),
            "{} init has non-finite values",
            name
        );
    }
}

#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use approx::assert_relative_eq;
use manifolds_rs::prelude::*;
use manifolds_rs::training::fa2_optimiser::optimise_fa2;
use manifolds_rs::*;

/////////////////////
// Python parity //
/////////////////////

// Positions after 5 iterations of the reference Python ForceAtlas2
// (github.com/bhargavchippada/forceatlas2), run through
// `uv run --with ~/repos/others/forceatlas2 --with numpy --with scipy --with tqdm python`:
//
//   fa = ForceAtlas2(barnesHutOptimize=False, backend="loop", verbose=False, **kw)
//   out = fa.forceatlas2(G, pos=pos, iterations=5)
//
// with `G` the dense symmetric adjacency of `parity_graph()`, `pos` the
// (30, 2) array of `parity_pos()`, and `kw` per mode:
//
//   REF_LINEAR          {}
//   REF_LIN_LOG         {"linLogMode": True}
//   REF_STRONG_GRAVITY  {"strongGravityMode": True, "gravity": 0.5, "scalingRatio": 3.0}
//   REF_EDGE_INFLUENCE  {"edgeWeightInfluence": 0.5, "jitterTolerance": 0.8}
//
// Values printed with `f"{v:.12e}"`.

#[rustfmt::skip]
const REF_LINEAR: [f64; 60] = [
    1.486358206724, 6.553654866032, 11.63925809572, 2.205563336935,
    2.883819925678, -3.700229600461, -2.672174934493, -10.90292515384,
    -3.844214534625, -17.69618206166, 2.751319297946, -12.15193694091,
    10.09697760856, -6.316290049193, 3.338008201526, 13.815831184,
    -6.898755134602, 17.28832607505, -7.945125472783, 12.18013999924,
    8.263355148073, 3.995635106262, 10.74316417734, -13.98559429103,
    3.729468396936, -18.4080399568, -9.443771936777, -16.54698252187,
    -3.873934087742, -8.9627050701, 1.18593071037, 1.909458464412,
    9.176437516384, 12.68861626944, -1.474116960416, 20.96224023763,
    -10.74163409589, 17.6562146633, -8.241688081904, -0.08468819208019,
    6.715703852196, -8.954553168647, 8.495824721603, -18.37018399522,
    -4.05911912872, -20.95309286149, -9.166993784745, -6.960606318646,
    -2.690928433504, 3.068826044962, 7.480388629445, 17.13642046343,
    4.807300588593, 19.3868826825, -1.954681162013, 15.24607457255,
    -13.65036422915, 2.18379804123, -6.172031594311, -0.9797500764215,
];
#[rustfmt::skip]
const REF_LIN_LOG: [f64; 60] = [
    0.7046737870946, 17.6128364742, 24.46349670634, 6.055848914459,
    6.155245579602, -8.293763928622, -9.897423821392, -18.06174291559,
    -11.19430725453, -25.23282768418, 6.440309362396, -18.13222265083,
    25.18175752969, -3.266070686856, 6.964710778435, 18.46305883417,
    -16.6470322569, 24.77630845913, -17.72025700575, 18.32749854452,
    8.374243166888, 7.400551786405, 23.66395326976, -12.37129069074,
    5.12882742226, -26.14884979255, -22.69135667522, -19.96027123742,
    -13.74923665683, -11.02447615839, 11.45301192595, 0.5421473023272,
    22.29985374181, 15.04370262804, -2.250301912085, 28.01321811598,
    -24.89673138591, 16.55951852612, -15.1102437389, 5.48400233225,
    17.21795243008, -15.72395232717, 18.06591801034, -24.73396902249,
    -4.887597659282, -29.54729847819, -24.10633673608, -8.866964372637,
    -7.606822808738, 2.010480316745, 15.35092660906, 21.17334643597,
    13.16763559446, 28.40286569064, -8.398530633298, 26.19248815677,
    -25.52912599987, 2.997729607478, -4.398847457381, -2.969153313205,
];
#[rustfmt::skip]
const REF_STRONG_GRAVITY: [f64; 60] = [
    0.3660721587054, 4.725013396713, 8.645017551239, 1.612731969361,
    1.838941352383, -2.879293139985, -3.649193978726, -7.371445333651,
    -4.368379436075, -11.52045164517, 1.442638809704, -7.140574473241,
    8.817592048654, -2.513902022601, 3.340981387007, 7.565514097215,
    -5.226814407079, 10.45932622805, -6.28500928245, 6.237377225937,
    4.718666493051, 3.225976827644, 8.606607957538, -7.250075635028,
    2.676491454964, -10.98077269464, -8.485520985778, -8.191730312464,
    -3.960053469006, -4.615628932645, 3.352733892169, 0.3896125178286,
    9.172170000687, 6.131371900208, -0.7498090192513, 12.08333375718,
    -8.914660883602, 8.502986152139, -5.823297422803, 2.1318032716,
    5.366311133539, -4.853546399157, 6.178973868069, -10.37854220389,
    -1.474757600354, -11.4380554081, -8.391417363733, -3.473693753016,
    -3.424644857216, 0.7551871314861, 6.400221179921, 9.446869057021,
    3.21901811307, 11.68045843416, -2.266319916334, 8.552490635978,
    -9.94103327582, 0.8154582087356, -2.139150856854, -1.09872889193,
];
#[rustfmt::skip]
const REF_EDGE_INFLUENCE: [f64; 60] = [
    1.30499445966, 8.547133772146, 12.76880672816, 3.477675363613,
    2.726784378568, -2.825744591557, -4.312328466721, -10.7751127803,
    -5.216209887585, -17.67066817502, 1.682786664755, -11.82308924547,
    10.7577328007, -4.641815973764, 4.445117845808, 13.4245072609,
    -7.444760394295, 17.03745189372, -8.610490463893, 10.65170575117,
    6.395510243814, 6.018098142559, 11.00808813109, -12.10130616742,
    2.443112347913, -17.36843552967, -10.04250479614, -15.53174054696,
    -3.045322438608, -8.798175176017, 3.102125660874, 2.356427062041,
    10.60250496194, 10.61199850154, -1.133509211544, 19.74264173198,
    -11.94226614485, 15.32725995438, -8.170054667774, 0.1068434924078,
    6.295099867346, -8.207365858307, 8.381133480341, -17.96453065548,
    -3.35932265443, -19.61070460773, -9.466912163656, -6.02567118921,
    -0.6838243731935, 3.31063396245, 7.915961342169, 16.07564299148,
    5.410750642001, 18.1641986342, -3.484276632875, 13.09298011235,
    -14.09623104, 1.530943484151, -6.061428167014, -1.68445301497,
];
/// Number of nodes in the parity graph.
const N_PARITY: usize = 30;

/// Weighted ring plus chords from every even node, both directions stored.
fn parity_graph() -> CoordinateList<f64> {
    let mut edges = Vec::new();
    for i in 0..N_PARITY {
        edges.push((i, (i + 1) % N_PARITY, 1.0 + (i % 3) as f64 * 0.5));
        if i % 2 == 0 {
            edges.push((i, (i + 7) % N_PARITY, 0.5 + (i % 5) as f64 * 0.1));
        }
    }
    let mut graph = CoordinateList {
        row_indices: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        n_samples: N_PARITY,
    };
    for (i, j, w) in edges {
        for (a, b) in [(i, j), (j, i)] {
            graph.row_indices.push(a);
            graph.col_indices.push(b);
            graph.values.push(w);
        }
    }
    graph
}

/// Deterministic starting layout shared with the Python script.
fn parity_pos() -> Vec<Vec<f64>> {
    (0..N_PARITY)
        .map(|i| {
            let i = i as f64;
            vec![10.0 * (1.3 * i).sin(), 10.0 * (0.7 * i + 0.5).cos()]
        })
        .collect()
}

/// Run five exact (theta = 0) epochs and compare against the reference.
fn check_parity(params: Fa2OptimParams<f64>, reference: &[f64]) {
    let params = Fa2OptimParams {
        n_epochs: 5,
        theta: 0.0,
        ..params
    };
    let mut embd = parity_pos();
    optimise_fa2(&mut embd, &params, &parity_graph(), 0).unwrap();
    for (i, p) in embd.iter().enumerate() {
        assert_relative_eq!(p[0], reference[2 * i], max_relative = 1e-9, epsilon = 1e-9);
        assert_relative_eq!(
            p[1],
            reference[2 * i + 1],
            max_relative = 1e-9,
            epsilon = 1e-9
        );
    }
}

#[test]
fn fa2_integration_02_parity_linear() {
    check_parity(Fa2OptimParams::default(), &REF_LINEAR);
}

#[test]
fn fa2_integration_03_parity_lin_log() {
    let params = Fa2OptimParams {
        lin_log: true,
        ..Fa2OptimParams::default()
    };
    check_parity(params, &REF_LIN_LOG);
}

#[test]
fn fa2_integration_04_parity_strong_gravity() {
    let params = Fa2OptimParams {
        strong_gravity: true,
        gravity: 0.5,
        scaling_ratio: 3.0,
        ..Fa2OptimParams::default()
    };
    check_parity(params, &REF_STRONG_GRAVITY);
}

#[test]
fn fa2_integration_05_parity_edge_weight_influence() {
    let params = Fa2OptimParams {
        edge_weight_influence: 0.5,
        jitter_tolerance: 0.8,
        ..Fa2OptimParams::default()
    };
    check_parity(params, &REF_EDGE_INFLUENCE);
}

/////////////////
// Integration //
/////////////////

/// Smallest distance between cluster centroids over the largest mean
/// within-cluster distance to the centroid.
fn separation_ratio(embd: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n_clusters = labels.iter().max().unwrap() + 1;
    let mut centres = vec![(0.0, 0.0, 0usize); n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        centres[l].0 += embd[0][i];
        centres[l].1 += embd[1][i];
        centres[l].2 += 1;
    }
    let centres: Vec<(f64, f64)> = centres
        .iter()
        .map(|&(x, y, c)| (x / c as f64, y / c as f64))
        .collect();

    let mut spread = vec![0.0; n_clusters];
    let mut counts = vec![0usize; n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        spread[l] += (embd[0][i] - centres[l].0).hypot(embd[1][i] - centres[l].1);
        counts[l] += 1;
    }
    let max_spread = spread
        .iter()
        .zip(&counts)
        .map(|(s, &c)| s / c as f64)
        .fold(0.0, f64::max);

    let mut min_between = f64::MAX;
    for a in 0..n_clusters {
        for b in (a + 1)..n_clusters {
            let d = (centres[a].0 - centres[b].0).hypot(centres[a].1 - centres[b].1);
            min_between = min_between.min(d);
        }
    }
    min_between / max_spread
}

#[test]
fn fa2_integration_01_graph_construction() {
    let (data, _) = create_diagnostic_data(100, 10, 42);
    let params = Fa2Params::<f64>::default();
    let (graph, knn_idx, _) = construct_fa2_graph(
        data.as_ref(),
        None,
        params.k,
        params.ann_type.clone(),
        &params.graph_params,
        &params.nn_params,
        42,
        0,
    )
    .unwrap();

    assert_eq!(graph.n_samples, 500);
    assert_eq!(knn_idx[0].len(), params.k);

    let mut edges = std::collections::HashMap::new();
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        assert_ne!(i, j, "self-loop at {i}");
        assert!(w > 0.0 && w <= 1.0, "weight {w} outside (0, 1]");
        edges.insert((i, j), w);
    }
    for (&(i, j), &w) in &edges {
        assert_eq!(
            edges.get(&(j, i)),
            Some(&w),
            "edge ({i}, {j}) not symmetric"
        );
    }
    let mut degree = vec![0usize; 500];
    for &i in &graph.row_indices {
        degree[i] += 1;
    }
    assert!(degree.iter().all(|&d| d >= 1), "isolated node in kNN graph");
}

#[test]
fn fa2_integration_06_cluster_separation() {
    let (data, labels) = create_diagnostic_data(200, 20, 42);
    let params = Fa2Params::<f64>::default();
    let embd = forceatlas2(data.as_ref(), None, &params, "bh", 42, 0).unwrap();
    assert_eq!(embd.len(), 2);
    assert_eq!(embd[0].len(), 1_000);
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
    let ratio = separation_ratio(&embd, &labels);
    assert!(ratio > 2.0, "clusters not separated, ratio = {ratio:.3}");
}

#[test]
fn fa2_integration_08_reproducibility() {
    let (data, _) = create_diagnostic_data(100, 10, 7);
    let params = Fa2Params::<f64> {
        optim_params: Fa2OptimParams {
            n_epochs: 100,
            ..Fa2OptimParams::default()
        },
        ..Fa2Params::default()
    };
    let a = forceatlas2(data.as_ref(), None, &params, "bh", 3, 0).unwrap();
    let b = forceatlas2(data.as_ref(), None, &params, "bh", 3, 0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn fa2_integration_09_different_seeds_differ() {
    let (data, _) = create_diagnostic_data(100, 10, 7);
    let params = Fa2Params::<f64> {
        initialisation: "random".to_string(),
        optim_params: Fa2OptimParams {
            n_epochs: 100,
            ..Fa2OptimParams::default()
        },
        ..Fa2Params::default()
    };
    let a = forceatlas2(data.as_ref(), None, &params, "bh", 42, 0).unwrap();
    let b = forceatlas2(data.as_ref(), None, &params, "bh", 123, 0).unwrap();
    let max_diff = a
        .iter()
        .zip(&b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 0.01,
        "different seeds produced identical results"
    );
}

#[test]
fn fa2_integration_10_precomputed_knn() {
    let (data, _) = create_diagnostic_data(100, 10, 11);
    let params = Fa2Params::<f64> {
        ann_type: "exhaustive".to_string(),
        optim_params: Fa2OptimParams {
            n_epochs: 50,
            ..Fa2OptimParams::default()
        },
        ..Fa2Params::default()
    };
    let knn = run_ann_search(
        data.as_ref(),
        params.k,
        params.ann_type.clone(),
        &params.nn_params,
        5,
        0,
    )
    .unwrap();
    let a = forceatlas2(data.as_ref(), Some(knn), &params, "bh", 5, 0).unwrap();
    let b = forceatlas2(data.as_ref(), None, &params, "bh", 5, 0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn fa2_integration_11_from_graph() {
    let graph = parity_graph();
    let embd =
        forceatlas2_from_graph(&graph, None, &Fa2OptimParams::default(), "bh", 1, 0).unwrap();
    assert_eq!(embd[0].len(), N_PARITY);
    assert!(embd.iter().flatten().all(|v| v.is_finite()));

    let mut asym = parity_graph();
    asym.values[0] += 1.0;
    let err = forceatlas2_from_graph(&asym, None, &Fa2OptimParams::default(), "bh", 1, 0);
    assert!(matches!(err, Err(ManifoldsError::AsymmetricGraph { .. })));
}

/////////////////
// Edge cases //
/////////////////

/// Graph with the given undirected edges, both directions stored, unit
/// weights.
fn graph_from_edges<T: ManifoldsFloat>(n: usize, edges: &[(usize, usize)]) -> CoordinateList<T> {
    let mut graph = CoordinateList {
        row_indices: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        n_samples: n,
    };
    for &(i, j) in edges {
        for (a, b) in [(i, j), (j, i)] {
            graph.row_indices.push(a);
            graph.col_indices.push(b);
            graph.values.push(T::one());
        }
    }
    graph
}

#[test]
fn fa2_integration_12_dissuade_hubs_regular_graph() {
    // every node has degree 2, so mean mass / own mass is 1 everywhere
    let edges: Vec<(usize, usize)> = (0..N_PARITY).map(|i| (i, (i + 1) % N_PARITY)).collect();
    let graph = graph_from_edges::<f64>(N_PARITY, &edges);
    let init = parity_pos();
    let run = |dissuade_hubs| {
        let params = Fa2OptimParams {
            n_epochs: 50,
            dissuade_hubs,
            ..Fa2OptimParams::default()
        };
        let mut embd = init.clone();
        optimise_fa2(&mut embd, &params, &graph, 0).unwrap();
        embd
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn fa2_integration_13_single_node_f32() {
    // zero force every epoch, so zero swing; the speed must not blow up
    let graph = graph_from_edges::<f32>(1, &[]);
    let embd = forceatlas2_from_graph(
        &graph,
        Some(vec![vec![0.0], vec![0.0]]),
        &Fa2OptimParams::default(),
        "bh",
        1,
        0,
    )
    .unwrap();
    assert_eq!(embd, vec![vec![0.0], vec![0.0]]);
}

#[test]
fn fa2_integration_14_no_edges() {
    let graph = graph_from_edges::<f64>(20, &[]);
    let embd =
        forceatlas2_from_graph(&graph, None, &Fa2OptimParams::default(), "bh", 3, 0).unwrap();
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn fa2_integration_15_self_loops_dropped() {
    let mut with_loops = parity_graph();
    for i in [0, 5, 17] {
        with_loops.row_indices.push(i);
        with_loops.col_indices.push(i);
        with_loops.values.push(2.0);
    }
    let init: Vec<Vec<f64>> = {
        let p = parity_pos();
        vec![
            p.iter().map(|q| q[0]).collect(),
            p.iter().map(|q| q[1]).collect(),
        ]
    };
    let params = Fa2OptimParams {
        n_epochs: 20,
        ..Fa2OptimParams::default()
    };
    let a =
        forceatlas2_from_graph(&parity_graph(), Some(init.clone()), &params, "bh", 1, 0).unwrap();
    let b = forceatlas2_from_graph(&with_loops, Some(init), &params, "bh", 1, 0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn fa2_integration_16_out_of_bounds_edge() {
    let mut graph = parity_graph();
    graph.row_indices.push(0);
    graph.col_indices.push(N_PARITY);
    graph.values.push(1.0);
    let err = forceatlas2_from_graph(&graph, None, &Fa2OptimParams::default(), "bh", 1, 0);
    assert!(matches!(
        err,
        Err(ManifoldsError::AsymmetricGraph { row: 0, col }) if col == N_PARITY
    ));
}

#[test]
fn fa2_integration_17_init_size_mismatch() {
    let init = vec![vec![0.0; N_PARITY], vec![0.0; N_PARITY - 3]];
    let err = forceatlas2_from_graph(
        &parity_graph(),
        Some(init),
        &Fa2OptimParams::default(),
        "bh",
        1,
        0,
    );
    assert!(matches!(
        err,
        Err(ManifoldsError::GraphSizeMismatch { n_graph, n_embd })
            if n_graph == N_PARITY && n_embd == N_PARITY - 3
    ));
}

#[test]
fn fa2_integration_18_invalid_params() {
    let bad = [
        Fa2OptimParams {
            scaling_ratio: -1.0,
            ..Fa2OptimParams::default()
        },
        Fa2OptimParams {
            jitter_tolerance: 0.0,
            ..Fa2OptimParams::default()
        },
        Fa2OptimParams {
            gravity: f64::NAN,
            ..Fa2OptimParams::default()
        },
        Fa2OptimParams {
            theta: -0.1,
            ..Fa2OptimParams::default()
        },
        Fa2OptimParams {
            edge_weight_influence: -1.0,
            ..Fa2OptimParams::default()
        },
    ];
    for params in &bad {
        let err = forceatlas2_from_graph(&parity_graph(), None, params, "bh", 1, 0);
        assert!(matches!(err, Err(ManifoldsError::Fa2InvalidParam { .. })));
    }
}

#[test]
fn fa2_integration_19_directed_mix_weight() {
    let (data, _) = create_diagnostic_data(20, 5, 1);
    let mut params = Fa2Params::<f64>::default();
    params.graph_params.mix_weight = 0.5;
    let err = forceatlas2(data.as_ref(), None, &params, "bh", 1, 0);
    assert!(matches!(
        err,
        Err(ManifoldsError::Fa2InvalidParam {
            param: "graph_params.mix_weight",
            ..
        })
    ));
}

#[test]
fn fa2_integration_07_cluster_separation_f32() {
    let (data, labels) = create_diagnostic_data(200, 20, 42);
    let data = mat_to_f32(data);
    let params = Fa2Params::<f32>::default();
    let embd = forceatlas2(data.as_ref(), None, &params, "bh", 42, 0).unwrap();
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
    let embd: Vec<Vec<f64>> = embd
        .iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    let ratio = separation_ratio(&embd, &labels);
    assert!(ratio > 2.0, "clusters not separated, ratio = {ratio:.3}");
}

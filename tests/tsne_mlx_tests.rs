#![cfg(mlx_available)]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::prelude::*;
use manifolds_rs::*;

/////////////
// Helpers //
/////////////

/// t-SNE parameters for the small diagnostic set, exhaustive kNN.
///
/// ### Params
///
/// * `lr` - Learning rate
/// * `n_epochs` - Epochs
///
/// ### Returns
///
/// The parameters
fn tsne_params(lr: f32, n_epochs: usize) -> TsneParams<f32> {
    let mut params = TsneParams {
        perplexity: 20.0,
        ann_type: "exhaustive".to_string(),
        init_range: Some(1e-4),
        ..TsneParams::default()
    };
    params.optim_params.lr = Some(lr);
    params.optim_params.n_epochs = n_epochs;
    params
}

/// Smallest centroid distance over the mean within-cluster spread.
///
/// ### Params
///
/// * `embd` - Embedding `[2][n]`
/// * `labels` - Cluster label per point
///
/// ### Returns
///
/// `min_inter / mean_intra`.
fn separation(embd: &[Vec<f32>], labels: &[usize]) -> f64 {
    let k = labels.iter().max().unwrap() + 1;
    let mut c = vec![[0.0f64; 3]; k];
    for (i, &l) in labels.iter().enumerate() {
        c[l][0] += embd[0][i] as f64;
        c[l][1] += embd[1][i] as f64;
        c[l][2] += 1.0;
    }
    let cent: Vec<(f64, f64)> = c.iter().map(|v| (v[0] / v[2], v[1] / v[2])).collect();
    let mut intra = vec![0.0f64; k];
    for (i, &l) in labels.iter().enumerate() {
        intra[l] += ((embd[0][i] as f64 - cent[l].0).powi(2)
            + (embd[1][i] as f64 - cent[l].1).powi(2))
        .sqrt();
    }
    let mean_intra = (0..k).map(|l| intra[l] / c[l][2]).sum::<f64>() / k as f64;
    let mut min_inter = f64::INFINITY;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = ((cent[i].0 - cent[j].0).powi(2) + (cent[i].1 - cent[j].1).powi(2)).sqrt();
            min_inter = min_inter.min(d);
        }
    }
    min_inter / mean_intra
}

/// Share of the embedding's variance explained by an affine map of the
/// reference, `R^2` of the least-squares fit `embd ~ A ref + b`.
///
/// ### Params
///
/// * `embd` - Embedding `[2][n]`
/// * `reference` - Reference `[2][n]`
///
/// ### Returns
///
/// `R^2` in `[0, 1]`.
fn affine_r2(embd: &[Vec<f32>], reference: &[Vec<f32>]) -> f64 {
    let n = embd[0].len();
    let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let (rx, ry) = (mean(&reference[0]), mean(&reference[1]));
    // normal equations of the 2x2 fit on centred data
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (reference[0][i] as f64 - rx, reference[1][i] as f64 - ry);
        sxx += a * a;
        sxy += a * b;
        syy += b * b;
    }
    let det = sxx * syy - sxy * sxy;
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for d in 0..2 {
        let ym = mean(&embd[d]);
        let (mut ta, mut tb) = (0.0, 0.0);
        for i in 0..n {
            let y = embd[d][i] as f64 - ym;
            ta += (reference[0][i] as f64 - rx) * y;
            tb += (reference[1][i] as f64 - ry) * y;
        }
        let ca = (syy * ta - sxy * tb) / det;
        let cb = (sxx * tb - sxy * ta) / det;
        for i in 0..n {
            let y = embd[d][i] as f64 - ym;
            let fit = ca * (reference[0][i] as f64 - rx) + cb * (reference[1][i] as f64 - ry);
            ss_res += (y - fit).powi(2);
            ss_tot += y * y;
        }
    }
    1.0 - ss_res / ss_tot
}

///////////
// Tests //
///////////

/// Test 1: MLX three-kernel FFT t-SNE separates the diagnostic clusters
#[test]
fn tsne_mlx_integration_01_quality() {
    let (data, labels) = create_diagnostic_data(100, 10, 123);
    let data = mat_to_f32(data);

    let embd = tsne(
        data.as_ref(),
        None,
        &tsne_params(100.0, 500),
        "fft_3k_mlx",
        42,
        0,
    )
    .unwrap();
    assert!(embd.iter().flatten().all(|x| x.is_finite()));
    let sep = separation(&embd, &labels);
    println!("MLX three-kernel FFT separation ratio: {sep:.2}");
    assert!(sep > 1.0);
}

/// Test 2: Same seed, same embedding
#[test]
fn tsne_mlx_integration_02_reproducibility() {
    let (data, _) = create_diagnostic_data(100, 10, 42);
    let data = mat_to_f32(data);

    let params = tsne_params(200.0, 200);
    let a = tsne(data.as_ref(), None, &params, "fft_3k_mlx", 123, 0).unwrap();
    let b = tsne(data.as_ref(), None, &params, "fft_3k_mlx", 123, 0).unwrap();
    assert_eq!(a, b, "same seed, different MLX t-SNE embedding");
}

/// Test 3: DREAMS on MLX pulls the layout towards the reference
#[test]
fn tsne_mlx_integration_03_dreams_follows_reference() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let data = mat_to_f32(data);
    let n = data.nrows();
    // the first two input features as the reference layout
    let reference: Vec<Vec<f32>> = (0..2)
        .map(|d| (0..n).map(|i| data[(i, d)]).collect())
        .collect();

    let plain = tsne(
        data.as_ref(),
        None,
        &tsne_params(200.0, 500),
        "fft_3k_mlx",
        42,
        0,
    )
    .unwrap();
    let params = DreamsParams::new(tsne_params(200.0, 500), 0.5f32);
    let reg = dreams(
        data.as_ref(),
        None,
        Some(reference.clone()),
        &params,
        "fft_3k_mlx",
        42,
        0,
    )
    .unwrap();
    assert!(reg.iter().flatten().all(|x| x.is_finite()));

    let (r2_plain, r2_dreams) = (affine_r2(&plain, &reference), affine_r2(&reg, &reference));
    println!(
        "R^2 vs reference: t-SNE {r2_plain:.3}, DREAMS {r2_dreams:.3}; separation {:.2}",
        separation(&reg, &labels)
    );
    assert!(r2_dreams > r2_plain);
}

/// Test 4: den-SNE is refused rather than silently dropping the density term
#[test]
fn tsne_mlx_integration_04_densne_refused() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    let params = DensneParams {
        tsne_params: tsne_params(200.0, 50),
        ..DensneParams::default()
    };
    let res = densne(data.as_ref(), None, &params, "fft_3k_mlx", 42, 0);
    assert!(matches!(
        res,
        Err(ManifoldsError::DensityNotSupported { .. })
    ));
}

//! Flexible matrix input tests.
//!
//! The point of these is that the representation a caller happens to hold must
//! not change the answer. Every test builds the same data in several shapes and
//! asserts they agree.

mod commons;
use commons::*;

use approx::assert_relative_eq;
use faer::Mat;
use manifolds_rs::prelude::*;
use manifolds_rs::*;

/// Flatten a faer matrix to a row-major buffer.
///
/// ### Params
///
/// * `mat` - Matrix to flatten
///
/// ### Returns
///
/// Row-major `Vec` of length `nrows * ncols`.
fn to_row_major(mat: &Mat<f64>) -> Vec<f64> {
    let (n, dim) = (mat.nrows(), mat.ncols());
    (0..n)
        .flat_map(|i| (0..dim).map(move |j| mat[(i, j)]))
        .collect()
}

/// Assert two embeddings are elementwise equal.
///
/// ### Params
///
/// * `a` - First embedding
/// * `b` - Second embedding
fn assert_embeddings_eq(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len(), "component count differs");
    for (ca, cb) in a.iter().zip(b.iter()) {
        assert_eq!(ca.len(), cb.len(), "sample count differs");
        for (x, y) in ca.iter().zip(cb.iter()) {
            assert_relative_eq!(x, y, epsilon = 1e-12);
        }
    }
}

/// Small, fast UMAP parameters for the equivalence tests.
///
/// ### Returns
///
/// Parameters using exhaustive kNN and random init so the run is deterministic
/// and quick.
fn umap_test_params() -> UmapParams<f64> {
    UmapParams::<f64> {
        k: 8,
        ann_type: "exhaustive".to_string(),
        initialisation: "random".to_string(),
        optim_params: UmapOptimParams {
            n_epochs: 20,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Small, fast tSNE parameters for the equivalence tests.
///
/// ### Returns
///
/// Parameters using exhaustive kNN and a short optimisation run.
fn tsne_test_params() -> TsneParams<f64> {
    TsneParams::<f64> {
        perplexity: 10.0,
        ann_type: "exhaustive".to_string(),
        optim_params: TsneOptimParams {
            n_epochs: 50,
            ..Default::default()
        },
        ..Default::default()
    }
}

///////////////////////
// Matrix conversion //
///////////////////////

/// The faer matrix and the row-major triple must describe the same matrix.
#[test]
fn input_faer_and_flat_triple_agree() {
    let (data, _) = create_diagnostic_data(20, 6, 42);
    let flat = to_row_major(&data);
    let (n, dim) = (data.nrows(), data.ncols());

    let mat = data.as_ref();
    let triple = (flat.as_slice(), n, dim);
    let from_faer = mat.to_mat_input();
    let from_flat = triple.to_mat_input();

    let a = from_faer.as_mat_ref();
    let b = from_flat.as_mat_ref();

    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
    for i in 0..n {
        for j in 0..dim {
            assert_relative_eq!(a[(i, j)], b[(i, j)]);
        }
    }
}

/// The owned flat triple must match the borrowed one.
#[test]
fn input_owned_flat_triple_matches_borrowed() {
    let (data, _) = create_diagnostic_data(10, 4, 7);
    let flat = to_row_major(&data);
    let (n, dim) = (data.nrows(), data.ncols());

    let owned_src = (flat.clone(), n, dim);
    let borrowed_src = (flat.as_slice(), n, dim);
    let owned = owned_src.to_mat_input();
    let borrowed = borrowed_src.to_mat_input();

    for i in 0..n {
        for j in 0..dim {
            assert_relative_eq!(owned.as_mat_ref()[(i, j)], borrowed.as_mat_ref()[(i, j)]);
        }
    }
}

/// A flat buffer whose length disagrees with the stated shape is a caller bug
/// and must not be silently reinterpreted.
#[test]
#[should_panic(expected = "does not match shape")]
fn input_flat_triple_rejects_wrong_shape() {
    let data = vec![0.0_f64; 12];
    let src = (data.as_slice(), 5, 3);
    let _ = src.to_mat_input();
}

//////////////////
// End-to-end //
//////////////////

/// UMAP must return the same embedding whichever representation carries the
/// data in.
#[test]
fn input_umap_agrees_across_representations() {
    let (data, _) = create_diagnostic_data(20, 6, 42);
    let flat = to_row_major(&data);
    let (n, dim) = (data.nrows(), data.ncols());

    let params = umap_test_params();

    let from_faer = umap(data.as_ref(), None, &params, 42, 0).unwrap();
    let from_flat = umap((flat.as_slice(), n, dim), None, &params, 42, 0).unwrap();
    let from_owned = umap((flat.clone(), n, dim), None, &params, 42, 0).unwrap();
    let from_mat = umap(&data, None, &params, 42, 0).unwrap();

    assert_embeddings_eq(&from_faer, &from_flat);
    assert_embeddings_eq(&from_faer, &from_owned);
    assert_embeddings_eq(&from_faer, &from_mat);
}

/// tSNE likewise, to confirm the conversion sits at the entry point rather than
/// in any one algorithm.
#[test]
fn input_tsne_agrees_across_representations() {
    let (data, _) = create_diagnostic_data(15, 5, 11);
    let flat = to_row_major(&data);
    let (n, dim) = (data.nrows(), data.ncols());

    let params = tsne_test_params();

    let from_faer = tsne(data.as_ref(), None, &params, "barnes_hut", 42, 0).unwrap();
    let from_flat = tsne(
        (flat.as_slice(), n, dim),
        None,
        &params,
        "barnes_hut",
        42,
        0,
    )
    .unwrap();

    assert_embeddings_eq(&from_faer, &from_flat);
}

/////////////
// ndarray //
/////////////

#[cfg(feature = "ndarray")]
mod ndarray_inputs {
    use super::*;
    use ndarray::Array2;

    /// A standard-layout array, its view, and a faer matrix must all agree.
    #[test]
    fn input_ndarray_matches_faer() {
        let (data, _) = create_diagnostic_data(15, 5, 3);
        let flat = to_row_major(&data);
        let (n, dim) = (data.nrows(), data.ncols());

        let arr = Array2::from_shape_vec((n, dim), flat).unwrap();

        let mat = data.as_ref();
        let view = arr.view();
        let from_faer = mat.to_mat_input();
        let from_view = view.to_mat_input();
        let from_owned = arr.to_mat_input();

        for i in 0..n {
            for j in 0..dim {
                let expected = from_faer.as_mat_ref()[(i, j)];
                assert_relative_eq!(from_view.as_mat_ref()[(i, j)], expected);
                assert_relative_eq!(from_owned.as_mat_ref()[(i, j)], expected);
            }
        }
    }

    /// A transposed view is not contiguous. It must be materialised in logical
    /// order, not reinterpreted as the underlying column-major buffer.
    #[test]
    fn input_transposed_ndarray_view_is_not_reinterpreted() {
        let (data, _) = create_diagnostic_data(10, 4, 5);
        let (n, dim) = (data.nrows(), data.ncols());

        // store column-major, then transpose the view back to samples x
        // features so it is strided rather than contiguous
        let col_major: Vec<f64> = (0..dim)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| data[(i, j)])
            .collect();
        let arr = Array2::from_shape_vec((dim, n), col_major).unwrap();
        let view = arr.t();

        assert_eq!(view.dim(), (n, dim));
        assert!(view.to_slice().is_none(), "expected a strided view");

        let input = view.to_mat_input();
        for i in 0..n {
            for j in 0..dim {
                assert_relative_eq!(input.as_mat_ref()[(i, j)], data[(i, j)]);
            }
        }
    }

    /// UMAP through an ndarray view must match UMAP through faer.
    #[test]
    fn input_umap_agrees_for_ndarray() {
        let (data, _) = create_diagnostic_data(20, 6, 42);
        let flat = to_row_major(&data);
        let (n, dim) = (data.nrows(), data.ncols());
        let arr = Array2::from_shape_vec((n, dim), flat).unwrap();

        let params = umap_test_params();

        let from_faer = umap(data.as_ref(), None, &params, 42, 0).unwrap();
        let from_view = umap(arr.view(), None, &params, 42, 0).unwrap();

        assert_embeddings_eq(&from_faer, &from_view);
    }
}

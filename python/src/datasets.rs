//! Synthetic data with the structure these algorithms are meant to find.
//!
//! The crate's own generators, bound as-is. Random normal noise tells you
//! nothing about whether an embedding worked: a swiss roll has a known
//! unrolling, a branching trajectory has known lineages, and hierarchical
//! clusters have a known two-level structure. Every one of these hands back the
//! labels alongside the matrix so a caller can score the result.
//!
//! float64 throughout, because the generators are. Cast on the Python side if
//! you want to feed a GPU path.

use manifolds_rs::prelude::{
    generate_clustered_data, generate_example_branches, generate_hierarchical_clusters,
    generate_swiss_roll_biased, generate_trajectory, parse_topology,
};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::mat_to_numpy;

///////////
// Types //
///////////

/// A design matrix with one integer label per sample.
type Labelled<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>);

/// A design matrix with one continuous parameter per sample.
type Parameterised<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);

/// A design matrix with a coarse and a fine label per sample.
type TwoLevel<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// A 2-D manifold rolled up in 3-D, the standard unrolling benchmark.
///
/// ### Params
///
/// * `n_samples` - Number of points.
/// * `noise` - Standard deviation of the Gaussian noise added to the surface.
/// * `density_bias` - Exponent applied to the uniform sample along the roll.
///   `1.0` is neutral and samples uniformly; higher values pile points up at
///   the inner end, which is what makes an embedding's density handling
///   visible. Note that `0.0` is degenerate rather than uniform: it collapses
///   every point onto the same `t`. The crate's own doc comment says otherwise
///   and is wrong.
/// * `seed` - Fixes the sampling.
///
/// ### Returns
///
/// `(X, t)` where `X` is `(n_samples, 3)` and `t` is the position along the
/// roll for each point, which is the ground truth an unrolling should recover.
///
/// Always the biased generator, at zero bias when none was asked for: the
/// unbiased one does not report `t`, and a caller with no ground truth cannot
/// score anything.
#[pyfunction]
#[pyo3(signature = (n_samples = 5000, *, noise = 0.05, density_bias = 1.0, seed = 42))]
pub fn swiss_roll(
    py: Python<'_>,
    n_samples: usize,
    noise: f64,
    density_bias: f64,
    seed: u64,
) -> PyResult<Parameterised<'_>> {
    let (mat, t) = generate_swiss_roll_biased(n_samples, noise, density_bias, seed);
    Ok((mat_to_numpy(py, &mat)?, t.into_pyarray(py)))
}

/// Isotropic Gaussian clusters of varying size and spread.
///
/// ### Params
///
/// * `n_samples` - Number of points.
/// * `dim` - Ambient dimensionality.
/// * `n_clusters` - Number of clusters.
/// * `seed` - Fixes the centres and the sampling.
///
/// ### Returns
///
/// `(X, labels)` where `X` is `(n_samples, dim)` and `labels` holds the cluster
/// index of each point.
#[pyfunction]
#[pyo3(signature = (n_samples = 5000, *, dim = 50, n_clusters = 10, seed = 42))]
pub fn clustered(
    py: Python<'_>,
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> PyResult<Labelled<'_>> {
    let (mat, labels) = generate_clustered_data(n_samples, dim, n_clusters, seed);
    Ok((mat_to_numpy(py, &mat)?, labels_to_numpy(py, labels)))
}

/// A branching differentiation trajectory.
///
/// ### Params
///
/// * `n_samples` - Number of points, split evenly across branches.
/// * `topology` - `"bifurcation"` for a cascading tree, `"linear"` for one
///   continuous lineage, `"combination"` for a backbone with branches leaving
///   it mid-way.
/// * `dim` - Ambient dimensionality. Must be at least the branch count.
/// * `noise` - Base noise standard deviation, scaled up along pseudotime.
/// * `seed` - Fixes the sampling.
///
/// ### Returns
///
/// `(X, branch)` where `X` is `(n_samples, dim)` and `branch` holds the branch
/// index of each point.
#[pyfunction]
#[pyo3(signature = (n_samples = 5000, *, topology = "bifurcation", dim = 50, noise = 0.1, seed = 42))]
pub fn trajectory<'py>(
    py: Python<'py>,
    n_samples: usize,
    topology: &str,
    dim: usize,
    noise: f64,
    seed: u64,
) -> PyResult<Labelled<'py>> {
    let topo = parse_topology(topology).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown topology {topology:?}; expected 'bifurcation', 'linear' or 'combination'"
        ))
    })?;
    let branches = generate_example_branches(&topo);
    if dim < branches.len() {
        return Err(PyValueError::new_err(format!(
            "dim must be at least the branch count ({}), got {dim}",
            branches.len()
        )));
    }
    let (mat, labels) = generate_trajectory(n_samples, &branches, dim, noise, seed);
    Ok((mat_to_numpy(py, &mat)?, labels_to_numpy(py, labels)))
}

/// Clusters within clusters, for testing whether global structure survives.
///
/// ### Params
///
/// * `n_samples` - Number of points.
/// * `dim` - Ambient dimensionality.
/// * `n_supergroups` - Number of top-level groups.
/// * `n_subclusters` - Subclusters within each group.
/// * `supergroup_spread` - How far apart the group centres sit.
/// * `subcluster_spread` - How far apart subcluster centres sit within a group.
/// * `point_std` - Spread of the points around a subcluster centre.
/// * `seed` - Fixes the centres and the sampling.
///
/// ### Returns
///
/// `(X, supergroup, subcluster)`. An embedding that keeps the supergroups apart
/// while resolving the subclusters has preserved both scales, which is the
/// thing global-structure claims are usually made about.
#[pyfunction]
#[pyo3(signature = (
    n_samples = 5000,
    *,
    dim = 50,
    n_supergroups = 4,
    n_subclusters = 5,
    supergroup_spread = 15.0,
    subcluster_spread = 3.0,
    point_std = 0.5,
    seed = 42
))]
#[allow(clippy::too_many_arguments)]
pub fn hierarchical(
    py: Python<'_>,
    n_samples: usize,
    dim: usize,
    n_supergroups: usize,
    n_subclusters: usize,
    supergroup_spread: f64,
    subcluster_spread: f64,
    point_std: f64,
    seed: u64,
) -> PyResult<TwoLevel<'_>> {
    let (mat, supergroup, subcluster) = generate_hierarchical_clusters(
        n_samples,
        dim,
        n_supergroups,
        n_subclusters,
        supergroup_spread,
        subcluster_spread,
        point_std,
        seed,
    );
    Ok((
        mat_to_numpy(py, &mat)?,
        labels_to_numpy(py, supergroup),
        labels_to_numpy(py, subcluster),
    ))
}

/////////////
// Helpers //
/////////////

/// Move a label vector into an int64 numpy array.
///
/// int64 rather than uint64 because that is what scikit-learn's scorers and
/// numpy's own indexing expect, and a label count never approaches the range
/// where the sign bit matters.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `labels` - One label per sample.
///
/// ### Returns
///
/// A 1-D int64 array. The conversion allocates; `into_pyarray` then moves.
fn labels_to_numpy(py: Python<'_>, labels: Vec<usize>) -> Bound<'_, PyArray1<i64>> {
    labels
        .into_iter()
        .map(|v| v as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py)
}

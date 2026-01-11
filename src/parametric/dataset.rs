use burn::{
    data::dataset::Dataset,
    prelude::*,
    tensor::{Element, Int},
};
use num_traits::{Float, FromPrimitive};

////////////
// Losses //
////////////

/// Compute UMAP loss for a batch
///
/// ### Params
///
/// * `src_embed` - Source node embeddings [batch_size, n_components]
/// * `dst_embed` - Destination node embeddings [batch_size, n_components]
/// * `targets` - Target values (1.0 for edges in graph, 0.0 for negatives)
/// * `a` - Curve parameter for attractive force (typically ~1.5 for 2D)
/// * `b` - Curve parameter for repulsive force (typically ~0.9 for 2D)
///
/// ### Returns
///
/// Binary cross-entropy loss between predicted and target probabilities
pub fn umap_loss<B, T>(
    src_embed: Tensor<B, 2>,
    dst_embed: Tensor<B, 2>,
    targets: Tensor<B, 1>,
    a: T,
    b: T,
) -> Tensor<B, 1>
where
    B: Backend,
    T: Float + FromPrimitive + Element,
{
    // compute squared distances
    let diff: Tensor<B, 2> = src_embed - dst_embed;
    // use ::<1> turbofish syntax <- important!
    let dist_sq: Tensor<B, 1> = diff.clone().powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]);

    let epsilon = T::from_f32(1e-6).unwrap();
    let dist_sq_safe = dist_sq + Tensor::from_floats([epsilon], &diff.device());

    // UMAP probability: 1 / (1 + a * dist^(2b))
    let a_tensor: Tensor<B, 1> = Tensor::from_floats([a], &diff.device());
    let power: Tensor<B, 1> = dist_sq_safe.powf_scalar(b);
    let qs: Tensor<B, 1> = (Tensor::ones_like(&power) + a_tensor * power).powf_scalar(-1.0);

    // binary cross-entropy: -[t*log(q) + (1-t)*log(1-q)]
    let eps = 1e-7;
    let qs_clamp: Tensor<B, 1> = qs.clamp(eps, 1.0 - eps);
    let log_qs: Tensor<B, 1> = qs_clamp.clone().log();
    let log_one_minus_qs: Tensor<B, 1> = (Tensor::ones_like(&qs_clamp) - qs_clamp).log();

    let bce: Tensor<B, 1> =
        -(targets.clone() * log_qs + (Tensor::ones_like(&targets) - targets) * log_one_minus_qs);

    bce.mean()
}

/// Compute negative Pearson correlation between input and embedding distances
///
/// Encourages embedding distances to correlate with original space distances.
///
/// ### Params
///
/// * `x_dist` - Distances in original space [batch_size]
/// * `z_dist` - Distances in embedding space [batch_size]
///
/// ### Returns
///
/// Negative correlation (to minimise during training)
pub fn correlation_loss<B: Backend>(x_dist: Tensor<B, 1>, z_dist: Tensor<B, 1>) -> Tensor<B, 1> {
    let epsilon = 1e-8;

    let x_mean = x_dist.clone().mean();
    let z_mean = z_dist.clone().mean();

    let x_centered = x_dist - x_mean;
    let z_centered = z_dist - z_mean;

    let numerator = (x_centered.clone() * z_centered.clone()).mean();
    let x_std = (x_centered.powf_scalar(2.0).mean()).sqrt();
    let z_std = (z_centered.powf_scalar(2.0).mean()).sqrt();

    let correlation = numerator / (x_std * z_std + epsilon);

    -correlation
}

/// Full UMAP loss with correlation regularisation
///
/// Combines standard UMAP loss with a correlation-based regularisation term
/// that encourages embedding distances to correlate with original space
/// distances.
///
/// ### Params
///
/// * `src_embed` - Source node embeddings [batch_size, n_components]
/// * `dst_embed` - Destination node embeddings [batch_size, n_components]
/// * `src_orig` - Original high-dimensional data [batch_size, n_features]
/// * `dst_orig` - Original high-dimensional data for destination nodes
///   [batch_size, n_features]
/// * `targets` - Target values (1.0 for edges in graph, 0.0 for negatives)
/// * `a` - Curve parameter for attractive force (typically ~1.5 for 2D)
/// * `b` - Curve parameter for repulsive force (typically ~0.9 for 2D)
/// * `correlation_weight` - Weight for correlation regularisation term
///
/// ### Returns
///
/// Combined loss: UMAP BCE loss + correlation_weight * correlation_loss
#[allow(clippy::too_many_arguments)]
pub fn umap_loss_with_correlation<B, T>(
    src_embed: Tensor<B, 2>,
    dst_embed: Tensor<B, 2>,
    src_orig: Tensor<B, 2>, // original high-dim data
    dst_orig: Tensor<B, 2>,
    targets: Tensor<B, 1>,
    a: T,
    b: T,
    correlation_weight: T,
) -> Tensor<B, 1>
where
    B: Backend,
    T: Float + FromPrimitive + Element,
{
    let umap = umap_loss(src_embed.clone(), dst_embed.clone(), targets, a, b);

    // Distances in embedding space
    let z_diff = src_embed - dst_embed;
    let z_dist = z_diff.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt();

    // Distances in original space
    let x_diff = src_orig - dst_orig;
    let x_dist = x_diff.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).sqrt();

    let corr = correlation_loss(x_dist, z_dist);

    umap.clone() + Tensor::from_floats([correlation_weight], &umap.device()) * corr
}

//////////////
// Data set //
//////////////

/// Batch for UMAP edge training
///
/// ### Structure
///
/// * `src_indices` - Source node
/// * `dst_indices` - Destination node
/// * `targets` - 1.0 for positive edge, 0.0 for negative edge
#[derive(Clone, Debug)]
pub struct UmapBatch<B: Backend> {
    pub src_indices: Tensor<B, 1, Int>,
    pub dst_indices: Tensor<B, 1, Int>,
    pub targets: Tensor<B, 1>,
}

/// UmapEdgeDataset
///
/// ### Fields
///
/// * `edges` - Vector of edges with `(src, dist, target)`
pub struct UmapEdgeDataset {
    pub edges: Vec<(usize, usize, f32)>,
}

impl Dataset<(usize, usize, f32)> for UmapEdgeDataset {
    /// Get the indices and the targets
    ///
    /// ### Params
    ///
    /// * `index` - Index of the data
    ///
    /// ### Returns
    ///
    /// Tuple of `(src, dst, target)`
    fn get(&self, index: usize) -> Option<(usize, usize, f32)> {
        self.edges.get(index).copied()
    }

    /// Get the total data set size
    ///
    /// ### Returns
    ///
    /// Get the number of edges
    fn len(&self) -> usize {
        self.edges.len()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod dataset_tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_umap_loss_shape() {
        let device = NdArrayDevice::Cpu;

        let src = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [1.0, 1.0]], &device);
        let dst = Tensor::<TestBackend, 2>::from_floats([[0.1, 0.1], [1.1, 1.1]], &device);
        let targets = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);

        let loss = umap_loss(src, dst, targets, 1.5f32, 0.9f32);

        assert_eq!(loss.dims().len(), 1);
        assert_eq!(loss.dims()[0], 1);
    }

    #[test]
    fn test_umap_loss_is_finite() {
        let device = NdArrayDevice::Cpu;

        let src = Tensor::<TestBackend, 2>::random(
            [10, 5],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let dst = Tensor::<TestBackend, 2>::random(
            [10, 5],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let targets = Tensor::<TestBackend, 1>::from_floats([1.0; 10], &device);

        let loss = umap_loss(src, dst, targets, 1.5f32, 0.9f32);
        let loss_val: f32 = loss.into_scalar().elem();

        assert!(loss_val.is_finite(), "Loss should be finite");
        assert!(loss_val >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_umap_loss_identical_embeddings() {
        let device = NdArrayDevice::Cpu;

        let src = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);
        let dst = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);
        let targets = Tensor::<TestBackend, 1>::from_floats([1.0], &device);

        let loss = umap_loss(src, dst, targets, 1.5f32, 0.9f32);
        let loss_val: f32 = loss.into_scalar().elem();

        // Identical points with target=1.0 should have low loss
        assert!(
            loss_val < 0.1,
            "Loss for identical positive pair should be small"
        );
    }

    #[test]
    fn test_umap_loss_distant_embeddings() {
        let device = NdArrayDevice::Cpu;

        let src = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0]], &device);
        let dst = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0]], &device);
        let targets = Tensor::<TestBackend, 1>::from_floats([0.0], &device);

        let loss = umap_loss(src, dst, targets, 1.5f32, 0.9f32);
        let loss_val: f32 = loss.into_scalar().elem();

        // Distant points with target=0.0 should have low loss
        assert!(
            loss_val < 0.5,
            "Loss for distant negative pair should be reasonable"
        );
    }

    #[test]
    fn test_correlation_loss_shape() {
        let device = NdArrayDevice::Cpu;

        let x_dist = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let z_dist = Tensor::<TestBackend, 1>::from_floats([1.1, 2.1, 3.1], &device);

        let loss = correlation_loss(x_dist, z_dist);

        assert_eq!(loss.dims().len(), 1);
        assert_eq!(loss.dims()[0], 1);
    }

    #[test]
    fn test_correlation_loss_perfect_correlation() {
        let device = NdArrayDevice::Cpu;

        let x_dist = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let z_dist = Tensor::<TestBackend, 1>::from_floats([2.0, 4.0, 6.0, 8.0], &device);

        let loss = correlation_loss(x_dist, z_dist);
        let loss_val: f32 = loss.into_scalar().elem();

        // Perfect correlation should give loss close to -1.0
        assert!(
            loss_val < -0.99,
            "Perfect correlation should give loss ~-1.0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_correlation_loss_no_correlation() {
        let device = NdArrayDevice::Cpu;

        let x_dist = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let z_dist = Tensor::<TestBackend, 1>::from_floats([4.0, 3.0, 2.0, 1.0], &device);

        let loss = correlation_loss(x_dist, z_dist);
        let loss_val: f32 = loss.into_scalar().elem();

        // Negative correlation should give positive loss
        assert!(
            loss_val > 0.99,
            "Negative correlation should give positive loss, got {}",
            loss_val
        );
    }

    #[test]
    fn test_umap_loss_with_correlation_shape() {
        let device = NdArrayDevice::Cpu;

        let src_embed = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [1.0, 1.0]], &device);
        let dst_embed = Tensor::<TestBackend, 2>::from_floats([[0.1, 0.1], [1.1, 1.1]], &device);
        let src_orig =
            Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], &device);
        let dst_orig =
            Tensor::<TestBackend, 2>::from_floats([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]], &device);
        let targets = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);

        let loss = umap_loss_with_correlation(
            src_embed, dst_embed, src_orig, dst_orig, targets, 1.5f32, 0.9f32, 0.1f32,
        );

        assert_eq!(loss.dims().len(), 1);
        assert_eq!(loss.dims()[0], 1);
    }

    #[test]
    fn test_umap_loss_with_correlation_is_finite() {
        let device = NdArrayDevice::Cpu;

        let src_embed = Tensor::<TestBackend, 2>::random(
            [10, 2],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let dst_embed = Tensor::<TestBackend, 2>::random(
            [10, 2],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let src_orig = Tensor::<TestBackend, 2>::random(
            [10, 5],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let dst_orig = Tensor::<TestBackend, 2>::random(
            [10, 5],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let targets = Tensor::<TestBackend, 1>::from_floats([1.0; 10], &device);

        let loss = umap_loss_with_correlation(
            src_embed, dst_embed, src_orig, dst_orig, targets, 1.5f32, 0.9f32, 0.5f32,
        );
        let loss_val: f32 = loss.into_scalar().elem();

        assert!(loss_val.is_finite(), "Loss should be finite");
        assert!(loss_val >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_umap_edge_dataset_get() {
        let edges = vec![(0, 1, 1.0), (2, 3, 0.5), (4, 5, 1.0)];
        let dataset = UmapEdgeDataset { edges };

        assert_eq!(dataset.get(0), Some((0, 1, 1.0)));
        assert_eq!(dataset.get(1), Some((2, 3, 0.5)));
        assert_eq!(dataset.get(2), Some((4, 5, 1.0)));
        assert_eq!(dataset.get(3), None);
    }

    #[test]
    fn test_umap_edge_dataset_len() {
        let edges = vec![(0, 1, 1.0), (2, 3, 0.5)];
        let dataset = UmapEdgeDataset { edges };

        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_umap_edge_dataset_empty() {
        let dataset = UmapEdgeDataset { edges: vec![] };

        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.get(0), None);
    }
}

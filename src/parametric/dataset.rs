use burn::data::dataset::Dataset;
use burn::prelude::*;
use burn::tensor::Int;

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
/// * `a` - UMAP parameter (typically 0.1)
/// * `b` - UMAP parameter (typically 1.0)
///
/// ### Returns
///
/// Binary cross-entropy loss between predicted and target probabilities
pub fn umap_loss<B: Backend>(
    src_embed: Tensor<B, 2>,
    dst_embed: Tensor<B, 2>,
    targets: Tensor<B, 1>,
    a: f32,
    b: f32,
) -> Tensor<B, 1> {
    // compute squared distances
    let diff: Tensor<B, 2> = src_embed - dst_embed;
    // use ::<1> turbofish syntax <- important!
    let dist_sq: Tensor<B, 1> = diff.clone().powf_scalar(2.0).sum_dim(1).squeeze::<1>();

    // UMAP probability: 1 / (1 + a * dist^(2b))
    let a_tensor: Tensor<B, 1> = Tensor::from_floats([a], &diff.device());
    let power: Tensor<B, 1> = dist_sq.powf_scalar(b);
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
    let x_mean = x_dist.clone().mean();
    let z_mean = z_dist.clone().mean();

    let x_centered = x_dist - x_mean;
    let z_centered = z_dist - z_mean;

    let numerator = (x_centered.clone() * z_centered.clone()).mean();
    let x_std = (x_centered.powf_scalar(2.0).mean()).sqrt();
    let z_std = (z_centered.powf_scalar(2.0).mean()).sqrt();

    let correlation = numerator / (x_std * z_std);

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
/// * `a` - UMAP parameter (typically 0.1)
/// * `b` - UMAP parameter (typically 1.0)
/// * `correlation_weight` - Weight for correlation regularisation term
///
/// ### Returns
///
/// Combined loss: UMAP BCE loss + correlation_weight * correlation_loss
#[allow(clippy::too_many_arguments)]
pub fn umap_loss_with_correlation<B: Backend>(
    src_embed: Tensor<B, 2>,
    dst_embed: Tensor<B, 2>,
    src_orig: Tensor<B, 2>, // original high-dim data
    dst_orig: Tensor<B, 2>,
    targets: Tensor<B, 1>,
    a: f32,
    b: f32,
    correlation_weight: f32,
) -> Tensor<B, 1> {
    let umap = umap_loss(src_embed.clone(), dst_embed.clone(), targets, a, b);

    // Distances in embedding space
    let z_diff = src_embed - dst_embed;
    let z_dist = z_diff.powf_scalar(2.0).sum_dim(1).squeeze::<1>().sqrt(); // Use ::<1>

    // Distances in original space
    let x_diff = src_orig - dst_orig;
    let x_dist = x_diff.powf_scalar(2.0).sum_dim(1).squeeze::<1>().sqrt(); // Use ::<1>

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

use burn::{
    data::dataloader::{DataLoaderBuilder, Dataset},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{backend::AutodiffBackend, Element},
};
use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::data::structures::*;
use crate::parametric::batch::*;
use crate::parametric::dataset::*;
use crate::parametric::model::*;

////////////
// Params //
////////////

/// Parameters for parametric UMAP training
///
/// ### Fields
///
/// * `a` - Curve parameter for attractive force (typically ~1.5 for 2D)
/// * `b` - Curve parameter for repulsive force (typically ~0.9 for 2D)
/// * `lr` - Learning rate for the optimiser
/// * `corr_weight` - Coefficient in front of the negative Pearson correlation
///   coefficient that encourages similar distance in the embedding space as
///   in the original target space.
/// * `n_epochs` - Number of epochs to train the neural net for (typically 500).
/// * `batch_size` - Number of samples to train in a given batch (typically
///   256).
/// * `neg_sample_rate` - Number of negative samples per positive edge
///   (typically 5).
#[derive(Clone, Debug)]
pub struct TrainParametricParams<T> {
    pub a: T,
    pub b: T,
    pub lr: T,
    pub corr_weight: T,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub neg_sample_rate: usize,
}

impl<T> TrainParametricParams<T>
where
    T: Float + FromPrimitive + Element,
{
    /// Default parameters for 2D embedding
    ///
    /// ### Returns
    ///
    /// Sensible default parameters for 2D embeddings
    pub fn default_2d() -> Self {
        Self {
            a: T::from_f64(1.5).unwrap(),
            b: T::from_f64(0.9).unwrap(),
            lr: T::from_f64(0.001).unwrap(),
            n_epochs: 500,
            batch_size: 256,
            neg_sample_rate: 5,
            corr_weight: T::from_f64(0.0).unwrap(),
        }
    }

    /// Parameters from specified minimum distance, spread and correlation loss
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance parameter
    /// * `spread` - Effective scale of embedded points
    /// * `corr_weight` - Coefficient in front of the Pearson correlation loss
    ///   that encourages similar distance between embedding space and original
    ///   space.
    /// * `lr` - Optional learning rate. Default: `0.001`.
    /// * `n_epochs` - Optional number of epochs. Default: `500`.
    /// * `batch_size` - Optional batch size. Default: `256`.
    /// * `neg_sample_rate` - Optional negative sampling rate. Defaults to `5`.
    pub fn from_min_dist_spread(
        min_dist: T,
        spread: T,
        corr_weight: T,
        lr: Option<T>,
        n_epochs: Option<usize>,
        batch_size: Option<usize>,
        neg_sample_rate: Option<usize>,
    ) -> Self {
        // calculate alpha and beta
        let (a, b) = Self::fit_params(min_dist, spread, None);
        // return
        Self {
            a,
            b,
            corr_weight,
            lr: lr.unwrap_or(T::from_f64(0.001).unwrap()),
            n_epochs: n_epochs.unwrap_or(500),
            batch_size: batch_size.unwrap_or(256),
            neg_sample_rate: neg_sample_rate.unwrap_or(5),
        }
    }

    /// Fit curve parameters from min_dist and spread
    ///
    /// Fits the UMAP curve: `f(x) = 1 / (1 + a + x^(2b))` such that
    /// `f(min_dist) ca. 1.0` and `f(spread) ca. 0.0`.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance parameter
    /// * `spread` - Effective scale of embedded points
    /// * `lr` - Learning rate for gradient descent (default: 0.1)
    /// * `n_iter` - Number of optimisation iterations (default: 100)
    ///
    /// ### Returns
    ///
    /// Tuple of `(a, b)` according to the optimisation problem above.
    fn fit_params(min_dist: T, spread: T, n_iter: Option<usize>) -> (T, T) {
        let n_iter = n_iter.unwrap_or(300);
        let n_points = 300;

        // Generate x values from 0 to spread * 3
        let three = T::from_f64(3.0).unwrap();
        let max_x = spread * three;
        let step = max_x / T::from_usize(n_points - 1).unwrap();

        // Generate target y values
        let mut xv = Vec::with_capacity(n_points);
        let mut yv = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let x = step * T::from_usize(i).unwrap();
            let y = if x < min_dist {
                T::one()
            } else {
                (-(x - min_dist) / spread).exp()
            };
            xv.push(x);
            yv.push(y);
        }

        let mut a = T::one();
        let mut b = T::one();
        let two = T::from_f64(2.0).unwrap();

        for _ in 0..n_iter {
            let mut grad_a = T::zero();
            let mut grad_b = T::zero();
            let n_points_t = T::from_usize(n_points).unwrap();

            for i in 0..n_points {
                let x = xv[i];
                if x <= T::zero() {
                    continue;
                }

                let y_target = yv[i];
                let x_2b = x.powf(two * b);
                let denom = T::one() + a * x_2b;
                let pred = T::one() / denom;
                let err = pred - y_target;

                grad_a = grad_a + err * (-x_2b / (denom * denom));

                let log_x = x.ln();
                grad_b = grad_b + err * (-two * a * x_2b * log_x / (denom * denom));
            }

            // Normalise gradients and use adaptive learning rate
            grad_a = grad_a / n_points_t;
            grad_b = grad_b / n_points_t;

            let lr_a = T::from_f64(1.0).unwrap();
            let lr_b = T::from_f64(1.0).unwrap();

            a = a - lr_a * grad_a;
            b = b - lr_b * grad_b;

            a = a
                .max(T::from_f64(0.001).unwrap())
                .min(T::from_f64(10.0).unwrap());
            b = b
                .max(T::from_f64(0.1).unwrap())
                .min(T::from_f64(2.0).unwrap());
        }

        (a, b)
    }
}

impl<T> Default for TrainParametricParams<T>
where
    T: Float + FromPrimitive + Element,
{
    fn default() -> Self {
        TrainParametricParams::default_2d()
    }
}

/////////////
// Helpers //
/////////////

/// Data to tensor
///
/// ### Params
///
/// * `data` - The data to transform to the tensor
/// * `device` - The device on which to store the tensor
///
/// ### Returns
///
/// Tensor of the data
pub fn data_to_tensor<T, B>(data: MatRef<T>, device: &B::Device) -> Tensor<B, 2>
where
    T: Element + Float + FromPrimitive,
    B: AutodiffBackend,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let data_flat: Vec<T> = (0..n_samples)
        .flat_map(|i| (0..n_features).map(move |j| data[(i, j)]))
        .collect();

    Tensor::<B, 2>::from_data(TensorData::new(data_flat, [n_samples, n_features]), device)
}

/// Transform a graph into a UmapEdgeDataSet
///
/// ### Params
///
/// * `graph_data` - The underlying graph data
///
/// ### Returns
///
/// The UmapEdgeDataset
pub fn graph_to_trainings_data<T>(graph_data: &SparseGraph<T>) -> UmapEdgeDataset {
    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(graph_data.col_indices.len());
    for (idx, &col) in graph_data.col_indices.iter().enumerate() {
        edges.push((graph_data.row_indices[idx], col, 1.0f32));
    }
    UmapEdgeDataset { edges }
}

//////////////
// Training //
//////////////

/// Trainings loop for parametric UMAP
pub fn train_parametric_umap<B, T>(
    data: MatRef<T>,
    graph_data: SparseGraph<T>,
    model_config: &UmapMlpConfig,
    train_params: &TrainParametricParams<T>,
    device: &B::Device,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Element + Float + FromPrimitive + ToPrimitive,
    B: AutodiffBackend,
{
    let n_samples = data.nrows();

    let tensor_data: Tensor<B, 2> = data_to_tensor(data, device);
    let edge_dataset = graph_to_trainings_data(&graph_data);

    if verbose {
        println!(
            "Training parametric UMAP on {} positive edges...",
            edge_dataset.len()
        );
    }

    let batcher = UmapBatcher::new(train_params.neg_sample_rate, n_samples, seed as u64);
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(train_params.batch_size)
        .shuffle(seed as u64)
        .num_workers(4)
        .build(edge_dataset);

    let mut model: UmapMlp<B> = model_config.init::<B>(device);
    let mut optim = AdamConfig::new().init();

    let use_correlation = train_params.corr_weight > T::zero();

    for epoch in 0..train_params.n_epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for batch in dataloader.iter() {
            let src_feats = tensor_data.clone().select(0, batch.src_indices.clone());
            let dst_feats = tensor_data.clone().select(0, batch.dst_indices.clone());

            let src_embed = model.forward(src_feats.clone());
            let dst_embed = model.forward(dst_feats.clone());

            let loss = if use_correlation {
                umap_loss_with_correlation(
                    src_embed,
                    dst_embed,
                    src_feats,
                    dst_feats,
                    batch.targets,
                    train_params.a,
                    train_params.b,
                    train_params.corr_weight,
                )
            } else {
                umap_loss(
                    src_embed,
                    dst_embed,
                    batch.targets,
                    train_params.a,
                    train_params.b,
                )
            };

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(ToPrimitive::to_f64(&train_params.lr).unwrap(), model, grads);

            total_loss += loss.clone().into_scalar().elem::<f64>();
            n_batches += 1;
        }

        if verbose && (epoch % 10 == 0 || epoch + 1 == train_params.n_epochs) {
            println!(
                "Epoch {}/{}: Loss = {:.6}",
                epoch + 1,
                train_params.n_epochs,
                total_loss / n_batches as f64
            );
        }
    }

    // Get final embeddings
    let embeddings = model.forward(tensor_data);

    // Convert to Vec<Vec<f32>> format [n_components][n_samples]
    let n_components = model_config.output_size;
    let embedding_data: Vec<T> = embeddings.into_data().to_vec().unwrap();

    let mut result = vec![vec![T::zero(); n_samples]; n_components];
    for i in 0..n_samples {
        for j in 0..n_components {
            result[j][i] = embedding_data[i * n_components + j];
        }
    }

    result
}

///////////
// Tests //
///////////

#[cfg(test)]
mod parametric_train_tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::backend::Autodiff;
    use faer::Mat;

    type TestBackend = Autodiff<NdArray<f64>>;

    #[test]
    fn test_data_to_tensor_shape() {
        let data = Mat::from_fn(10, 5, |i, j| (i * 5 + j) as f64);
        let device = NdArrayDevice::Cpu;

        let tensor = data_to_tensor::<f64, TestBackend>(data.as_ref(), &device);

        assert_eq!(tensor.dims()[0], 10);
        assert_eq!(tensor.dims()[1], 5);
    }

    #[test]
    fn test_train_params_default() {
        let params = TrainParametricParams::<f64>::default_2d();

        assert!(params.a > 0.0);
        assert!(params.b > 0.0);
        assert!(params.lr > 0.0);
        assert!(params.n_epochs > 0);
        assert!(params.batch_size > 0);
        assert!(params.neg_sample_rate > 0);
    }

    #[test]
    fn test_train_params_from_min_dist_spread() {
        let params = TrainParametricParams::<f64>::from_min_dist_spread(
            0.1, 1.0, 0.0, None, None, None, None,
        );

        assert!(params.a > 0.0);
        assert!(params.b > 0.0);
        assert_eq!(params.corr_weight, 0.0);
    }

    #[test]
    fn test_graph_to_training_data() {
        use crate::data::structures::SparseGraph;

        let graph = SparseGraph {
            n_vertices: 10,
            row_indices: vec![0, 0, 1, 1, 2],
            col_indices: vec![1, 2, 0, 2, 1],
            values: vec![0.9, 0.8, 0.9, 0.7, 0.8],
        };

        let dataset = graph_to_trainings_data(&graph);

        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get(0), Some((0, 1, 1.0)));
        assert_eq!(dataset.get(1), Some((0, 2, 1.0)));
        assert_eq!(dataset.get(2), Some((1, 0, 1.0)));
    }
}

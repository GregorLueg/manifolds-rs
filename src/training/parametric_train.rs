use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::Dataset;
use burn::optim::GradientsParams;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Element;
use faer::MatRef;

use crate::data::structures::*;
use crate::parametric::batch::*;
use crate::parametric::dataset::*;
use crate::parametric::model::*;

////////////
// Params //
////////////

/// Parameters for parametric UMAP training
#[derive(Clone, Debug)]
pub struct TrainParametricParams {
    pub a: f32,
    pub b: f32,
    pub lr: f64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub neg_sample_rate: usize,
    pub corr_weight: f32,
}

impl TrainParametricParams {
    /// Default parameters for 2D embedding
    pub fn default_2d() -> Self {
        Self {
            a: 1.5,
            b: 0.9,
            lr: 0.001,
            n_epochs: 500,
            batch_size: 256,
            neg_sample_rate: 5,
            corr_weight: 0.0,
        }
    }

    /// Parameters from specified minimum distance and spread
    pub fn from_min_dist_spread(
        min_dist: f32,
        spread: f32,
        lr: f64,
        n_epochs: usize,
        batch_size: usize,
        neg_sample_rate: usize,
        corr_weight: f32,
    ) -> Self {
        let (a, b) = Self::fit_params(min_dist, spread, None);
        Self {
            a,
            b,
            lr,
            n_epochs,
            batch_size,
            neg_sample_rate,
            corr_weight,
        }
    }

    /// Fit curve parameters from min_dist and spread
    fn fit_params(min_dist: f32, spread: f32, n_iter: Option<usize>) -> (f32, f32) {
        let n_iter = n_iter.unwrap_or(300);
        let n_points = 300;

        let max_x = spread * 3.0;
        let step = max_x / (n_points - 1) as f32;

        let mut xv = Vec::with_capacity(n_points);
        let mut yv = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let x = step * i as f32;
            let y = if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            };
            xv.push(x);
            yv.push(y);
        }

        let mut a = 1.0f32;
        let mut b = 1.0f32;

        for _ in 0..n_iter {
            let mut grad_a = 0.0f32;
            let mut grad_b = 0.0f32;

            for i in 0..n_points {
                let x = xv[i];
                if x <= 0.0 {
                    continue;
                }

                let y_target = yv[i];
                let x_2b = x.powf(2.0 * b);
                let denom = 1.0 + a * x_2b;
                let pred = 1.0 / denom;
                let err = pred - y_target;

                grad_a += err * (-x_2b / (denom * denom));

                let log_x = x.ln();
                grad_b += err * (-2.0 * a * x_2b * log_x / (denom * denom));
            }

            grad_a /= n_points as f32;
            grad_b /= n_points as f32;

            a -= grad_a;
            b -= grad_b;

            a = a.clamp(0.001, 10.0);
            b = b.clamp(0.1, 2.0);
        }

        (a, b)
    }
}

impl Default for TrainParametricParams {
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
    T: Element,
    B: AutodiffBackend,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let data_flat: Vec<T> = (0..n_samples)
        .flat_map(|i| (0..n_features).map(move |j| data[(i, j)]))
        .collect();

    Tensor::<B, 2>::from_floats(&data_flat[..], device).reshape([n_samples, n_features])
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
    train_params: &TrainParametricParams,
    device: &B::Device,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<f32>>
where
    T: Element,
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

    let use_correlation = train_params.corr_weight > 0.0;

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
            model = optim.step(train_params.lr, model, grads);

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
    let embedding_data: Vec<f32> = embeddings.into_data().to_vec().unwrap();

    let mut result = vec![vec![0.0f32; n_samples]; n_components];
    for i in 0..n_samples {
        for j in 0..n_components {
            result[j][i] = embedding_data[i * n_components + j];
        }
    }

    result
}

use burn::{data::dataloader::batcher::Batcher, prelude::*, tensor::TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::{Arc, Mutex};

use crate::parametric::dataset::*;

/// Batcher that converts UMAP edges into training batches with negative sampling
///
/// ### Fields
///
/// * `neg_sample_rate` - Number of negative samples per positive edge
/// * `n_samples` - Total number of samples in the dataset (for negative
///   sampling)
/// * `rng` - Thread safe seeded RNG for negative sampling
#[derive(Clone)]
pub struct UmapBatcher {
    pub neg_sample_rate: usize,
    pub n_samples: usize,
    rng: Arc<Mutex<StdRng>>,
}

impl UmapBatcher {
    /// Generate a new UmapBatcher
    ///
    /// ### Params
    ///
    /// * `neg_sample_rate` - Number of negative samples to generate
    /// * `n_samples` - Total number of samples in the data set
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Self
    pub fn new(neg_sample_rate: usize, n_samples: usize, seed: u64) -> Self {
        Self {
            neg_sample_rate,
            n_samples,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }
}

impl<B: Backend> Batcher<B, (usize, usize, f32), UmapBatch<B>> for UmapBatcher {
    /// Generate a batch
    ///
    /// ### Params
    ///
    /// * `items` -
    /// * `device` -
    ///
    /// ### Returns
    ///
    ///
    fn batch(&self, items: Vec<(usize, usize, f32)>, device: &B::Device) -> UmapBatch<B> {
        let n_pos = items.len();
        let capacity = n_pos * (1 + self.neg_sample_rate);

        let mut src_indices = Vec::with_capacity(capacity);
        let mut dst_indices = Vec::with_capacity(capacity);
        let mut targets = Vec::with_capacity(capacity);

        // Add positive edges
        for (src, dst, target) in items.iter() {
            src_indices.push(*src as i64);
            dst_indices.push(*dst as i64);
            targets.push(*target);
        }

        // Add negative samples
        let mut rng = self.rng.lock().unwrap();
        for (src, _, _) in items.iter() {
            for _ in 0..self.neg_sample_rate {
                let neg_dst = rng.random_range(0..self.n_samples);
                if neg_dst != *src {
                    src_indices.push(*src as i64);
                    dst_indices.push(neg_dst as i64);
                    targets.push(0.0);
                }
            }
        }

        let n = src_indices.len();

        UmapBatch {
            src_indices: Tensor::from_data(
                TensorData::new(src_indices, [n]).convert::<B::IntElem>(),
                device,
            ),
            dst_indices: Tensor::from_data(
                TensorData::new(dst_indices, [n]).convert::<B::IntElem>(),
                device,
            ),
            targets: Tensor::from_floats(targets.as_slice(), device),
        }
    }
}

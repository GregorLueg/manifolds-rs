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
                // Keep trying until we get a valid negative sample
                let neg_dst = loop {
                    let candidate = rng.random_range(0..self.n_samples);
                    if candidate != *src {
                        break candidate;
                    }
                };

                src_indices.push(*src as i64);
                dst_indices.push(neg_dst as i64);
                targets.push(0.0);
            }
        }

        let n = src_indices.len();
        assert_eq!(
            n, capacity,
            "Batch size mismatch: expected {}, got {}",
            capacity, n
        );

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

///////////
// Tests //
///////////

#[cfg(test)]
mod batch_tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_batcher_creates_correct_batch_size() {
        let batcher = UmapBatcher::new(5, 100, 42);
        let items = vec![(0, 1, 1.0), (2, 3, 1.0), (4, 5, 1.0)];
        let device = NdArrayDevice::Cpu;

        let batch: UmapBatch<TestBackend> = batcher.batch(items.clone(), &device);

        // 3 positive edges + 3 * 5 negative samples = 18 total
        let expected_size = items.len() * (1 + 5);
        assert_eq!(
            batch.src_indices.dims()[0],
            expected_size,
            "src_indices should have {} elements",
            expected_size
        );
        assert_eq!(
            batch.dst_indices.dims()[0],
            expected_size,
            "dst_indices should have {} elements",
            expected_size
        );
        assert_eq!(
            batch.targets.dims()[0],
            expected_size,
            "targets should have {} elements",
            expected_size
        );
    }

    #[test]
    fn test_batcher_handles_self_collision() {
        // Test case where negative sampling might pick the source node
        let batcher = UmapBatcher::new(20, 10, 42);
        let items = vec![(5, 1, 1.0)];
        let device = NdArrayDevice::Cpu;

        let batch: UmapBatch<TestBackend> = batcher.batch(items.clone(), &device);

        // Should still produce exactly 1 + 20 samples
        let expected_size = 21;
        assert_eq!(
            batch.src_indices.dims()[0],
            expected_size,
            "Batch size should be consistent even with self-collisions"
        );
    }

    #[test]
    fn test_batcher_positive_edges_are_first() {
        let batcher = UmapBatcher::new(3, 100, 42);
        let items = vec![(0, 1, 1.0), (2, 3, 0.5)];
        let device = NdArrayDevice::Cpu;

        let batch: UmapBatch<TestBackend> = batcher.batch(items.clone(), &device);

        let targets_data = batch.targets.to_data();
        let targets_vec: Vec<f32> = targets_data.to_vec().unwrap();

        // First two should be the positive edges
        assert_eq!(targets_vec[0], 1.0);
        assert_eq!(targets_vec[1], 0.5);

        // Rest should be negative (0.0)
        for i in 2..targets_vec.len() {
            assert_eq!(
                targets_vec[i], 0.0,
                "Negative samples should have target 0.0"
            );
        }
    }

    #[test]
    fn test_batcher_src_indices_match() {
        let batcher = UmapBatcher::new(5, 100, 42);
        let items = vec![(10, 20, 1.0), (30, 40, 1.0)];
        let device = NdArrayDevice::Cpu;

        let batch: UmapBatch<TestBackend> = batcher.batch(items.clone(), &device);

        let src_data = batch.src_indices.to_data();
        let src_vec: Vec<i64> = src_data.to_vec().unwrap();

        // First positive edge
        assert_eq!(src_vec[0], 10);
        // Second positive edge
        assert_eq!(src_vec[1], 30);

        // Next 5 should be negative samples from node 10
        for i in 2..7 {
            assert_eq!(src_vec[i], 10, "Negative samples should match source");
        }

        // Next 5 should be negative samples from node 30
        for i in 7..12 {
            assert_eq!(src_vec[i], 30, "Negative samples should match source");
        }
    }

    #[test]
    fn test_batcher_reproducible_with_seed() {
        let batcher1 = UmapBatcher::new(5, 100, 42);
        let batcher2 = UmapBatcher::new(5, 100, 42);
        let items = vec![(0, 1, 1.0), (2, 3, 1.0)];
        let device = NdArrayDevice::Cpu;

        let batch1: UmapBatch<TestBackend> = batcher1.batch(items.clone(), &device);
        let batch2: UmapBatch<TestBackend> = batcher2.batch(items.clone(), &device);

        let dst1: Vec<i64> = batch1.dst_indices.to_data().to_vec().unwrap();
        let dst2: Vec<i64> = batch2.dst_indices.to_data().to_vec().unwrap();

        assert_eq!(
            dst1, dst2,
            "Same seed should produce identical negative samples"
        );
    }

    #[test]
    fn test_batcher_no_self_loops() {
        let batcher = UmapBatcher::new(10, 50, 42);
        let items = vec![(5, 10, 1.0), (15, 20, 1.0)];
        let device = NdArrayDevice::Cpu;

        let batch: UmapBatch<TestBackend> = batcher.batch(items.clone(), &device);

        let src_data: Vec<i64> = batch.src_indices.to_data().to_vec().unwrap();
        let dst_data: Vec<i64> = batch.dst_indices.to_data().to_vec().unwrap();

        // Check negative samples don't create self-loops
        for i in 2..src_data.len() {
            if dst_data[i] == src_data[i] {
                panic!(
                    "Self-loop detected at index {}: src={}, dst={}",
                    i, src_data[i], dst_data[i]
                );
            }
        }
    }
}

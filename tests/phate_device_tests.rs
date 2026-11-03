//! PHATE with spectral landmark selection on wgpu and MLX, against the CPU path.

#![cfg(any(feature = "gpu", mlx_available))]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use faer::Mat;
use manifolds_rs::data::structures::{CompressedSparseData, CompressedSparseFormat};
use manifolds_rs::prelude::*;
use manifolds_rs::utils::math::sparse_randomised_svd;
use manifolds_rs::*;

/// Symmetric test matrix with a decaying spectrum: a ring graph with a few long
/// edges, sparse enough to exercise short and long CSR rows.
fn ring_matrix(n: usize) -> CompressedSparseData<f32> {
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let mut row: Vec<(usize, f32)> = vec![(i, 1.0)];
        for d in 1..=6 {
            let w = 0.8f32.powi(d as i32);
            row.push(((i + d) % n, w));
            row.push(((i + n - d) % n, w));
        }
        row.sort_by_key(|x| x.0);
        for (j, v) in row {
            indices.push(j);
            data.push(v);
        }
        indptr.push(indices.len());
    }
    CompressedSparseData {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        shape: (n, n),
    }
}

/// Mean cosine of the principal angles between two column spaces.
fn mean_subspace_cos(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let g = a.transpose() * b;
    let sv = g.thin_svd().unwrap();
    let c = sv.S().column_vector();
    c.iter().sum::<f64>() / c.nrows() as f64
}

/// The same sketch in `f32` must land on the CPU's subspace, closer than the CPU
/// lands on itself from another seed. The second assert is what lets the first
/// one fail on a device path that drew a different sketch.
fn assert_matches_cpu(run: impl Fn(usize) -> Mat<f64>) {
    let m = ring_matrix(3000);
    let cpu = sparse_randomised_svd(&m, 20, 42, Some(10), Some(7))
        .unwrap()
        .v;
    let device = run(7);
    assert_eq!(device.shape(), (3000, 20));

    let same = mean_subspace_cos(&cpu, &device);
    assert!(same > 0.999, "mean cos against the CPU {same}");

    let other_seed = sparse_randomised_svd(&m, 20, 7, Some(10), Some(7))
        .unwrap()
        .v;
    let noise = mean_subspace_cos(&cpu, &other_seed);
    assert!(
        noise < same,
        "other seed {noise} not below same seed {same}"
    );
}

fn device_params() -> PhateParams<f32> {
    PhateParams {
        k: 5,
        diffusion_params: PhateDiffusionParams::new(
            Some(40.0),
            1.0,
            1e-4,
            "add".to_string(),
            Some(100),
            "spectral".to_string(),
            Some(20),
            None,
            None,
            1.0,
        ),
        ..PhateParams::default()
    }
}

fn data_f32() -> (Vec<f32>, usize, usize) {
    let (mat, _) = create_diagnostic_data(400, 16, 42);
    let m = mat_to_f32(mat);
    let (n, d) = m.shape();
    let mut v = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            v[i * d + j] = m[(i, j)];
        }
    }
    (v, n, d)
}

#[cfg(feature = "gpu")]
mod wgpu_path {
    use super::*;
    use cubecl::wgpu::WgpuDevice;
    use manifolds_rs::utils::sparse_svd_gpu::symmetric_svd_vectors_gpu;

    #[test]
    fn test_phate_svd_wgpu_matches_cpu() {
        let client = cubecl::Device::from(WgpuDevice::default()).client();
        let m = ring_matrix(3000);
        assert_matches_cpu(|p| symmetric_svd_vectors_gpu(&m, 20, 42, 10, p, &client).unwrap());
    }

    #[test]
    fn test_phate_gpu_runs_end_to_end() {
        let (v, n, d) = data_f32();
        let nn = NearestNeighbourParamsGpu::<f32>::default();
        let e = phate_gpu(
            (v.as_slice(), n, d),
            None,
            device_params(),
            &nn,
            WgpuDevice::default().into(),
            42,
            0,
        )
        .unwrap();
        assert_eq!((e.len(), e[0].len()), (2, n));
        assert!(e.iter().flatten().all(|x| x.is_finite()));
    }
}

#[cfg(mlx_available)]
mod mlx_path {
    use super::*;
    use manifolds_rs::mlx::sparse_svd_mlx::symmetric_svd_vectors_mlx;

    #[test]
    fn test_phate_svd_mlx_matches_cpu() {
        let m = ring_matrix(3000);
        assert_matches_cpu(|p| symmetric_svd_vectors_mlx(&m, 20, 42, 10, p).unwrap());
    }

    #[test]
    fn test_phate_mlx_runs_end_to_end() {
        let (v, n, d) = data_f32();
        let params = PhateParams {
            ann_type: "nndescent_mlx".to_string(),
            ..device_params()
        };
        let e = phate_mlx((v.as_slice(), n, d), None, params, 42, 0).unwrap();
        assert_eq!((e.len(), e[0].len()), (2, n));
        assert!(e.iter().flatten().all(|x| x.is_finite()));
    }
}

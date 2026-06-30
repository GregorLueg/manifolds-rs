//! GPU-accelerated UMAP optimiser. Mirrors `optimise_embedding_adam_parallel`
//! but offloads gradient accumulation, Adam updates and edge-schedule
//! advancement to the GPU.

use ann_search_rs::gpu::tensor::GpuTensor;
use cubecl::prelude::*;

use crate::prelude::*;
use crate::training::umap_optimisers::UmapOptimParams;

///////////////////
// Host-side CSR //
///////////////////

/// Host-side CSR representation of the symmetrised UMAP graph, ready to be
/// uploaded to the GPU. Each undirected edge `(i, j, w)` with `i < j` appears
/// twice in the CSR arrays: once under node `i` (with `is_smaller = 1`) and
/// once under node `j` (with `is_smaller = 0`). Only the smaller-indexed side
/// of an edge schedules negative samples, matching the CPU code.
pub(crate) struct UmapCsrGraph<T> {
    /// Number of nodes
    pub n: usize,
    /// Number of unique undirected edges
    pub n_edges: usize,
    /// CSR row pointers `[n + 1]`
    pub node_edge_offsets: Vec<u32>,
    /// Edge index per `(node, edge)` entry `[2 * n_edges]`
    pub csr_edge_idx: Vec<u32>,
    /// Other endpoint per `(node, edge)` entry `[2 * n_edges]`
    pub csr_other_node: Vec<u32>,
    /// Sampling period per edge `[n_edges]`
    pub epochs_per_sample: Vec<T>,
}

impl<T> UmapCsrGraph<T>
where
    T: ManifoldsFloat,
{
    /// Build CSR layout from an adjacency-list graph.
    ///
    /// The `i < j` filter deduplicates entries when the input graph is
    /// symmetric, and silently drops self-loops.
    pub fn from_graph(graph: &[Vec<(usize, T)>]) -> Result<Self, ManifoldsError> {
        let n = graph.len();
        if n == 0 {
            return Err(ManifoldsError::NoData);
        }

        let mut edges: Vec<(usize, usize, T)> = Vec::new();
        let mut degree = vec![0u32; n];

        for (i, neighbours) in graph.iter().enumerate() {
            for &(j, w) in neighbours {
                if i < j {
                    edges.push((i, j, w));
                    degree[i] += 1;
                    degree[j] += 1;
                }
            }
        }

        if edges.is_empty() {
            return Err(ManifoldsError::NoGraphEdges);
        }

        let n_edges = edges.len();

        let mut node_edge_offsets = vec![0u32; n + 1];
        for i in 0..n {
            node_edge_offsets[i + 1] = node_edge_offsets[i] + degree[i];
        }

        let mut csr_edge_idx = vec![0u32; 2 * n_edges];
        let mut csr_other_node = vec![0u32; 2 * n_edges];
        let mut csr_is_smaller = vec![0u32; 2 * n_edges];

        let mut cursor = node_edge_offsets.clone();
        for (edge_idx, &(i, j, _)) in edges.iter().enumerate() {
            let pos_i = cursor[i] as usize;
            csr_edge_idx[pos_i] = edge_idx as u32;
            csr_other_node[pos_i] = j as u32;
            csr_is_smaller[pos_i] = 1;
            cursor[i] += 1;

            let pos_j = cursor[j] as usize;
            csr_edge_idx[pos_j] = edge_idx as u32;
            csr_other_node[pos_j] = i as u32;
            csr_is_smaller[pos_j] = 0;
            cursor[j] += 1;
        }

        let zero = T::zero();
        let one = T::one();
        let large_epoch = T::from(1e8).unwrap();
        let max_weight = edges
            .iter()
            .map(|(_, _, w)| *w)
            .fold(zero, |acc, w| if w > acc { w } else { acc });

        let epochs_per_sample: Vec<T> = edges
            .iter()
            .map(|(_, _, w)| {
                let norm = *w / max_weight;
                if norm > zero {
                    one / norm
                } else {
                    large_epoch
                }
            })
            .collect();

        Ok(Self {
            n,
            n_edges,
            node_edge_offsets,
            csr_edge_idx,
            csr_other_node,
            epochs_per_sample,
        })
    }
}

//////////////////
// Device state //
//////////////////

/// All GPU-resident state for the UMAP optimiser. Built once via `upload`,
/// mutated in place by the per-epoch kernels.
pub(crate) struct UmapGpuState<R, T>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    // Mutable, updated every epoch
    pub embd: GpuTensor<R, T>,      // [n, n_dim]
    pub m: GpuTensor<R, T>,         // [n, n_dim]
    pub v: GpuTensor<R, T>,         // [n, n_dim]
    pub node_grad: GpuTensor<R, T>, // [n, n_dim] scratch

    // Immutable CSR, uploaded once
    pub node_edge_offsets: GpuTensor<R, u32>, // [n + 1]
    pub csr_edge_idx: GpuTensor<R, u32>,      // [2 * n_edges]
    pub csr_other_node: GpuTensor<R, u32>,    // [2 * n_edges]

    // Edge schedule
    pub epochs_per_sample: GpuTensor<R, T>,
    pub epoch_of_next_sample: GpuTensor<R, T>,

    // Pre-baked per-epoch schedules
    pub lr_schedule: GpuTensor<R, T>,
    pub ad_scale: GpuTensor<R, T>,
    pub epsc: GpuTensor<R, T>,

    pub n: usize,
    pub n_dim: usize,
    pub n_edges: usize,
    pub n_epochs: usize,
}

impl<R, T> UmapGpuState<R, T>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    /// Build device state from CPU inputs. All buffers are uploaded; no
    /// kernels run.
    pub fn upload(
        embd: &[Vec<T>],
        csr: &UmapCsrGraph<T>,
        params: &UmapOptimParams<T>,
        client: &ComputeClient<R>,
    ) -> Result<Self, ManifoldsError> {
        let n = embd.len();
        if n == 0 {
            return Err(ManifoldsError::NoData);
        }
        let n_dim = embd[0].len();
        let n_edges = csr.n_edges;
        let n_epochs = params.n_epochs;

        let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
        for point in embd {
            embd_flat.extend_from_slice(point);
        }

        // Per-epoch schedules. Formulas mirror optimise_embedding_adam_parallel
        // exactly so a downstream Adam update kernel can read these element-
        // by-element.
        let one = T::one();
        let n_epochs_f = T::from(n_epochs).unwrap();
        let lr_schedule: Vec<T> = (0..n_epochs)
            .map(|e| params.lr * (one - T::from(e).unwrap() / n_epochs_f))
            .collect();

        let (ad_scale, epsc): (Vec<T>, Vec<T>) = (0..n_epochs)
            .map(|epoch| {
                let t = T::from(epoch + 1).unwrap();
                let beta1t = num_traits::Float::powf(params.beta1, t);
                let beta2t = num_traits::Float::powf(params.beta2, t);
                let sqrt_b2t1 = num_traits::Float::sqrt(one - beta2t);
                (sqrt_b2t1 / (one - beta1t), sqrt_b2t1 * params.eps)
            })
            .unzip();

        let zeros = vec![T::zero(); n * n_dim];

        Ok(Self {
            embd: GpuTensor::from_slice(&embd_flat, vec![n, n_dim], client),
            m: GpuTensor::from_slice(&zeros, vec![n, n_dim], client),
            v: GpuTensor::from_slice(&zeros, vec![n, n_dim], client),
            node_grad: GpuTensor::from_slice(&zeros, vec![n, n_dim], client),

            node_edge_offsets: GpuTensor::from_slice(&csr.node_edge_offsets, vec![n + 1], client),
            csr_edge_idx: GpuTensor::from_slice(&csr.csr_edge_idx, vec![2 * n_edges], client),
            csr_other_node: GpuTensor::from_slice(&csr.csr_other_node, vec![2 * n_edges], client),

            epochs_per_sample: GpuTensor::from_slice(&csr.epochs_per_sample, vec![n_edges], client),

            // Cursors initialise to the period itself (first sample event)
            epoch_of_next_sample: GpuTensor::from_slice(
                &csr.epochs_per_sample,
                vec![n_edges],
                client,
            ),

            lr_schedule: GpuTensor::from_slice(&lr_schedule, vec![n_epochs], client),
            ad_scale: GpuTensor::from_slice(&ad_scale, vec![n_epochs], client),
            epsc: GpuTensor::from_slice(&epsc, vec![n_epochs], client),

            n,
            n_dim,
            n_edges,
            n_epochs,
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    // Symmetric triangle, weights (0,1)=1.0, (0,2)=0.5, (1,2)=1.0
    fn triangle_graph() -> Vec<Vec<(usize, f64)>> {
        vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ]
    }

    fn triangle_setup() -> (Vec<Vec<f32>>, Vec<Vec<(usize, f32)>>, UmapOptimParams<f32>) {
        let embd = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let graph = vec![
            vec![(1_usize, 1.0_f32), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];
        let params = UmapOptimParams {
            a: 1.5,
            b: 0.9,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 5,
            min_dist: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-7,
        };
        (embd, graph, params)
    }

    fn build_state(
        device: &WgpuDevice,
    ) -> (
        ComputeClient<WgpuRuntime>,
        UmapCsrGraph<f32>,
        UmapOptimParams<f32>,
        UmapGpuState<WgpuRuntime, f32>,
    ) {
        let client = WgpuRuntime::client(device);
        let (embd, graph, params) = triangle_setup();
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let state =
            UmapGpuState::<WgpuRuntime, f32>::upload(&embd, &csr, &params, &client).unwrap();
        (client, csr, params, state)
    }

    #[test]
    fn test_csr_triangle_layout() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();

        assert_eq!(csr.n, 3);
        assert_eq!(csr.n_edges, 3);
        assert_eq!(csr.node_edge_offsets, vec![0, 2, 4, 6]);

        // Edge ordering: (0,1)=edge0, (0,2)=edge1, (1,2)=edge2
        assert_eq!(csr.csr_edge_idx, vec![0, 1, 0, 2, 1, 2]);
        assert_eq!(csr.csr_other_node, vec![1, 2, 0, 2, 0, 1]);
    }

    #[test]
    fn test_csr_triangle_sampling_periods() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();

        // max_weight = 1.0; epochs_per_sample = max_weight / w
        assert_eq!(csr.epochs_per_sample, vec![1.0, 2.0, 1.0]);
    }

    // Invariant: degree accumulation must equal 2 * n_edges (each edge
    // contributes to two endpoints).
    #[test]
    fn test_csr_offsets_sum_to_2x_edges() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();
        assert_eq!(
            *csr.node_edge_offsets.last().unwrap() as usize,
            2 * csr.n_edges
        );
    }

    #[test]
    fn test_csr_zero_nodes_error() {
        let g: Vec<Vec<(usize, f64)>> = vec![];
        assert!(matches!(
            UmapCsrGraph::<f64>::from_graph(&g),
            Err(ManifoldsError::NoData)
        ));
    }

    #[test]
    fn test_csr_no_edges_error() {
        let g: Vec<Vec<(usize, f64)>> = vec![vec![], vec![], vec![]];
        assert!(matches!(
            UmapCsrGraph::<f64>::from_graph(&g),
            Err(ManifoldsError::NoGraphEdges)
        ));
    }

    // Self-loops `(i, i, w)` must be dropped by the i < j filter.
    #[test]
    fn test_csr_drops_self_loops() {
        let g: Vec<Vec<(usize, f64)>> = vec![vec![(0, 1.0), (1, 1.0)], vec![(1, 1.0), (0, 1.0)]];
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();
        assert_eq!(csr.n_edges, 1);
        assert_eq!(csr.csr_other_node, vec![1, 0]);
    }

    // Asymmetric input: only graph[0] specifies edges; still gets a valid CSR.
    #[test]
    fn test_csr_asymmetric_input() {
        let g: Vec<Vec<(usize, f64)>> = vec![vec![(1, 1.0), (2, 0.5)], vec![], vec![]];
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();
        assert_eq!(csr.n_edges, 2);
        assert_eq!(csr.node_edge_offsets, vec![0, 2, 3, 4]);
    }

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    #[test]
    fn test_embedding_roundtrip() {
        let Some(device) = try_device() else { return };
        let (client, _, _, state) = build_state(&device);
        let got = state.embd.clone().read(&client).unwrap();
        let want = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert_eq!(got, want);
    }

    #[test]
    fn test_m_v_node_grad_zeroed() {
        let Some(device) = try_device() else { return };
        let (client, _, _, state) = build_state(&device);
        let zeros = vec![0.0_f32; state.n * state.n_dim];
        assert_eq!(state.m.clone().read(&client).unwrap(), zeros);
        assert_eq!(state.v.clone().read(&client).unwrap(), zeros);
        assert_eq!(state.node_grad.clone().read(&client).unwrap(), zeros);
    }

    #[test]
    fn test_csr_arrays_roundtrip() {
        let Some(device) = try_device() else { return };
        let (client, csr, _, state) = build_state(&device);
        assert_eq!(
            state.node_edge_offsets.clone().read(&client).unwrap(),
            csr.node_edge_offsets
        );
        assert_eq!(
            state.csr_edge_idx.clone().read(&client).unwrap(),
            csr.csr_edge_idx
        );
        assert_eq!(
            state.csr_other_node.clone().read(&client).unwrap(),
            csr.csr_other_node
        );
    }

    #[test]
    fn test_edge_schedule_initial_state() {
        let Some(device) = try_device() else { return };
        let (client, csr, _, state) = build_state(&device);
        assert_eq!(
            state.epochs_per_sample.clone().read(&client).unwrap(),
            csr.epochs_per_sample
        );
        // Cursors initialised to one period from time 0
        assert_eq!(
            state.epoch_of_next_sample.clone().read(&client).unwrap(),
            csr.epochs_per_sample
        );
    }

    #[test]
    fn test_per_epoch_schedules() {
        let Some(device) = try_device() else { return };
        let (client, _, params, state) = build_state(&device);

        let lr_got = state.lr_schedule.clone().read(&client).unwrap();
        let ad_got = state.ad_scale.clone().read(&client).unwrap();
        let epsc_got = state.epsc.clone().read(&client).unwrap();

        let n_epochs_f = params.n_epochs as f32;
        for e in 0..params.n_epochs {
            let lr_want = params.lr * (1.0 - e as f32 / n_epochs_f);
            assert!((lr_got[e] - lr_want).abs() < 1e-6, "lr[{}]", e);

            let t = (e + 1) as f32;
            let sqrt_b2t1 = (1.0 - params.beta2.powf(t)).sqrt();
            let ad_want = sqrt_b2t1 / (1.0 - params.beta1.powf(t));
            let epsc_want = sqrt_b2t1 * params.eps;

            assert!((ad_got[e] - ad_want).abs() < 1e-6, "ad_scale[{}]", e);
            assert!((epsc_got[e] - epsc_want).abs() < 1e-9, "epsc[{}]", e);
        }
    }

    // n_dim != 2 must still work; tests the dim handling end-to-end.
    #[test]
    fn test_upload_3d_embedding() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let embd = vec![
            vec![0.0_f32, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
        ];
        let (_, graph, mut params) = triangle_setup();
        params.n_epochs = 5;
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let state =
            UmapGpuState::<WgpuRuntime, f32>::upload(&embd, &csr, &params, &client).unwrap();

        assert_eq!(state.n_dim, 3);
        let got = state.embd.clone().read(&client).unwrap();
        assert_eq!(got, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}

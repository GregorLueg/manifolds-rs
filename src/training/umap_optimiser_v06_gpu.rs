//! GPU port of the umap-learn 0.6 Adam ([`crate::training::umap_optimiser_v06`]).
//!
//! Two kernels per epoch: a per-node gradient walk over the node's CSR row,
//! which also advances that row's sampling clocks, and a per-coordinate Adam
//! step. The negative order (a shuffle, or the hard-negative projection order
//! on large layouts) is built on the host with the CPU module's code and
//! uploaded every epoch; the projection order needs the embedding read back
//! first. The force-ratio adaptation runs on the host off that same readback.
//! `f32` only.

#![allow(missing_docs)] // cubecl weirdness

use cubecl::prelude::*;
use cubecl::Device;
use cubecl_utils_rs::prelude::*;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use crate::prelude::*;
use crate::training::umap_csr::UmapCsrGraph;
use crate::training::umap_optimiser_gpu::{sync_client, umap_workgroup_size};
use crate::training::umap_optimiser_v06::*;
use crate::training::umap_optimisers::{SplitPowLut, UmapOptimParams};

///////////////
// Constants //
///////////////

/// Bits per limb in [`v06_mulmod`]. Both factors sit below the modulus, so
/// with a modulus under `2^(2 * V06_MULMOD_LIMB)` every partial product fits
/// in a `u32`.
const V06_MULMOD_LIMB: u32 = 10;

// The hash modulus is the window or the node count, never above the default
// window on the GPU path, so the limb split above cannot overflow.
const _: () = assert!(V06_DEFAULT_NEG_SELECTION_RANGE < 1 << (2 * V06_MULMOD_LIMB));

//////////////////
// Device state //
//////////////////

/// Device-resident state of one 0.6 Adam run.
struct V06GpuState {
    /// Embedding `[n * n_dim]`
    embd: GpuTensor<f32>,
    /// First Adam moment `[n * n_dim]`
    m: GpuTensor<f32>,
    /// Second Adam moment `[n * n_dim]`
    v: GpuTensor<f32>,
    /// Gradient scratch `[n * n_dim]`, overwritten every epoch
    grad: GpuTensor<f32>,
    /// CSR row pointers `[n + 1]`
    offsets: GpuTensor<u32>,
    /// Edge index per CSR entry `[2 * n_edges]`
    edge_idx: GpuTensor<u32>,
    /// Other endpoint per CSR entry `[2 * n_edges]`
    other: GpuTensor<u32>,
    /// Sampling period per edge `[n_edges]`
    epochs_per_sample: GpuTensor<f32>,
    /// Negative-sampling period per edge `[n_edges]`
    epochs_per_neg: GpuTensor<f32>,
    /// Next sample epoch per CSR entry `[2 * n_edges]`
    next_sample: GpuTensor<f32>,
    /// Negatives drawn up to this epoch, per CSR entry `[2 * n_edges]`
    next_neg: GpuTensor<f32>,
    /// Workgroup size of every launch
    wg: u32,
    /// One unit per node
    cubes_node: CubeCount,
    /// One unit per embedding coordinate
    cubes_elem: CubeCount,
}

impl V06GpuState {
    /// Upload the embedding and graph.
    ///
    /// ### Params
    ///
    /// * `embd` - Flat row-major embedding `[n * n_dim]`
    /// * `n_dim` - Embedding dimension
    /// * `csr` - Graph in CSR form
    /// * `neg_period_scale` - `V06_NEG_PERIOD_SCALE / neg_sample_rate`
    /// * `client` - CubeCL client
    ///
    /// ### Returns
    ///
    /// The state, or the allocation or grid error.
    fn upload(
        embd: &[f32],
        n_dim: usize,
        csr: &UmapCsrGraph<f32>,
        neg_period_scale: f32,
        client: &Client,
    ) -> Result<Self, ManifoldsError> {
        let n = csr.n;
        let n_entries = csr.csr_edge_idx.len();
        let limits = GpuLimits::from_client(client);
        let wg = umap_workgroup_size(&limits);
        let plan = |kernel: &'static str, total: usize| -> Result<CubeCount, ManifoldsError> {
            let (gx, gy) = grid_2d((total as u32).div_ceil(wg), &limits)?;
            Ok(checked_cube_count(kernel, gx, gy, 1, &limits)?)
        };

        let epochs_per_neg: Vec<f32> = csr
            .epochs_per_sample
            .iter()
            .map(|&e| e * neg_period_scale)
            .collect();
        let next_sample: Vec<f32> = csr
            .csr_edge_idx
            .iter()
            .map(|&e| csr.epochs_per_sample[e as usize])
            .collect();
        let next_neg: Vec<f32> = csr
            .csr_edge_idx
            .iter()
            .map(|&e| epochs_per_neg[e as usize])
            .collect();
        let zeros = vec![0.0f32; n * n_dim];

        Ok(Self {
            embd: GpuTensor::from_slice(embd, vec![n * n_dim], client)?,
            m: GpuTensor::from_slice(&zeros, vec![n * n_dim], client)?,
            v: GpuTensor::from_slice(&zeros, vec![n * n_dim], client)?,
            grad: GpuTensor::from_slice(&zeros, vec![n * n_dim], client)?,
            offsets: GpuTensor::from_slice(&csr.node_edge_offsets, vec![n + 1], client)?,
            edge_idx: GpuTensor::from_slice(&csr.csr_edge_idx, vec![n_entries], client)?,
            other: GpuTensor::from_slice(&csr.csr_other_node, vec![n_entries], client)?,
            epochs_per_sample: GpuTensor::from_slice(
                &csr.epochs_per_sample,
                vec![csr.n_edges],
                client,
            )?,
            epochs_per_neg: GpuTensor::from_slice(&epochs_per_neg, vec![csr.n_edges], client)?,
            next_sample: GpuTensor::from_slice(&next_sample, vec![n_entries], client)?,
            next_neg: GpuTensor::from_slice(&next_neg, vec![n_entries], client)?,
            wg,
            cubes_node: plan("v06_grad_accum", n)?,
            cubes_elem: plan("v06_adam_update", n * n_dim)?,
        })
    }
}

/////////////
// Kernels //
/////////////

/// `(a * b) mod m` without overflowing `u32`, for `a, b < m < 2^20`.
///
/// ### Params
///
/// * `a`, `b` - Factors, both below `m`
/// * `m` - Modulus
///
/// ### Returns
///
/// The product modulo `m`.
#[cube]
fn v06_mulmod(a: u32, b: u32, m: u32) -> u32 {
    let mask = (1u32 << V06_MULMOD_LIMB) - 1u32;
    let hi = (a * (b >> V06_MULMOD_LIMB)) % m;
    ((hi << V06_MULMOD_LIMB) + a * (b & mask)) % m
}

/// Gradient accumulation of the 0.6 Adam. One thread per node.
///
/// Walks the node's CSR row. Every entry whose clock has come due pulls the
/// node towards its neighbour and draws its due negatives,
/// `order[(win_start + (p * (epoch + q + 1)) mod modulus) mod n]` for CSR
/// position `p` and draw `q`, each repelling with the tanh soft clip. The
/// thread owns its row's clocks, so it advances them in place. Writes the
/// node's gradient in full.
///
/// ### Params
///
/// * `embd` - Embedding `[n * n_dim]`
/// * `offsets` - CSR row pointers `[n + 1]`
/// * `edge_idx` - Edge index per CSR entry
/// * `other` - Other endpoint per CSR entry
/// * `epochs_per_sample` - Sampling period per edge
/// * `epochs_per_neg` - Negative period per edge
/// * `next_sample` - Next sample epoch per CSR entry, advanced in place
/// * `next_neg` - Negative clock per CSR entry, advanced in place
/// * `order` - Negative order `[n]`
/// * `ranks` - Rank of each node in `order` `[n]`; read only when `windowed`
/// * `grad` - Gradient output `[n * n_dim]`
/// * `n` - Number of nodes
/// * `epoch` - Current epoch
/// * `epoch_f` - Current epoch as a float
/// * `a`, `b` - Curve parameters
/// * `attr_coeff` - `attr_scale * 2 * a * b`
/// * `two_gamma_b` - `2 * gamma * b`
/// * `gamma` - Repulsion strength
/// * `clip` - `min(gamma, V06_REP_CLIP_MAX)`
/// * `rep_eps` - Additive epsilon on the repulsive squared distance
/// * `neg_scale` - Hard-negative repulsion scale
/// * `range` - Hard-negative window
/// * `modulus` - Hash modulus, the window or `n`
/// * `wg_size` - Workgroup size; comptime
/// * `n_dim_ct` - Embedding dimension; comptime, loops over it unroll
/// * `windowed` - Whether negatives come from a window around the node's rank;
///   comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> node
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn v06_grad_accum<F: Float + CubeElement + LaunchArg>(
    embd: &Tensor<F>,
    offsets: &Tensor<u32>,
    edge_idx: &Tensor<u32>,
    other: &Tensor<u32>,
    epochs_per_sample: &Tensor<F>,
    epochs_per_neg: &Tensor<F>,
    next_sample: &mut Tensor<F>,
    next_neg: &mut Tensor<F>,
    order: &Tensor<u32>,
    ranks: &Tensor<u32>,
    grad: &mut Tensor<F>,
    n: u32,
    epoch: u32,
    epoch_f: F,
    a: F,
    b: F,
    attr_coeff: F,
    two_gamma_b: F,
    gamma: F,
    clip: F,
    rep_eps: F,
    neg_scale: F,
    range: u32,
    modulus: u32,
    #[comptime] wg_size: u32,
    #[comptime] n_dim_ct: u32,
    #[comptime] windowed: bool,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if node >= n {
        terminate!();
    }
    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let base_self = node * n_dim_ct;

    let mut cur = Array::<F>::new(n_dim_ct as usize);
    let mut g = Array::<F>::new(n_dim_ct as usize);
    #[unroll]
    for d in 0..n_dim_ct {
        cur[d as usize] = embd[(base_self + d) as usize];
        g[d as usize] = zero;
    }

    let mut win_start = 0u32;
    if windowed {
        let center = ranks[node as usize];
        let half = range / 2u32;
        if center > half {
            win_start = center - half;
        }
        if win_start + range > n {
            win_start = n - range;
        }
    }

    let start = offsets[node as usize];
    let end = offsets[(node + 1u32) as usize];
    let mut p = start;
    while p < end {
        let ns = next_sample[p as usize];
        if ns <= epoch_f {
            let e = edge_idx[p as usize];
            let base_j = other[p as usize] * n_dim_ct;

            let mut dsq = zero;
            #[unroll]
            for d in 0..n_dim_ct {
                let diff = cur[d as usize] - embd[(base_j + d) as usize];
                dsq += diff * diff;
            }
            if dsq > zero {
                let dsq_b = F::powf(dsq, b);
                let coeff = -attr_coeff * dsq_b / (dsq * (a * dsq_b + one));
                #[unroll]
                for d in 0..n_dim_ct {
                    g[d as usize] += coeff * (cur[d as usize] - embd[(base_j + d) as usize]);
                }
            }
            next_sample[p as usize] = ns + epochs_per_sample[e as usize];

            let epn = epochs_per_neg[e as usize];
            let nn = next_neg[p as usize];
            let n_neg_f = F::floor((epoch_f - nn) / epn);
            let mut n_neg = 0u32;
            if n_neg_f > zero {
                n_neg = u32::cast_from(n_neg_f);
            }
            let p_mod = p % modulus;
            let mut q = 0u32;
            while q < n_neg {
                let hash = v06_mulmod(p_mod, (epoch + q + 1u32) % modulus, modulus);
                let base_k = order[((win_start + hash) % n) as usize] * n_dim_ct;
                let mut dsq_k = zero;
                #[unroll]
                for d in 0..n_dim_ct {
                    let diff = cur[d as usize] - embd[(base_k + d) as usize];
                    dsq_k += diff * diff;
                }
                if dsq_k > zero {
                    let coeff = two_gamma_b / ((rep_eps + dsq_k) * (a * F::powf(dsq_k, b) + one));
                    if coeff > zero {
                        let norm = coeff * F::sqrt(dsq_k);
                        let c = neg_scale * coeff * clip * F::tanh(norm / gamma) / norm;
                        #[unroll]
                        for d in 0..n_dim_ct {
                            g[d as usize] += c * (cur[d as usize] - embd[(base_k + d) as usize]);
                        }
                    }
                }
                q += 1u32;
            }
            next_neg[p as usize] = nn + F::cast_from(n_neg) * epn;
        }
        p += 1u32;
    }

    #[unroll]
    for d in 0..n_dim_ct {
        grad[(base_self + d) as usize] = g[d as usize];
    }
}

/// Adam step of the 0.6 optimiser. One thread per coordinate; coordinates
/// with a zero gradient keep their moments and position, as the reference.
///
/// ### Params
///
/// * `grad` - Gradients `[n * n_dim]`
/// * `m`, `v` - Adam moments, updated in place
/// * `embd` - Embedding, updated in place
/// * `n_total` - `n * n_dim`
/// * `step` - `alpha / (1 - beta1^t)`
/// * `inv_bc2` - `1 / (1 - beta2^t)`
/// * `beta1`, `beta2` - Moment decays
/// * `eps` - Denominator epsilon
/// * `wg_size` - Workgroup size; comptime
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` ->
///   coordinate
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn v06_adam_update<F: Float + CubeElement + LaunchArg>(
    grad: &Tensor<F>,
    m: &mut Tensor<F>,
    v: &mut Tensor<F>,
    embd: &mut Tensor<F>,
    n_total: u32,
    step: F,
    inv_bc2: F,
    beta1: F,
    beta2: F,
    eps: F,
    #[comptime] wg_size: u32,
) {
    let i = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if i >= n_total {
        terminate!();
    }
    let g = grad[i as usize];
    if g != F::new(0.0_f32) {
        let one = F::new(1.0_f32);
        let mi = beta1 * m[i as usize] + (one - beta1) * g;
        let vi = beta2 * v[i as usize] + (one - beta2) * g * g;
        m[i as usize] = mi;
        v[i as usize] = vi;
        embd[i as usize] += step * mi / (F::sqrt(vi * inv_bc2) + eps);
    }
}

//////////
// Main //
//////////

/// Run the 0.6 Adam layout on the GPU.
///
/// Same schedule, negatives and adaptation as
/// [`optimise_layout_v06`]; a seed draws the same orders as the CPU run.
/// Differs only in float rounding (device `powf` against the CPU's lookup
/// table).
///
/// ### Params
///
/// * `embd` - Flat row-major embedding `[n * n_dim]`, modified in place
/// * `n_dim` - Embedding dimension
/// * `csr` - Graph in CSR form
/// * `cfg` - Layout configuration
/// * `device` - CubeCL device
/// * `seed` - Seed for node orders, projections and the scale adaptation
/// * `verbose` - `0` silent, `1` or `2` print the epoch every 50 epochs
///
/// ### Returns
///
/// `Ok(())`, `ManifoldsError::GraphSizeMismatch` if the graph and the
/// embedding disagree, or a device error.
fn optimise_layout_v06_gpu(
    embd: &mut [f32],
    n_dim: usize,
    csr: &UmapCsrGraph<f32>,
    cfg: &V06Config<f32>,
    device: Device,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError> {
    let verbosity = parse_verbosity_level(verbose);
    let n = csr.n;
    if embd.len() != n * n_dim {
        return Err(ManifoldsError::GraphSizeMismatch {
            n_graph: n,
            n_embd: embd.len() / n_dim.max(1),
        });
    }

    let small = n <= V06_SMALL_LAYOUT_MAX;
    let hard_negatives = cfg.neg_selection_range < n;
    let windowed = hard_negatives && !small;
    let range = n.min(V06_MIN_NEG_SELECTION_RANGE.max(cfg.neg_selection_range));
    let modulus = if small { n } else { range };
    let mut neg_scale = HardNegScale::new(cfg.neg_selection_range, n);
    let lut = SplitPowLut::new(cfg.b);
    let attr_scale = if small {
        V06_SMALL_ATTRACTION_SCALE as f32
    } else {
        1.0
    };

    let client = device.client();
    let neg_period_scale = V06_NEG_PERIOD_SCALE as f32 / cfg.neg_sample_rate.max(1) as f32;
    let state = V06GpuState::upload(embd, n_dim, csr, neg_period_scale, &client)?;
    let wg = state.wg;

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut adapt_rng = SmallRng::seed_from_u64(seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
    let (mut order, mut ranks): (Vec<u32>, Vec<u32>) = if windowed {
        projection_order(embd, n_dim, &mut rng)
    } else {
        ((0..n as u32).collect(), vec![0u32; 1])
    };
    let mut order_gpu = GpuTensor::from_slice(&order, vec![n], &client)?;
    let mut ranks_gpu = GpuTensor::from_slice(&ranks, vec![ranks.len()], &client)?;

    let alphas = alpha_schedule(cfg.n_epochs, cfg.lr as f64, cfg.good_init);
    let (beta1, beta2) = (V06_BETA1 as f32, V06_BETA2 as f32);
    let two_gamma_b = 2.0 * cfg.gamma * cfg.b;
    let clip = cfg.gamma.min(V06_REP_CLIP_MAX as f32);

    for epoch in 0..cfg.n_epochs {
        if windowed && HardNegScale::due(epoch, cfg.n_epochs) {
            neg_scale.update(negative_force_ratio(
                embd,
                n_dim,
                &order,
                cfg.neg_selection_range.max(1),
                cfg,
                &lut,
                &mut adapt_rng,
            ));
        }
        let scale = if small { 1.0 } else { neg_scale.scale as f32 };

        unsafe {
            v06_grad_accum::launch_unchecked::<f32>(
                &client,
                state.cubes_node.clone(),
                CubeDim::new_1d(wg),
                state.embd.into_tensor_arg(),
                state.offsets.into_tensor_arg(),
                state.edge_idx.into_tensor_arg(),
                state.other.into_tensor_arg(),
                state.epochs_per_sample.into_tensor_arg(),
                state.epochs_per_neg.into_tensor_arg(),
                state.next_sample.into_tensor_arg(),
                state.next_neg.into_tensor_arg(),
                order_gpu.into_tensor_arg(),
                ranks_gpu.into_tensor_arg(),
                state.grad.into_tensor_arg(),
                n as u32,
                epoch as u32,
                epoch as f32,
                cfg.a,
                cfg.b,
                attr_scale * 2.0 * cfg.a * cfg.b,
                two_gamma_b,
                cfg.gamma,
                clip,
                V06_REP_EPS as f32,
                scale,
                range as u32,
                modulus as u32,
                wg,
                n_dim as u32,
                windowed,
            );
        }

        let t = (epoch + 1) as i32;
        let bc1 = 1.0 - V06_BETA1.powi(t);
        let bc2 = 1.0 - V06_BETA2.powi(t);
        unsafe {
            v06_adam_update::launch_unchecked::<f32>(
                &client,
                state.cubes_elem.clone(),
                CubeDim::new_1d(wg),
                state.grad.into_tensor_arg(),
                state.m.into_tensor_arg(),
                state.v.into_tensor_arg(),
                state.embd.into_tensor_arg(),
                (n * n_dim) as u32,
                (alphas[epoch] / bc1) as f32,
                (1.0 / bc2) as f32,
                beta1,
                beta2,
                V06_ADAM_EPS as f32,
                wg,
            );
        }

        if hard_negatives {
            embd.copy_from_slice(&state.embd.clone().read(&client)?);
            (order, ranks) = projection_order(embd, n_dim, &mut rng);
            ranks_gpu = GpuTensor::from_slice(&ranks, vec![n], &client)?;
        } else {
            order.shuffle(&mut rng);
        }
        order_gpu = GpuTensor::from_slice(&order, vec![n], &client)?;

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == cfg.n_epochs) {
            // launches are async; sync so the print tracks the device
            sync_client(&client)?;
            println!("  Epoch {}/{}", epoch + 1, cfg.n_epochs);
        }
    }

    embd.copy_from_slice(&state.embd.read(&client)?);
    Ok(())
}

/// Optimise a UMAP embedding with the umap-learn 0.6 Adam on the GPU.
///
/// Rescales the initialisation with `rescale_isotropic` on the host, then
/// runs the layout on the device. `beta1`, `beta2` and `eps` in `params` are
/// ignored; the 0.6 constants are fixed.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][n_dim]`, modified in place
/// * `graph` - Symmetric adjacency list
/// * `params` - Curve, learning rate, repulsion, epochs and negative rate
/// * `good_init` - Use the short warm-up meant for a good initialisation
/// * `device` - CubeCL device
/// * `seed` - Random seed
/// * `verbose` - `0` silent, `1` or `2` print the epoch every 50 epochs
///
/// ### Returns
///
/// `Ok(())`, the graph construction error or a device error.
pub fn optimise_embedding_adam_v06_gpu(
    embd: &mut [Vec<f32>],
    graph: &[Vec<(usize, f32)>],
    params: &UmapOptimParams<f32>,
    good_init: bool,
    device: Device,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError> {
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let n_dim = embd[0].len();
    let mut flat: Vec<f32> = embd.iter().flatten().copied().collect();
    rescale_isotropic(&mut flat, n_dim, params.gamma as f64);

    let csr = UmapCsrGraph::from_graph(graph)?;
    let cfg = V06Config {
        a: params.a,
        b: params.b,
        gamma: params.gamma,
        lr: params.lr,
        n_epochs: params.n_epochs,
        neg_sample_rate: params.neg_sample_rate,
        neg_selection_range: V06_DEFAULT_NEG_SELECTION_RANGE,
        good_init,
    };
    optimise_layout_v06_gpu(&mut flat, n_dim, &csr, &cfg, device, seed, verbose)?;

    for (point, chunk) in embd.iter_mut().zip(flat.chunks_exact(n_dim)) {
        point.copy_from_slice(chunk);
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use rand::Rng;

    /// Default wgpu device, or `None` without a usable adapter.
    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::default();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <WgpuRuntime>::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Two rings of `n_per` nodes, each node linked to its `k` ring
    /// neighbours on either side.
    fn two_rings(n_per: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
        let mut graph = vec![Vec::new(); 2 * n_per];
        for c in 0..2 {
            for i in 0..n_per {
                for o in 1..=k {
                    let a = c * n_per + i;
                    let b = c * n_per + (i + o) % n_per;
                    graph[a].push((b, 1.0));
                    graph[b].push((a, 1.0));
                }
            }
        }
        graph
    }

    /// Run CPU and GPU from the same start and seed, return the largest
    /// coordinate gap relative to the embedding's span.
    fn cpu_gpu_gap(n_per: usize, range: usize, n_epochs: usize) -> Option<f32> {
        let device = try_device()?;
        let graph = two_rings(n_per, 5);
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let p = UmapOptimParams::<f32>::default_2d();
        let cfg = V06Config {
            a: p.a,
            b: p.b,
            gamma: 1.0,
            lr: 1.0,
            n_epochs,
            neg_sample_rate: 5,
            neg_selection_range: range,
            good_init: false,
        };
        let mut rng = SmallRng::seed_from_u64(3);
        let mut cpu: Vec<f32> = (0..4 * n_per)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        rescale_isotropic(&mut cpu, 2, 1.0);
        let mut gpu = cpu.clone();

        optimise_layout_v06(&mut cpu, 2, &csr, &cfg, 11, 0).unwrap();
        optimise_layout_v06_gpu(&mut gpu, 2, &csr, &cfg, device.into(), 11, 0).unwrap();

        assert!(gpu.iter().all(|x| x.is_finite()));
        let span = cpu.iter().fold(0.0f32, |s, x| s.max(x.abs()));
        let gap = cpu
            .iter()
            .zip(&gpu)
            .fold(0.0f32, |s, (a, b)| s.max((a - b).abs()));
        Some(gap / span)
    }

    #[test]
    fn test_v06_gpu_matches_cpu_small_layout() {
        if let Some(gap) = cpu_gpu_gap(200, V06_DEFAULT_NEG_SELECTION_RANGE, 5) {
            assert!(gap < 1e-3, "relative gap {gap}");
        }
    }

    #[test]
    fn test_v06_gpu_matches_cpu_uniform_negatives() {
        if let Some(gap) = cpu_gpu_gap(1500, V06_DEFAULT_NEG_SELECTION_RANGE, 5) {
            assert!(gap < 1e-3, "relative gap {gap}");
        }
    }

    #[test]
    fn test_v06_gpu_matches_cpu_hard_negatives() {
        // a few epochs only: rounding eventually swaps two near-tied ranks in
        // the projection order, after which CPU and GPU draw different
        // negatives (as two CPU runs do from starts 1e-7 apart)
        if let Some(gap) = cpu_gpu_gap(1500, 500, 3) {
            assert!(gap < 1e-3, "relative gap {gap}");
        }
    }
}

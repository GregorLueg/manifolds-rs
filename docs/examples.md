# Examples

One worked example per method. The parameter structs live at the crate root,
the shared helpers (synthetic data, `run_ann_search`, `ManifoldsError`) in the
prelude, so both glob imports are worth having. Every entry point returns
`Result<Vec<Vec<T>>, ManifoldsError>`; the examples `.unwrap()` for brevity.

## UMAP

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(
    1000,  // n_samples
    50,    // dimensionality
    5,     // n_clusters
    42,    // seed
);

// Configure UMAP parameters
let params = UmapParams::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
);

// Run UMAP
let embedding = umap(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
```

`UmapParams::default()` uses the `"adam_parallel"` optimiser, `k = 15` and
spectral initialisation. For full control over every field (optimiser, ANN
backend, graph parameters, epochs) use `UmapParams::new`.

## densMAP

densMAP is UMAP plus a gradient term that keeps local density interpretable.
`lambda` controls how hard it pushes; larger values preserve density more
aggressively at the cost of cluster separation. The default is `2.0`, and the
term is only active over the final 30% of the epochs.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure densMAP parameters
let params = DensmapParams::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
    Some(2.0),   // lambda (density weight)
);

// Run densMAP
let embedding = densmap(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

For finer control, `DensmapParams::new` takes a full `UmapParams` and a
`DensParams` (`lambda`, `frac`, `var_shift`). Setting `lambda = 0.0` recovers
plain UMAP.

## t-SNE

Below are examples of how to use t-SNE.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure t-SNE parameters
let params = TsneParams::new_default_2d(
    Some(30.0),   // perplexity
);

// Run t-SNE (Barnes-Hut)
let embedding = tsne(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    "bh",        // "bh", "qd", "fft" | "fft_3k" (fft_tsne), "fft_3k_mlx" (mlx)
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

`"qd"` is the quick-and-dirty Barnes-Hut after
[qdtsne](https://github.com/libscran/qdtsne): the tree depth is capped at
`TsneOptimParams::max_depth` (default 7) and repulsion is computed once per
leaf. On 20k points it halved the optimiser time against plain Barnes-Hut for a
small loss in local structure.

## den-SNE

Same idea as densMAP, applied to t-SNE. The default `lambda` here is `0.1`,
twenty times smaller than densMAP's, because the Student-t gradients are much
larger.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure den-SNE parameters
let params = DensneParams::new_default_2d(
    Some(30.0),   // perplexity
    Some(0.1),    // lambda (density weight)
);

// Run den-SNE
let embedding = densne(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    "bh",        // approximation: "barnes_hut" | "bh", "qd" | "bh_qd", or "fft" | "fft_3k" (fft_tsne feature)
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

Both `densmap` and `densne` return `ManifoldsError::DegenerateLocalRadii` if
every point ends up with the same local radius, since the correlation is then
undefined.

## DREAMS

t-SNE regularised towards a global reference layout (Kury, Kobak & Damrich,
TMLR 2026), so you keep the clusters and get a sensible arrangement of them.
`lambda` slides from plain t-SNE (0) to the reference (1); 0.15 is the paper's
default. The reference is PCA unless you hand one in as `[2][n_samples]`, the
layout every embedding here returns (MDS, PHATE, whatever you trust globally).

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

let params = DreamsParams::new_default_2d(
    Some(30.0),   // perplexity
    Some(0.15),   // lambda
);

let embedding = dreams(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    None,        // reference (None = PCA)
    &params,
    "bh",        // same approximations as t-SNE
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

`dreams_gpu` with `DreamsParamsGpu` runs the regulariser inside the
`"fft_3k_gpu"` optimiser as well.

## Precomputed kNN

Every entry point takes a precomputed k-nearest neighbour graph, which pays off
when you run several embeddings over the same data:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

let (data, _) = generate_clustered_data(500, 50, 5, 42);

// Compute k-NN once
let nn_params = NearestNeighbourParams::default();
let (knn_indices, knn_dist) = run_ann_search(
    data.as_ref(),
    15,              // k
    "hnsw".to_string(),
    &nn_params,
    42,              // seed
    1,               // verbose -> light levels of verbosity
)
.unwrap();

// Use precomputed k-NN for UMAP
let params = UmapParams::new_default_2d(None, None);
let embedding = umap(
    data.as_ref(),
    Some((knn_indices.clone(), knn_dist.clone())),
    &params,
    42,
    0,              // no verbose
)
.unwrap();
```

The indices and distances must exclude the point itself, which is what
`run_ann_search` gives you. The graph is taken as-is, no `k` check, so make it
match the `k` in the parameter struct. PaCMAP is the exception: it needs
`k >= mn_candidate_end` for mid-near pair sampling.

## GPU (`gpu` feature)

GPU kernels run in `f32`, so the data has to be `f32` as well.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use cubecl::wgpu::WgpuDevice;
use faer::Mat;

// Generate synthetic clustered data and cast it down to f32
let (data, _) = generate_clustered_data(
    10_000,  // n_samples
    50,      // dimensionality
    5,       // n_clusters
    42,      // seed
);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

// Configure GPU UMAP parameters
// Defaults to the "adam_gpu" optimiser
let params = UmapParamsGpu::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
);

// Run GPU UMAP. ann_type defaults to "nndescent_gpu"; alternatives are
// "exhaustive_gpu" (deterministic, brute force) and "ivf_gpu".
let device = WgpuDevice::default().into(); // a cubecl::Device
let embedding = umap_gpu::<f32>(
    data.as_ref(),
    None,        // precomputed kNN
    &params,
    device,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

GPU t-SNE works analogously. With `"bh"` only the kNN search runs on the GPU;
`"fft_3k_gpu"` moves the optimiser there as well:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use cubecl::wgpu::WgpuDevice;
use faer::Mat;

let (data, _) = generate_clustered_data(10_000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

let params = TsneParamsGpu::new_default_2d(
    Some(30.0),   // perplexity
);

let device = WgpuDevice::default().into(); // a cubecl::Device
let embedding = tsne_gpu::<f32>(
    data.as_ref(),
    None,
    &params,
    "fft_3k_gpu", // or "bh", "qd"; "fft_3k_mlx" with the mlx feature
    device,
    42,
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

densMAP has a GPU path too, via `densmap_gpu` and `DensmapParamsGpu`:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use cubecl::wgpu::WgpuDevice;
use faer::Mat;

let (data, _) = generate_clustered_data(10_000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

let params = DensmapParamsGpu::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
    Some(2.0),   // lambda (density weight)
);

let device = WgpuDevice::default().into(); // a cubecl::Device
let embedding = densmap_gpu::<f32>(
    data.as_ref(),
    None,
    &params,
    device,
    42,
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

With the default `"adam_gpu"` optimiser the density term runs on the device as
well: one kernel computes the embedding radii, another writes the per-node
correlation sensitivities. The radii are read back in between so the
correlation statistics can be accumulated in `f64` on the host, since summing
`n` log-radii in `f32` loses its significant digits. That readback is a sync
point, but it only happens over the final 30% of the epochs. Pick any of the
other optimisers (`"sgd"`, `"adam"`, `"adam_parallel"`) and the density path
falls back to CPU, with only the kNN search on the GPU.

The GPU entry points take a `cubecl::Device`, which picks the runtime at run
time rather than through a type parameter. For CUDA, enable cubecl's `cuda`
feature and pass `cubecl::Device::Cuda(CudaDevice::default())`. On Linux CI,
Vulkan via `mesa-vulkan-drivers` (lavapipe) is the simplest path for headless
testing.

## Parametric UMAP (`parametric` feature)

Parametric UMAP learns a neural network encoder that can transform new data
points:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use burn::tensor::Device;
use faer::Mat;

// Generate synthetic clustered data in f32
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

// Configure the training parameters
let fit_params = TrainParametricParams::from_min_dist_spread(
    0.1,       // min_dist
    1.0,       // spread
    0.0,       // corr_weight
    None,      // learning rate
    Some(100), // n_epochs
    Some(16),  // batch_size
    None,      // negative sample rate
);

let params = ParametricUmapParams::new(
    2,                                   // n_dim (output dimensions)
    15,                                  // k (number of neighbours)
    "hnsw".into(),                       // ann_type
    vec![128, 64],                       // hidden_layers (encoder architecture)
    NearestNeighbourParams::default(),   // nn_params
    UmapGraphParams::default(),          // umap_graph_params
    fit_params,                          // training parameters
);

// Any burn device; autodiff is switched on internally
let device = Device::flex();

// Train parametric UMAP
let embedding = parametric_umap::<f32>(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    &device,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

`parametric_umap` throws the encoder away once it has the embedding. Want to
keep it, to embed new points later or write it to disk? Use
`train_parametric_umap_model`, which returns `(embedding, TrainedUmapModel)`.

## PHATE

PHATE is well-suited for data with continuous structure and branching
trajectories, such as single-cell differentiation data.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate a synthetic branching trajectory
let branches = generate_example_branches(&TrajectoryTopology::DeepBifurcation);
let (data, branch_assignments) = generate_trajectory(
    1000,        // n_samples
    &branches,   // branch topology
    50,          // dimensionality
    0.5,         // noise
    42,          // seed
);

// Configure PHATE parameters
let diffusion_params = PhateDiffusionParams::new(
    Some(40.0),              // decay
    1.0,                     // bandwidth_scale
    1e-4,                    // thresh
    "average".to_string(),   // graph_symmetry
    None,                    // n_landmarks (None = full operator)
    "spectral".to_string(),  // landmark_method
    None,                    // n_svd
    None,                    // t_max (None = default cap)
    None,                    // t_custom (None = auto-select via VNE knee)
    1.0,                     // gamma
);

let params = PhateParams::new(
    2,                                   // n_dim (output dimensions)
    5,                                   // k (number of neighbours)
    "kmknn".to_string(),                 // ann_type
    NearestNeighbourParams::default(),   // ann_params
    diffusion_params,
    "sgd_dense".to_string(),             // mds_method
    None,                                // mds_iter
    true,                                // randomised
);

// Run PHATE
let embedding = phate(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    params,      // note: consumed by value, not borrowed
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

`PhateParams::new_default_2d(Some(k))` gives the same thing with sensible
defaults if you only want to tune the neighbour count.

## PaCMAP

PaCMAP preserves both local and global structure via three pair types (near,
mid-near, and further pairs) and a phased optimisation schedule. **Warning:**
Prior to version `"0.3.6"` there is a nasty bug in the PaCMAP implementation.
Please use `"0.3.6"` and later for correct PaCMAP.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure PaCMAP parameters
let params = PacmapParams::default();

// Run PaCMAP
let embedding = pacmap(
    data.as_ref(),
    None,    // precomputed kNN (None = compute internally)
    &params,
    42,      // seed
    1,       // verbose -> light levels of verbosity
)
.unwrap();
```

The defaults are 10 near, 5 mid-near and 20 further pairs, PCA initialisation
and the `"adam_parallel"` optimiser. `PacmapParams::new_default_2d(Some(n))`
gives the same with a different near-pair count.

## ForceAtlas2

ForceAtlas2 lays out the fuzzy kNN graph (the same one UMAP builds) with
repulsion between all node pairs, attraction along edges and gravity towards
the origin. Node masses are `1 + degree`, so hubs push harder. Layouts grow
with the data: expect coordinates in the thousands to tens of thousands.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Default: k = 15, spectral initialisation, 500 epochs, Gephi's settings
let params = Fa2Params::default();

// Run ForceAtlas2
let embedding = forceatlas2(
    data.as_ref(),
    None,    // precomputed kNN (None = compute internally)
    &params,
    "bh",    // repulsion approximation: Barnes-Hut
    42,      // seed
    1,       // verbose -> light levels of verbosity
)
.unwrap();
```

The knobs live in `params.optim_params` (`Fa2OptimParams`): `scaling_ratio`
(repulsion), `gravity`, `strong_gravity`, `lin_log` (tighter communities),
`dissuade_hubs`, `edge_weight_influence`, `jitter_tolerance` and the
Barnes-Hut `theta` (on Gephi's scale, default `1.2`).

Already have a graph, say an SNN graph from R? Hand it over directly. It must
be symmetric (both directions stored with equal weights); self-loops are
dropped.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Undirected triangle, both directions stored
let graph = CoordinateList {
    row_indices: vec![0, 1, 1, 2, 2, 0],
    col_indices: vec![1, 0, 2, 1, 0, 2],
    values: vec![1.0_f64; 6],
    n_samples: 3,
};

let embedding = forceatlas2_from_graph(
    &graph,
    None,                        // initial layout (None = random in [-10, 10])
    &Fa2OptimParams::default(),
    "bh",
    42,                          // seed
    0,                           // verbose
)
.unwrap();
```

## Diffusion maps

Classical diffusion maps embed the data via the top eigenvectors of a
row-stochastic diffusion operator, optionally with anisotropic normalisation
to correct for non-uniform sampling density. Setting `alpha_norm = 1.0`
recovers the Laplace-Beltrami operator; `0.5` gives the Fokker-Planck
operator; `0.0` gives the normalised graph Laplacian.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, _) = generate_clustered_data(1000, 50, 5, 42);

// Configure diffusion maps parameters
let params = DiffusionMapsParams::new(
    2,                                   // n_dim (output dimensions)
    5,                                   // k (number of neighbours)
    "kmknn".to_string(),                 // ann_type
    NearestNeighbourParams::default(),   // ann_params
    1.0,                                 // bandwidth_scale
    1e-4,                                // thresh
    "add".to_string(),                   // graph_symmetry
    1.0,                                 // alpha_norm (Laplace-Beltrami)
    PhateTime::Auto { t_max: 100 },      // t (Auto = VNE knee up to t_max)
    None,                                // n_landmarks (None = full operator)
    "spectral".to_string(),              // landmark_method
    None,                                // n_svd
);

// Run diffusion maps
let embedding = diffusion_maps(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    params,      // note: consumed by value, not borrowed
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();

// embedding[0] contains the first non-trivial diffusion component
// embedding[1] contains the second non-trivial diffusion component
```

## MLX (`mlx` feature)

Apple Silicon only, `f32` only. The CPU entry points take the MLX names for the
kNN search and the optimiser, so the whole pipeline runs on MLX without the
`gpu` feature:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use faer::Mat;

let (data, _) = generate_clustered_data(10_000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

let params = UmapParams {
    ann_type: "nndescent_mlx".into(), // or "exhaustive_mlx", "ivf_mlx"
    optimiser: "adam_mlx".into(),
    ..UmapParams::default()
};
let embedding = umap(data.as_ref(), None, &params, 42, 1).unwrap();

let mut params = TsneParams::new_default_2d(Some(30.0));
params.ann_type = "nndescent_mlx".into();
let embedding = tsne(data.as_ref(), None, &params, "fft_3k_mlx", 42, 1).unwrap();
```

`densmap` takes `"adam_mlx"` too, `dreams` takes `"fft_3k_mlx"`. den-SNE has
no MLX path and returns `ManifoldsError::DensityNotSupported`.


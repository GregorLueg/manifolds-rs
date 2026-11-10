# News

Changes to the `manifolds-rs` crate. The Python package `manifolds-rs` has its
own changelog at [`python/CHANGELOG.md`](python/CHANGELOG.md).

## 0.9.1

**Fixes**

Different coordinates than 0.9.0 at the same seed with an automatic `t`: PHATE
and diffusion maps on 5000 or more points (full operator or landmarks), and
`f32` runs of either below that.

- The von Neumann entropy behind the automatic `t` kept only the top 100
  singular values from 5000 points up. The dropped tail carries most of the
  spectral mass at small `t`, so the knee landed far too late: on a real single
  cell PCA, diffusion maps on 10k cells picked `t = 44` where the full spectrum
  gives 12. It now runs stochastic Lanczos quadrature over the whole spectrum,
  which lands within one step of the exact `t` at about the cost of the old path
  (1.4 s on 100k cells).
- Below 5000 points the singular values come from the eigenvalues of `AᵀA`
  rather than a full SVD, in `f64` for every `T`. Same `t`, about half the time,
  which is most of the cost of choosing `t` on 2048 landmarks.
- Spectral landmark selection for diffusion maps picks the point nearest each
  centroid in one pass over the points instead of one pass per landmark. Same
  landmarks.

## 0.9.0

**Breaking changes**

Different coordinates than 0.8.0 at the same seed: PaCMAP, PHATE, parametric
UMAP, UMAP and densMAP with `"sgd"`, diffusion maps with spectral landmarks,
`f32` classic MDS, `f32` PCA initialisation and every `f64` run on an
approximate kNN backend.

- Linear algebra runs in `f64`. `compute_largest_eigenpairs_lanczos`,
  `compute_smallest_eigenpairs_lanczos`, `eigendecompose`, `randomised_svd` and
  `sparse_randomised_svd` return `f64`, and `randomised_svd` and
  `compute_landmark_embedding` take it.
- `run_ann_search` searches in `f32` on every CPU backend bar `"exhaustive"`,
  which stays in `T` for ground truth. Distances come back as `T`.
- PaCMAP follows the reference.
  - `learning_rate` defaults to `1.0` (was `0.01`), `ann_type` to `"ivf"` (was
    `"nndescent"`).
  - Near pairs are the best `n_near` of an `n_near + 50` wide search by locally
    scaled distance. A precomputed graph needs at least `n_near` neighbours and
    should be `n_near + 50` wide.
  - Mid-near pairs are the second closest of six random points. Further pairs
    exclude the near pairs.
  - PCA initialisation is the PC scores times `range` (`None` means `0.01`), so
    the ratio between components survives.
  - The Adam epsilon is no longer scaled by the bias correction.
  - `PacmapParams` drops `mn_candidate_start` and `mn_candidate_end`,
    `construct_pacmap_pairs` takes the data and a metric, and
    `NotEnoughNeighbours` carries `n_near`.
- PHATE follows the reference.
  - `n_landmarks` defaults to `PHATE_DEFAULT_LANDMARKS` (2048), was `None`.
  - `ann_type` defaults to `"ivf"` (was `"nndescent"`).
  - The kNN search is `6k + 5` wide (`phate_search_width`), and neighbours past
    `k` join the kernel while their affinity clears `thresh`. Build a
    precomputed graph that wide.
  - The kernel has a unit diagonal, and the landmark operator sums the kernel,
    not `P`.
  - A zero entry of `P^t` has log potential `-ln(1e-7)`, not `0`.
  - SGD-MDS runs 300 iterations below 1000 points, 500 below 5000, 800 otherwise.
- Renames: `ann_params` is `nn_params` (`PhateParams`, `DiffusionMapsParams`),
  `randomised_init` is `randomised` (t-SNE family, `Fa2Params`), `optimiser_type`
  is `optimiser` (`PacmapParams`), `umap_graph_params` is `graph_params` (UMAP
  family, `ParametricUmapParams`), `parse_initilisation` is
  `parse_initialisation`, `SpareMatrixMustBeSquare` is `SparseMatrixMustBeSquare`
  and `PHATE_DEFAULT_N_SVD` is `LANDMARK_DEFAULT_N_SVD`.
- Diffusion maps use `LANDMARK_DEFAULT_N_SVD` too: spectral landmarks with
  `n_svd = None` take 100 singular vectors, was 10.
- Parametric UMAP has one default curve, fitted to `DEFAULT_MIN_DIST` (0.5) and
  `DEFAULT_SPREAD` as for UMAP. It was `a = 1.5, b = 0.9` in
  `TrainParametricParams::default_2d` and `min_dist = 0.1` in
  `ParametricUmapParams::new_default_2d`.
- UMAP, densMAP, ForceAtlas2, parametric UMAP, their GPU variants and
  `construct_*_graph` helpers cut a precomputed kNN graph wider than `k` to `k`,
  with a warning on stderr. They used every neighbour given.
- Errors, not panics: Manhattan distance in PHATE and landmark diffusion maps
  (`MetricNotSupported`), `"fft"` and `"fft_3k"` without `fft_tsne`
  (`TsneOptimiserNeedsFft`).
- One verbosity grammar. Level 1 prints `Running <name> on <n> samples ...`,
  timed stages, epochs every 50 and `... finished <name> in`. Level 2 adds
  parameters, landmark selection, kNN internals and the GPU UMAP gradient norm.
  Unknown-parameter warnings go to stderr. `train_parametric_umap` takes
  `Verbosity`, not `bool`.
- `"sgd"` applies an edge's negative samples as one step and takes `x^b` from
  a lookup table.
- `ManifoldsFloat` requires `AtomicFloat`. Only matters if you implement it
  yourself.

**Features**

- `phate_gpu`: PHATE with the kNN search and, for spectral landmarks, the
  randomised SVD and k-means on a cubecl device, in `f32`.
- `landmark_method = "spectral_mlx"` for `phate` and `phate_gpu`: the landmark
  SVD and k-means on MLX, in `f32`. Both backends draw the CPU sketch, so the
  singular vectors match the CPU up to `f32` rounding.
- `PhateLandmarks::build_with_backend` and `SpectralBackend`. `build` stays on
  the CPU.
- `"sgd_parallel"` (`optimise_embedding_sgd_parallel`): `"sgd"` run lock-free
  over all threads. Not reproducible run to run.

**Performance**

- PHATE powers the landmark operator as a dense matrix (`dense_matrix_power`,
  `PhateLandmarks::power_dense`). The sparse randomised SVD uses Cholesky QR for
  all but the last power iteration.
- `"sgd"` is about 2.4x faster.
- Software prefetch on aarch64 and x86_64 for scattered reads: SGD and parallel Adam
  negative samples, FFT t-SNE charge spreading, den-SNE attraction.

**Bug fixes**

- PCA initialisation centres and decomposes in `f64`. In `f32` it collapsed large
  layouts onto a few points.
- PaCMAP no longer compresses the layout. Mid-near pairs were kNN slots 4 to 8,
  so already near pairs.
- Spectral landmark selection works at scale: seven power iterations on the
  symmetrically normalised kernel, was two on the raw one.
- PHATE honours `n_svd` and, with landmarks, `mds_iter`.
- PHATE drops empty landmarks instead of keeping them as zero rows.

## 0.8.0

**Breaking changes**

- Spectral init, PHATE and diffusion maps give different coordinates than
  0.7.x at the same seed (new eigensolver). Still bit-identical per seed.
- New `ManifoldsError::TooManyVertices` and `TooManyEdges` variants.
- `ann-search-rs` 0.12.0. NN-Descent (CPU, wgpu, MLX) builds the kNN graph
  directly, no index and no query. Dropped from `NearestNeighbourParams`:
  `diversify_prob`, `ef_budget`, `extract_knn`. Dropped from
  `NearestNeighbourParamsGpu`: `k`, `beam_width`, `max_beam_iters`,
  `n_entry_points`, `extract_knn`.
- kNN defaults follow `k`: `"nndescent"` for UMAP, densMAP, PHATE, PaCMAP,
  diffusion maps, ForceAtlas2 and parametric UMAP; `"ivf"` for tSNE, densNE
  and DREAMS; `"ivf_gpu"` for GPU tSNE. Different embeddings at the same seed.
- `Verbosity` is now the `ann-search-rs` one, re-exported from the prelude.
  Level 1 also prints the kNN build and query, level 2 the index internals.

**Performance**

- Spectral init uses about half the memory on connected graphs and a quarter
  on disconnected ones.
- Lanczos is now thick-restarted with a bounded basis (Wu & Simon, 2000).
  `lanczos_thick_restart` and `LanczosParams` are public.
- UMAP optimisers use 25-40% less memory. Parallel Adam is ~5% slower.
- Leaner tSNE optimisers and graph symmetrisation. Same embeddings per seed.

## 0.7.3

**Chore**

- `faer` without its `npy` feature and `ann-search-rs` 0.11.2. Fixes Windows
  ARM64 cross-builds. Dependants need to drop `npy` too.

## 0.7.2

**Performance**

- Less memory in kNN graph filtering and symmetrisation.

## 0.7.1

**Fixes**

- `parametric` builds burn without default features; `burn-train` and its
  dependencies are gone. Enable `train` on your own `burn` if you need it.

## 0.7.0

**Features**

- DREAMS (Kury, Kobak & Damrich, TMLR 2026) via `dreams()` and `dreams_gpu()`:
  t-SNE regularised towards a global reference layout.
- Experimental `mlx` feature for Apple Silicon: `"adam_mlx"`, `"fft_3k_mlx"`
  and MLX kNN indices. See [`docs/benchmarks_mlx.md`](docs/benchmarks_mlx.md).
- Faster wgpu three-kernel FFT t-SNE.
- `fft_tsne` runs on pure-Rust `rustfft` instead of FFTW.
- `cubecl` 0.11, `burn` 0.22, `faer` 0.24, `cubecl-utils-rs` 0.2.0,
  `ann-search-rs` 0.11.0.

**Fixes**

- `run_ann_search` no longer keeps a point as its own neighbour when the data
  has exact duplicates.

**Breaking changes**

- The `optimise_*_tsne` functions take an extra `dreams` argument; pass `None`.
- New `ManifoldsError` variants for DREAMS and MLX.
- GPU functions take a `cubecl::Device` instead of an `R: Runtime` parameter.
- Parametric UMAP takes a `&burn::tensor::Device` instead of a backend type.
- `FftwFloat` is now `FftFloat`; `FftLane3k` is gone.

## 0.6.2

**Features**

- `ann-search-rs` 0.10.1 with improved k-means.

## 0.6.1

**Features**

- `ann-search-rs` 0.10.0 with faster GPU indices.

## 0.6.0

**Features**

- Quick-and-dirty Barnes-Hut t-SNE (`"qd"`) after
  [qdtsne](https://github.com/libscran/qdtsne), with a capped tree depth.
- `ann-search-rs` 0.9.3.

**Breaking changes**

- `TsneOptimParams` has a new `max_depth` field.

## 0.5.4

**Features**

- `ann-search-rs` 0.9.1 with improved k-means and Accelerate on macOS.

## 0.5.3

**Features**

- Three-kernel FFT t-SNE, on CPU and GPU.

## 0.5.2

**Features**

- Faster FFT t-SNE.

## 0.5.1

**Features**

- ForceAtlas2 via `forceatlas2()` and `forceatlas2_from_graph()`.

**Breaking changes**

- `bh_tree`: `Node::count` is now `Node::mass`, and `BarnesHutTree::rebuild()`
  takes an extra `masses` argument.

## 0.5.0

**Features**

- `ann-search-rs` bumped to recent major version.

## 0.4.1

**Features**

- avx2 and avx512 properly enabled from `ann-search-rs = "0.8.3"` without any
  compile flags.

## 0.4.0

**Features**

- Speed improvements taken from `ann-search-rs = "0.8.0"`
- The NNDescent (GPU and CPU) have now the options to directly extract the kNN
  graph without any beam search. This is faster, but can yield a slightly worse
  graph. Think Recall ~ 0.98 vs Recall ~ 0.99/1.00 compared against exhaustive
  query.

**Breaking changes**

- The Euclidean distance used to return squared Euclidean. This has now been
  changed it the kNN backends will automatically transform the squared L2
  distance to L2.

## 0.3.15

**Feature**

- Faster UMAP, tSNE and PHATE graph generation.

## 0.3.14

**Feature:**

- Version bump to the latest `ann-search-rs` with faster GPU-acceleration for
  the vector searches.

## 0.3.13

**Feature:**

- GPU pieces now depend on `cubecl-utils-rs` which unifies all of the cubecl
  utilties scattered across different crates.
- GPU-accelerated Adam optimiser for Umap is also updated to now use that one.

## 0.3.12

**Feature:**

- Improved spectral initialisation via better layout and parallelisation in
  some parts.
- Improved GPU-accelerated Adam optimiser for even faster UMAP performance.

## 0.3.11

**Feature:**

- Implementations of dens-map and dens-sne, density-preserving versions of
  UMAP and tSNE, see [Narayan et al.](https://www.nature.com/articles/s41587-020-00801-7)

## 0.3.10

**Feature:**

- Version update on `ann-search-rs` to profit from faster GPU-acceleration.

## 0.3.9

**Fix:**

- Version bump to latest `ann-search-rs` that will ensure that the IVF indices
  do return the expected k neighbours and will query more lists.

## 0.3.8

**Fix:**

- The forwarding of parameters was not behaving for the NNDescent wrapper. This
  is fixed now.

## 0.3.7

**Features:**

- Performance improvements for BH and FFT-accelerate tSNE.

## 0.3.6

**Fix:**

- Incorrect PacMAP implementation: gradients were wrongly calculated and one of
  of the parameters was wrongly propagated. This has been fixed now.

## 0.3.5

**Features:**

- GPU-accelerated Adam optimiser for UMAP.
- GPU-accelerated kNN searches pass through `fp32` forced to avoid issues with
  wgpu not supporting `fp64`.

## 0.3.4

**Features:**

- Serialisation added to the trained parametric UMAP model.

## 0.3.3

**Fix:**

- Version bump to the latest `ann-search-rs` (to version `"0.4.3"`) to avoid the
  weird bug between wgpu <> metal affecting the CAGRA approximate nearest
  neighbour search.

## 0.3.2

**Features:**

- Added the option for late exaggeration to tSNE to keep structure on data sets
  with large N.

**Fix:**

- Numerical stability problems for very large data sets with tSNE when the input
  is `fp32`. It now casts to `fp64` independent at specific points to avoid
  collapsing embeddings (at least to some extent. Larger data sets should use
  `fp64` generally speaking.)

## 0.3.1

**Features:**

- Breaking change in terms of how the verbosity parameter is supplied (`usize`
  with options `0`, `1` or `2`, instead of a `bool`) given more finegrained
  control over verbosity.
- Reduced number of k means clustering iterations for the landmark approach ->
  approximation anyways and exact centroids are not needed here.
- Pull in the fixes for GPU-based methods when dimensionality is very high.
- More errors over panics.

## 0.3.0

**Features:**

- Version updates for `ann-search-rs` and better errors across the board
- Better default LR based on a heuristic. and optional late exaggeration rate
  for tSNE.
- Updated interfaces for the parameters which breaks old interfaces, but is more
  Rust idiomatic.
- Better spectral initialisation on disconnected graphs, akin to the
  scikit-learn approach.

## 0.2.4

**Features:**

- Version bump on `burn`.
- Removed unnecessary warning from tSNE.

## 0.2.3

**Features:**

- Version bump on `ann-search-rs`.

## 0.2.2

**Features:**

- Fixed the broken `gpu` feature flag and modified CI/CD to spot this earlier

## 0.2.1

(Yanked - broken `gpu` feature)

**Features:**

- Diffusion maps added.
- Better error handling added.

## 0.2.0

**Features:**

- GPU-accelerated kNN searches available -> supporting a GPU-accelerated version
  of UMAP and tSNE.
- Added KmKnn nearest neighbour search as a default.

## 0.1.15

**Features:**

- Fix: IVF approximate nearest neighbour search can actually be used now.

## 0.1.14

**Features:**

- Version bump to latest version of `ann-search-rs`

## 0.1.13

**Features:**

- Faster parametric UMAP implementation with less data shuffling between CPU
  and GPU.

## 0.1.12

**Features:**

- Version bump of `ann-search-rs` that has faster Annoy and IVF.

## 0.1.11

**Fix:**

- Version bump of `ann-search-rs` to an unyanked version.

## 0.1.10

*(Yanked due to version problem with `ann-search-rs -> enforced MiMalloc as
allocator without the user having a choice.)*

**Features:**

- IVF index added
- Version bump of `ann-search-rs` to take advantage of faster kNN searches
  for various indices.

## 0.1.9

**Features:**

- PaCMAP implemented

## 0.1.8

**Features:**

- PHATE implemented
- Improvements on the UMAP optimisers to be even faster
- Improvements on the tSNE optimisers to be even faster

## 0.1.7

**Features:**

NA

**Fixes:**

- Hotfix from ann-search-rs with avx512 instructions

## 0.1.6

**Features:**

- Function can take pre-computed kNN graphs

## 0.1.5

**Features:**

- Function can take pre-computed kNN graphs

## 0.1.4

**Features:**

- Parametric UMAP added

## 0.1.3

**Features:**

- tSNE added (FFT and BH versions)
- Fixes to the optimisers

## 0.1.2

<span style="color:red">Yanked!</span>

## 0.1.1

**Features:**

- Initial UMAP with SGD

## 0.1.0

<span style="color:red">Yanked!</span>

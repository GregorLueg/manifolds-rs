# News

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

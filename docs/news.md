# News

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

# AnnData / scanpy

`mf.umap` and `mf.tsne` are drop-ins for `sc.tl.umap` and `sc.tl.tsne`. They
write the same slots, so everything downstream (`sc.pl.umap`, `sc.pl.tsne`,
`sc.pl.embedding`) works unchanged. They need the `anndata` extra; scanpy itself
is not a dependency.

```bash
uv pip install 'manifolds-rs[anndata]'
```

## UMAP

```python
import scanpy as sc
import manifolds_rs as mf

sc.pp.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15)
mf.umap(adata)
sc.pl.umap(adata, color="leiden")
```

Like scanpy's, `mf.umap` embeds the `sc.pp.neighbors` graph rather than
searching again. It reads the neighbour indices and distances back out of
`obsp["distances"]` and the crate builds its own fuzzy graph from them, so
`n_neighbors` and the metric are whatever `sc.pp.neighbors` used. Scanpy counts
the cell itself and this does not, so `n_neighbors=15` there is recorded as 14
here. The
representation (`use_rep`, `n_pcs`) comes from the same
`uns["neighbors"]["params"]` and feeds the initialisation.

`obsp["distances"]` has to hold the same number of neighbours on every row,
which it does for `sc.pp.neighbors` with its default settings. A ragged graph is
an error.

## t-SNE

```python
mf.tsne(adata)  # obsm["X_pca"]
mf.tsne(adata, n_pcs=30)  # the first 30 PCs
mf.tsne(adata, use_rep="X_scVI")  # any dense obsm key, or "X"
```

`mf.tsne` does its own `3 * perplexity` neighbour search on the representation,
as `sc.tl.tsne` does. The 15-neighbour graph from `sc.pp.neighbors` is far too
narrow for t-SNE at the usual perplexity. There is no automatic PCA: without
`obsm["X_pca"]` and no `use_rep`, it raises and tells you to run `sc.pp.pca`.

## GPU

```python
mf.umap(adata, device="gpu")
mf.tsne(adata, device="gpu", approx="fft_3k_gpu")
```

`device="gpu"` swaps in `UMAPGpu` / `TSNEGpu`, float32 throughout. See
[GPU](gpu.md) for what runs on the device.

## What gets written

| Call | `obsm` | `uns` |
| --- | --- | --- |
| `mf.umap(adata)` | `"X_umap"` | `"umap"` |
| `mf.tsne(adata)` | `"X_tsne"` | `"tsne"` |
| `key_added="foo"` | `"foo"` | `"foo"` |

`uns[...]["params"]` holds the estimator's scalar parameters, plus `device` and
`neighbors_key` (UMAP) or `use_rep` / `n_pcs` (t-SNE). Only scalars are kept, so
the result survives `write_h5ad`. UMAP's `a` and `b` are not stored: the crate
fits them from `min_dist` and `spread` and does not hand them back.

`copy=True` returns a modified copy and leaves `adata` alone.

## Everything else

Any other constructor argument of the estimator goes through as a keyword:

```python
mf.umap(adata, n_epochs=200, init="pca", optim_params=mf.UmapOptim(gamma=1.5))
mf.tsne(adata, learning_rate=500.0, ann="hnsw", metric="cosine")
```

Unknown names raise, same as the estimator constructors.

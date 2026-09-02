# manifolds-rs

Dimensionality reduction for single-cell and computational biology. The
[Rust crate](https://github.com/GregorLueg/manifolds-rs) does the work. This is a
scikit-learn shaped layer over it.

Seven algorithms on the CPU, three of them with GPU variants where the neighbour
search and the Adam update move to the device. No CUDA runtime to install: the
GPU backend is wgpu, so it runs on Metal, Vulkan or DX12 and ships in the
ordinary wheel.

## Install

```bash
uv pip install manifolds-rs
```

Wheels are built for Linux x86_64 and macOS on both architectures, against
Python 3.10 and up. numpy and beartype are the only runtime dependencies.

## Thirty seconds

```python
import manifolds_rs as mf

X, labels = mf.datasets.clustered(20_000, dim=50, n_clusters=12)

embedding = mf.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```

Parameters in the constructor, data in `fit`, an `(n_samples, n_components)`
array back. `get_params` and `set_params` are there, so `clone` and `Pipeline`
work without scikit-learn being an install requirement.

## The parameter story

The knobs people actually turn are ordinary constructor arguments. Everything
else lives in frozen dataclasses, one per group, and anything you leave alone
keeps the crate's default:

```python
mf.UMAP(
    n_neighbors=30,
    ann="hnsw",
    nn_params=mf.NeighbourParams(m=32, ef_search=200),
    optim_params=mf.UmapOptim(gamma=1.5, neg_sample_rate=10),
)
```

Every parameter has exactly one home. If it is a constructor argument it is not
in the group, so there is never a question of which wins.

## Where to go next

- [Choosing an algorithm](choosing.md) if you don't already know which one you
  want. Seven is a lot of choice and most of them are wrong for your problem.
- [Quickstart](quickstart.md) for worked examples over the synthetic generators,
  including how to score an embedding rather than squint at it.
- [GPU](gpu.md) for the three device-accelerated estimators and what they
  actually buy you.
- [Guide](guide.md) for metrics, precision, threads, reusing the neighbour graph
  and the sharp edges.
- [API reference](api/embeddings.md) for every parameter of every estimator,
  with what each `None` resolves to.

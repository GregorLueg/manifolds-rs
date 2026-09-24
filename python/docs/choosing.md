# Choosing an algorithm

Eight of them, and the honest summary is that three cover most work. Here is how
to pick without running all eight.

## Start here

| Your data | Start with |
| --- | --- |
| Cell types you expect to separate | `UMAP` |
| A differentiation trajectory, or anything continuous | `PHATE` |
| You care about the arrangement of clusters, not just the clusters | `PaCMAP` |
| You want the tightest possible local structure and nothing else | `TSNE` |
| Relative density in the plot should mean something | `DensMAP` |
| You want the spectral embedding itself, not a layout of it | `DiffusionMaps` |
| You want a force-directed graph layout, scanpy `draw_graph` style | `ForceAtlas2` |

## What each one is actually doing

**UMAP** builds a fuzzy simplicial set from the kNN graph and optimises a
cross-entropy against it. The spectral initialisation is where most of its
global structure comes from, and it is the default here for that reason. Fast,
well understood, and the sensible first thing to run.

**densMAP** is UMAP plus a term correlating the embedding's local radii with the
original space's. Plain UMAP is free to stretch a sparse region and squash a
dense one, so relative density in a UMAP plot means nothing at all. If you are
about to say "this cluster is tighter, so those cells are more homogeneous",
you want densMAP. The term only switches on for the last 30% of the run, so it
corrects a settled embedding rather than fighting the layout.

**t-SNE** optimises a KL divergence between neighbourhood distributions. It is
the best of these at local structure and the worst at global: distances between
t-SNE clusters carry very little information, and cluster sizes carry none. The
learning rate here defaults to `max(N / 12, 200)` rather than a fixed 200, so it
does not need retuning as the dataset grows. Two-dimensional only.

**den-SNE** is the same density correction applied to t-SNE. The default weight
is much smaller than densMAP's because the gradients are on a different scale,
not because the effect is meant to be weaker.

**PHATE** powers a diffusion operator to time `t`, takes the potential distance
between the resulting distributions, and lays those out with MDS. That is a
completely different object from a kNN graph layout, and it is why trajectories
and branch points survive PHATE that t-SNE tears into blobs. If your biology is
continuous, this is the one.

**PaCMAP** uses three kinds of pair: near pairs pull, further pairs push, and
mid-near pairs hold the global arrangement together while their weight decays
over the first two phases. That decay is what lets it keep global structure
without leaning on a spectral initialisation. PCA init is close to required;
random init throws away most of the advantage.

**Diffusion maps** is the spectral embedding PHATE builds on. Reach for it when
you want the diffusion coordinates themselves rather than a 2-D picture of
them. The knob worth understanding is `alpha`: `0` gives the normalised graph
Laplacian, `0.5` the Fokker-Planck operator, `1` the Laplace-Beltrami operator,
which is the one that removes the influence of sampling density.

**ForceAtlas2** is Gephi's force-directed layout, run on the same fuzzy kNN
graph UMAP builds, which is what scanpy's `draw_graph` does. Every pair of
points repels, edges attract, and gravity keeps disconnected components from
drifting off. Unlike UMAP there is no edge sampling: every edge is visited every
epoch. `lin_log=True` pulls communities tighter; `scaling_ratio` spreads the
whole layout. `graph_params.mix_weight` has to stay at 1, because the layout
needs a symmetric graph. Two-dimensional only.

## Cost

Rough ordering on the same data, dominated by different things:

- `TSNE`, `DensNE` and `ForceAtlas2` pay for the Barnes-Hut tree every epoch.
- `UMAP`, `DensMAP` and `PaCMAP` pay for the neighbour search and then a cheap
  per-edge update. The parallel Adam optimiser is the default.
- `PHATE` and `DiffusionMaps` pay for an eigendecomposition. Set `n_landmarks`
  above roughly 50k points or you will wait.

For all of them, past a few hundred thousand points the neighbour search
dominates. Switch `ann` to `"nndescent"` or `"hnsw"`, or build the graph once
with [`knn_graph`](api/neighbours.md) and hand it to each estimator.

## What none of them do

Project new points. There is no `transform`. Embedding new data means refitting
on the combined set, which moves the existing coordinates too, and any method
that appears to avoid that is doing something to the new points that is not the
operation the old ones went through.

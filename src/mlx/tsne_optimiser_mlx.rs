//! Three-kernel FFT t-SNE (and DREAMS) on MLX. Same schedule, grid geometry
//! and update rule as the cubecl version in `training::tsne_optimiser_gpu`.
//!
//! Where cubecl needs hand-written pieces, this leans on MLX ops:
//!
//! * box bucketing is an `argsort` of the box ids plus a binary-search kernel
//!   for the box starts, instead of atomics, a scan and a scatter
//! * the convolution is `rfft2`, one broadcast complex multiply against the
//!   three cached kernel spectra and a batched `irfft2`, instead of a custom
//!   power-of-two FFT
//! * `Z`, the recentring mean, the DREAMS norm and the extent are MLX
//!   reductions
//!
//! Custom Metal kernels remain for the per-point work: box assignment and
//! Lagrange weights, the splat, the gather and the fused attraction / gains /
//! momentum step. All `f32`.

use crate::data::graph::coo_to_adjacency_list;
use crate::data::structures::*;
use crate::mlx::ffi::*;
use crate::prelude::*;
use crate::training::tsne_optimiser::*;

////////////
// Consts //
////////////

/// Threads per threadgroup for every kernel here. Not tuned on MLX.
const TSNE_MLX_THREADGROUP: i32 = 256;

/// Epochs between extent readbacks. Same as the cubecl path.
const GRID_CHECK_INTERVAL: usize = 10;

/// Multiple of the extent growth over the last check interval that the grid
/// is sized ahead for. Same as the cubecl path.
const GRID_GROWTH_SAFETY: f64 = 1.5;

/// Entries in the per-epoch f32 parameter buffer of the update kernel, see
/// [`UPDATE_SOURCE`] for the layout.
const N_STEP_PARAMS: usize = 9;

////////////////////
// Kernel sources //
////////////////////

/// Lagrange basis `k` on `NI` equispaced nodes in `[0, 1]` (node `j` at
/// `(j + 0.5) / NI`), evaluated at `t`.
const LAGRANGE_HEADER: &str = r#"
template <int NI>
inline float lagrange_basis(float t, int k) {
    float inv = 1.0f / (float)NI;
    float sk = ((float)k + 0.5f) * inv;
    float num = 1.0f;
    float den = 1.0f;
    for (int j = 0; j < NI; j++) {
        if (j != k) {
            float sj = ((float)j + 0.5f) * inv;
            num *= t - sj;
            den *= sk - sj;
        }
    }
    return num / den;
}
"#;

/// Box id and Lagrange weights per point. A point past the grid edge between
/// extent checks snaps to the edge instead of extrapolating the weights.
///
/// Template int: `NI`. `gu = [n, nb, nf]`, `gp = [coord_min, box_width]`.
const PREP_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint n = gu[0];
    if (i >= n) return;
    uint nb = gu[1];
    float top = (float)(nb - 1);
    float rx = (pos[2 * i] - gp[0]) / gp[1];
    float ry = (pos[2 * i + 1] - gp[0]) / gp[1];
    float fx = clamp(floor(rx), 0.0f, top);
    float fy = clamp(floor(ry), 0.0f, top);
    float tx = clamp(rx - fx, 0.0f, 1.0f);
    float ty = clamp(ry - fy, 0.0f, 1.0f);
    for (int k = 0; k < NI; k++) {
        wx[i * NI + k] = lagrange_basis<NI>(tx, k);
        wy[i * NI + k] = lagrange_basis<NI>(ty, k);
    }
    box_id[i] = (uint)fy * nb + (uint)fx;
"#;

/// Box starts from the sorted box ids: `box_start[b]` is the first sorted
/// position with id `>= b`, by binary search. One thread per box, plus one
/// for the end sentinel.
///
/// `gu = [n, nb, nf]`.
const BOX_START_SOURCE: &str = r#"
    uint b = thread_position_in_grid.x;
    uint n = gu[0];
    uint nb = gu[1];
    if (b > nb * nb) return;
    uint lo = 0;
    uint hi = n;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (sorted_box[mid] < b) lo = mid + 1; else hi = mid;
    }
    box_start[b] = lo;
"#;

/// Unit-charge splat onto the zero-padded grid, one thread per grid node.
/// Node `(gy, gx)` lies in box `(gy / NI, gx / NI)` and sums the weights of
/// that box's points; nodes outside the `m x m` corner stay zero.
///
/// Template int: `NI`. `gu = [n, nb, nf]`.
const SPLAT_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    uint nb = gu[1];
    uint nf = gu[2];
    if (idx >= nf * nf) return;
    uint gy = idx / nf;
    uint gx = idx % nf;
    uint m = nb * NI;
    float acc = 0.0f;
    if (gy < m && gx < m) {
        uint b = (gy / NI) * nb + gx / NI;
        uint iy = gy % NI;
        uint ix = gx % NI;
        for (uint p = box_start[b]; p < box_start[b + 1]; p++) {
            uint pt = order[p];
            acc += wy[pt * NI + iy] * wx[pt * NI + ix];
        }
    }
    grid[idx] = acc;
"#;

/// Interpolate `Z_i` and the raw repulsion per point from the three
/// convolved grids `conv = [q, q^2 dx, q^2 dy]`. Writes `Z_i - 1`, the self
/// term removed, for the global sum.
///
/// Template int: `NI`. `gu = [n, nb, nf]`.
const GATHER_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint n = gu[0];
    if (i >= n) return;
    uint nb = gu[1];
    uint nf = gu[2];
    uint nn = nf * nf;
    uint m = nb * NI;
    uint b = box_id[i];
    uint by = b / nb;
    uint bx = b % nb;
    float z = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    for (int iy = 0; iy < NI; iy++) {
        float w_y = wy[i * NI + iy];
        uint row = (m + by * NI + iy) * nf + m + bx * NI;
        for (int ix = 0; ix < NI; ix++) {
            float w = w_y * wx[i * NI + ix];
            uint o = row + ix;
            z += w * conv[o];
            fx += w * conv[nn + o];
            fy += w * conv[2 * nn + o];
        }
    }
    rep[2 * i] = fx;
    rep[2 * i + 1] = fy;
    zc[i] = z - 1.0f;
"#;

/// Attraction over the CSR graph fused with the Jacobs gains, momentum, step
/// cap and (with `DREAMS`) the regulariser step. Writes fresh position,
/// velocity and gain buffers, so neighbour reads always see the old step.
///
/// Template int: `DREAMS` (0 or 1). `z` and `norm` are 0-d arrays, which MLX
/// binds by value: `sum_i Z_i - n` and `||Y||^2` of the incoming positions. `fp = [exag, lr, momentum,
/// min_gain, max_step, z_floor, inv_ref_norm, dreams_keep, dreams_coef]`.
const UPDATE_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    uint n = gu[0];
    if (i >= n) return;
    float exag = fp[0];
    float lr = fp[1];
    float momentum = fp[2];
    float min_gain = fp[3];
    float max_step = fp[4];
    float z_floor = fp[5];
    float z_raw = z;
    float zz = z_raw > z_floor ? z_raw : 1.0f;

    float px = pos[2 * i];
    float py = pos[2 * i + 1];
    // two independent accumulators keep two neighbour gathers in flight
    float ax0 = 0.0f, ay0 = 0.0f, ax1 = 0.0f, ay1 = 0.0f;
    uint e = indptr[i];
    uint end = indptr[i + 1];
    for (; e + 1 < end; e += 2) {
        uint j0 = indices[e];
        uint j1 = indices[e + 1];
        float dx0 = px - pos[2 * j0];
        float dy0 = py - pos[2 * j0 + 1];
        float dx1 = px - pos[2 * j1];
        float dy1 = py - pos[2 * j1 + 1];
        float f0 = values[e] / (1.0f + dx0 * dx0 + dy0 * dy0);
        float f1 = values[e + 1] / (1.0f + dx1 * dx1 + dy1 * dy1);
        ax0 += f0 * dx0;
        ay0 += f0 * dy0;
        ax1 += f1 * dx1;
        ay1 += f1 * dy1;
    }
    if (e < end) {
        uint j = indices[e];
        float dx = px - pos[2 * j];
        float dy = py - pos[2 * j + 1];
        float f = values[e] / (1.0f + dx * dx + dy * dy);
        ax0 += f * dx;
        ay0 += f * dy;
    }

    float gx = exag * (ax0 + ax1) - rep[2 * i] / zz;
    float gy = exag * (ay0 + ay1) - rep[2 * i + 1] / zz;
    float ux = vel[2 * i];
    float uy = vel[2 * i + 1];
    float g_x = gains[2 * i];
    float g_y = gains[2 * i + 1];
    // Jacobs gains: grow when gradient and update disagree in sign
    g_x = ((gx > 0.0f) != (ux > 0.0f)) ? g_x + 0.2f : g_x * 0.8f;
    g_y = ((gy > 0.0f) != (uy > 0.0f)) ? g_y + 0.2f : g_y * 0.8f;
    g_x = max(g_x, min_gain);
    g_y = max(g_y, min_gain);
    ux = momentum * ux - lr * g_x * gx;
    uy = momentum * uy - lr * g_y * gy;
    float step_sq = ux * ux + uy * uy;
    if (step_sq > max_step * max_step) {
        float scale = max_step / sqrt(step_sq);
        ux *= scale;
        uy *= scale;
    }
    vel_out[2 * i] = ux;
    vel_out[2 * i + 1] = uy;
    gains_out[2 * i] = g_x;
    gains_out[2 * i + 1] = g_y;
    if (DREAMS) {
        float alpha = sqrt(norm) * fp[6];
        float keep = fp[7];
        float coef = fp[8];
        pos_out[2 * i] = px + keep * ux - coef * (px - alpha * reference[2 * i]);
        pos_out[2 * i + 1] = py + keep * uy - coef * (py - alpha * reference[2 * i + 1]);
    } else {
        pos_out[2 * i] = px + ux;
        pos_out[2 * i + 1] = py + uy;
    }
"#;

//////////////
// Geometry //
//////////////

/// FFT grid geometry. The CPU path's box count and half-width; the FFT side
/// is the next power of two that holds the zero-padded `2 m` nodes.
#[derive(Clone, Copy, Debug)]
struct GridGeometry {
    /// FFT grid side
    n_fft: usize,
    /// Boxes per dimension
    n_boxes: usize,
    /// Half-width of the grid in embedding units
    half: f64,
}

impl GridGeometry {
    /// Pick the grid for an embedding extent.
    ///
    /// ### Params
    ///
    /// * `half_span` - Largest absolute coordinate of the embedding
    /// * `ni` - Interpolation nodes per box
    ///
    /// ### Returns
    ///
    /// The geometry.
    fn for_extent(half_span: f64, ni: usize) -> Self {
        let (n_boxes, _, half) = fft_grid_geometry(half_span, TSNE_FFT_MIN_INTERVALS);
        Self {
            n_fft: (2 * ni * n_boxes).next_power_of_two(),
            n_boxes,
            half,
        }
    }

    /// Box width in embedding units.
    fn box_width(&self) -> f64 {
        2.0 * self.half / self.n_boxes as f64
    }

    /// Whether the grid must be rebuilt: the box count for the new extent
    /// changed, or the embedding came within one box of the edge.
    ///
    /// ### Params
    ///
    /// * `half_span` - Largest absolute coordinate of the embedding
    /// * `ni` - Interpolation nodes per box
    ///
    /// ### Returns
    ///
    /// `true` when the grid must be rebuilt.
    fn needs_rebuild(&self, half_span: f64, ni: usize) -> bool {
        Self::for_extent(half_span, ni).n_boxes != self.n_boxes
            || half_span >= self.half - self.box_width()
    }
}

/// Grid-dependent device state: the geometry scalars and the kernel spectra.
struct Grid {
    /// Geometry the spectra were built for
    geom: GridGeometry,
    /// `[n, nb, nf]`
    gu: Array,
    /// `[coord_min, box_width]`
    gp: Array,
    /// Half spectra of `[q, q^2 dx, q^2 dy]`, complex `[3, nf, nf / 2 + 1]`
    kspec: Array,
}

impl Grid {
    /// Build the spatial kernels on the host and transform them on the
    /// device. Element `(r, c)` of kernel `t` holds the kernel at offset
    /// `((c - m) h, (r - m) h)` for `1 <= r, c <= 2m - 1`, zero elsewhere.
    ///
    /// ### Params
    ///
    /// * `geom` - Grid geometry
    /// * `n` - Number of points
    /// * `ni` - Interpolation nodes per box
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The grid state, spectra evaluated.
    fn new(geom: GridGeometry, n: usize, ni: usize, s: &Stream) -> Result<Self, ManifoldsError> {
        let nf = geom.n_fft;
        let nn = nf * nf;
        let m = geom.n_boxes * ni;
        let h = geom.box_width() / ni as f64;
        let mut k = vec![0.0f32; 3 * nn];
        for r in 1..2 * m {
            for c in 1..2 * m {
                let dx = (c as f64 - m as f64) * h;
                let dy = (r as f64 - m as f64) * h;
                let q = 1.0 / (1.0 + dx * dx + dy * dy);
                let o = r * nf + c;
                k[o] = q as f32;
                k[nn + o] = (q * q * dx) as f32;
                k[2 * nn + o] = (q * q * dy) as f32;
            }
        }
        let nfi = nf as i32;
        let kspec = Array::from_f32(&k, &[3, nfi, nfi]).rfft2([nfi, nfi], s)?;
        eval_all(&[&kspec], false)?;
        Ok(Self {
            geom,
            gu: Array::from_u32(&[n as u32, geom.n_boxes as u32, nf as u32], &[3]),
            gp: Array::from_f32(&[-geom.half as f32, geom.box_width() as f32], &[2]),
            kspec,
        })
    }
}

/////////////
// Kernels //
/////////////

/// The compiled kernels.
struct Kernels {
    /// [`PREP_SOURCE`]
    prep: MetalKernel,
    /// [`BOX_START_SOURCE`]
    box_start: MetalKernel,
    /// [`SPLAT_SOURCE`]
    splat: MetalKernel,
    /// [`GATHER_SOURCE`]
    gather: MetalKernel,
    /// [`UPDATE_SOURCE`]
    update: MetalKernel,
}

impl Kernels {
    /// Register every kernel. MLX compiles each on first use.
    ///
    /// ### Returns
    ///
    /// The kernel set
    fn new() -> Self {
        Self {
            prep: MetalKernel::new(
                "manifolds_tsne_prep",
                &["pos", "gu", "gp"],
                &["box_id", "wx", "wy"],
                LAGRANGE_HEADER,
                PREP_SOURCE,
            ),
            box_start: MetalKernel::new(
                "manifolds_tsne_box_start",
                &["sorted_box", "gu"],
                &["box_start"],
                "",
                BOX_START_SOURCE,
            ),
            splat: MetalKernel::new(
                "manifolds_tsne_splat",
                &["order", "box_start", "wx", "wy", "gu"],
                &["grid"],
                "",
                SPLAT_SOURCE,
            ),
            gather: MetalKernel::new(
                "manifolds_tsne_gather",
                &["conv", "box_id", "wx", "wy", "gu"],
                &["rep", "zc"],
                "",
                GATHER_SOURCE,
            ),
            update: MetalKernel::new(
                "manifolds_tsne_update",
                &[
                    "pos",
                    "vel",
                    "gains",
                    "rep",
                    "z",
                    "indptr",
                    "indices",
                    "values",
                    "reference",
                    "norm",
                    "gu",
                    "fp",
                ],
                &["pos_out", "vel_out", "gains_out"],
                "",
                UPDATE_SOURCE,
            ),
        }
    }
}

/// One-dimensional launch over `total` threads.
///
/// ### Params
///
/// * `total` - Number of threads
///
/// ### Returns
///
/// `(grid, threadgroup)` for [`MetalKernel::apply`]
fn launch_1d(total: usize) -> ([i32; 3], [i32; 3]) {
    ([total as i32, 1, 1], [TSNE_MLX_THREADGROUP, 1, 1])
}

/// Output spec shorthand.
///
/// ### Params
///
/// * `shape` - Output shape
/// * `dtype` - Element type
///
/// ### Returns
///
/// The spec
fn spec(shape: &[i32], dtype: mlx_dtype) -> OutputSpec<'_> {
    OutputSpec { shape, dtype }
}

//////////////////
// Device state //
//////////////////

/// Graph and DREAMS arrays, constant over the run.
struct Constants {
    /// CSR row pointers `[n + 1]`
    indptr: Array,
    /// CSR neighbours `[nnz]`
    indices: Array,
    /// CSR affinities `[nnz]`
    values: Array,
    /// Centred DREAMS reference `[n, 2]`, or a `[2]` stub without DREAMS
    reference: Array,
}

/// The arrays the step rewrites every epoch.
struct StepState {
    /// Positions `[n, 2]`, centred
    pos: Array,
    /// Momentum `[n, 2]`
    vel: Array,
    /// Gains `[n, 2]`
    gains: Array,
}

/// Queue the repulsion for the current positions: bucketing, splat, FFT
/// convolution and gather.
///
/// ### Params
///
/// * `k` - Kernel set
/// * `grid` - Grid state
/// * `pos` - Positions `[n, 2]`
/// * `n` - Number of points
/// * `ni` - Interpolation nodes per box
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) raw repulsion `sum_j q^2 (y_i - y_j)` `[n, 2]` and the scalar
/// `Z = sum_{i != j} q_ij`.
fn repulsion(
    k: &Kernels,
    grid: &Grid,
    pos: &Array,
    n: usize,
    ni: usize,
    s: &Stream,
) -> Result<(Array, Array), ManifoldsError> {
    let nf = grid.geom.n_fft;
    let nb2 = grid.geom.n_boxes * grid.geom.n_boxes;
    let ni_t = [("NI", ni as i32)];
    let (nn_i, nw_i, nb2_i) = (n as i32, (n * ni) as i32, nb2 as i32 + 1);

    // bucketing: box id and weights per point, then box order by sorting
    let (g, tg) = launch_1d(n);
    let mut prep = k
        .prep
        .apply(
            &[pos, &grid.gu, &grid.gp],
            &[
                spec(&[nn_i], MLX_UINT32),
                spec(&[nw_i], MLX_FLOAT32),
                spec(&[nw_i], MLX_FLOAT32),
            ],
            g,
            tg,
            &ni_t,
            s,
        )?
        .into_iter();
    let box_id = prep.next().expect("prep has three outputs");
    let wx = prep.next().expect("prep has three outputs");
    let wy = prep.next().expect("prep has three outputs");
    let order = box_id.argsort_axis(0, s)?;
    let sorted = box_id.take_axis(&order, 0, s)?;
    let (g, tg) = launch_1d(nb2 + 1);
    let box_start = k
        .box_start
        .apply(
            &[&sorted, &grid.gu],
            &[spec(&[nb2_i], MLX_UINT32)],
            g,
            tg,
            &[],
            s,
        )?
        .pop()
        .expect("box_start has one output");

    // splat, convolve, gather
    let nfi = nf as i32;
    let (g, tg) = launch_1d(nf * nf);
    let charges = k
        .splat
        .apply(
            &[&order, &box_start, &wx, &wy, &grid.gu],
            &[spec(&[nfi, nfi], MLX_FLOAT32)],
            g,
            tg,
            &ni_t,
            s,
        )?
        .pop()
        .expect("splat has one output");
    let conv = charges
        .rfft2([nfi, nfi], s)?
        .multiply(&grid.kspec, s)?
        .irfft2([nfi, nfi], s)?;
    let (g, tg) = launch_1d(n);
    let mut gathered = k
        .gather
        .apply(
            &[&conv, &box_id, &wx, &wy, &grid.gu],
            &[spec(&[nn_i, 2], MLX_FLOAT32), spec(&[nn_i], MLX_FLOAT32)],
            g,
            tg,
            &ni_t,
            s,
        )?
        .into_iter();
    let rep = gathered.next().expect("gather has two outputs");
    let z = gathered
        .next()
        .expect("gather has two outputs")
        .sum_all(s)?;
    Ok((rep, z))
}

/// Queue one full epoch: repulsion, the fused step and the recentring.
///
/// ### Params
///
/// * `k` - Kernel set
/// * `grid` - Grid state
/// * `c` - Constant arrays
/// * `st` - State of the previous epoch
/// * `n` - Number of points
/// * `ni` - Interpolation nodes per box
/// * `dreams` - Compile in the DREAMS step
/// * `fp` - Per-epoch parameters, see [`UPDATE_SOURCE`]
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The (lazy) state of the next epoch
#[allow(clippy::too_many_arguments)]
fn epoch(
    k: &Kernels,
    grid: &Grid,
    c: &Constants,
    st: &StepState,
    n: usize,
    ni: usize,
    dreams: bool,
    fp: &Array,
    s: &Stream,
) -> Result<StepState, ManifoldsError> {
    let (rep, z) = repulsion(k, grid, &st.pos, n, ni, s)?;
    let norm = if dreams {
        st.pos.square(s)?.sum_all(s)?
    } else {
        Array::from_f32(&[0.0], &[])
    };

    let nn_i = n as i32;
    let (g, tg) = launch_1d(n);
    let mut out = k
        .update
        .apply(
            &[
                &st.pos,
                &st.vel,
                &st.gains,
                &rep,
                &z,
                &c.indptr,
                &c.indices,
                &c.values,
                &c.reference,
                &norm,
                &grid.gu,
                fp,
            ],
            &[
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[nn_i, 2], MLX_FLOAT32),
                spec(&[nn_i, 2], MLX_FLOAT32),
            ],
            g,
            tg,
            &[("DREAMS", dreams as i32)],
            s,
        )?
        .into_iter();
    let pos = out.next().expect("update has three outputs");
    let vel = out.next().expect("update has three outputs");
    let gains = out.next().expect("update has three outputs");
    let pos = pos.subtract(&pos.mean_axis_keep(0, s)?, s)?;
    Ok(StepState { pos, vel, gains })
}

//////////
// Main //
//////////

/// Optimise a 2D embedding with three-kernel FFT t-SNE on MLX.
///
/// Same schedule and update rule as `optimise_fft3k_tsne_gpu` (momentum
/// switch, early / late exaggeration, Jacobs gains, step cap, recentring,
/// optional DREAMS step), one extent readback every `GRID_CHECK_INTERVAL`
/// epochs, and the embedding read back once at the end. Computes in `f32`
/// whatever `T` is.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][2]`, overwritten with the result
/// * `params` - Optimisation hyperparameters
/// * `graph` - Symmetric t-SNE affinities
/// * `dreams` - DREAMS regulariser state, or `None` for plain t-SNE
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `Ok(())` on success; `NoData` if the embedding is empty, `IncorrectDim` if
/// it is not 2D, `GraphSizeMismatch` if the graph covers a different number
/// of points, `MlxError` if an mlx-c call fails.
///
/// ### References
///
/// Linderman et al., Nature Methods, 2019 (FIt-SNE).
pub fn optimise_fft3k_tsne_mlx<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dreams: Option<&DreamsState<T>>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let verbosity = parse_verbosity_level(verbose);
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let n_dim = embd[0].len();
    if n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim { n_dim });
    }
    if graph.n_samples != n {
        return Err(ManifoldsError::GraphSizeMismatch {
            n_graph: graph.n_samples,
            n_embd: n,
        });
    }
    let ni = params.n_interp_points;
    let f = |x: T| x.to_f32().unwrap();

    // CSR, with a dummy entry so an edgeless graph still binds
    let adj = coo_to_adjacency_list(graph);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in &adj {
        for &(j, w) in row {
            indices.push(j as u32);
            values.push(f(w));
        }
        indptr.push(indices.len() as u32);
    }
    if indices.is_empty() {
        indices.push(0);
        values.push(0.0);
    }

    let pos: Vec<f32> = embd.iter().flat_map(|p| [f(p[0]), f(p[1])]).collect();
    let half_span = pos.iter().fold(0.0f64, |a, &v| a.max(v.abs() as f64));
    let reference: Vec<f32> = match dreams {
        Some(d) => d.reference().iter().map(|&x| f(x)).collect(),
        None => vec![0.0; 2],
    };

    install_error_handler();
    let s = Stream::default_gpu();
    let k = Kernels::new();
    let nn_i = n as i32;
    let c = Constants {
        indptr: Array::from_u32(&indptr, &[nn_i + 1]),
        indices: Array::from_u32(&indices, &[indices.len() as i32]),
        values: Array::from_f32(&values, &[values.len() as i32]),
        reference: Array::from_f32(&reference, &[reference.len() as i32]),
    };
    let mut st = StepState {
        pos: Array::from_f32(&pos, &[nn_i, 2]),
        vel: Array::from_f32(&vec![0.0; 2 * n], &[nn_i, 2]),
        gains: Array::from_f32(&vec![1.0; 2 * n], &[nn_i, 2]),
    };
    let mut grid = Grid::new(GridGeometry::for_extent(half_span, ni), n, ni, &s)?;

    let lr = params.get_lr(n);
    let mut fp = [0.0f32; N_STEP_PARAMS];
    fp[1] = f(lr);
    fp[3] = TSNE_MIN_GAIN as f32;
    fp[4] = f(step_cap_from_lr(lr));
    fp[5] = TSNE_EPS as f32;
    fp[6] = dreams.map_or(1.0, |d| 1.0 / d.ref_norm()) as f32;
    fp[7] = dreams.map_or(1.0, |d| f(d.keep()));

    if verbosity.normal_verbosity() {
        println!(
            "Running {} epochs of three-kernel FFT t-SNE on MLX.",
            params.n_epochs
        );
    }

    let mut last_span = half_span;
    for ep in 0..params.n_epochs {
        if ep > 0 && ep % GRID_CHECK_INTERVAL == 0 {
            let ext = st.pos.abs(&s)?.max_all(&s)?;
            eval_all(&[&ext], false)?;
            let span = ext.as_f32()?[0] as f64;
            let ahead = span + GRID_GROWTH_SAFETY * (span - last_span).max(0.0);
            last_span = span;
            if grid.geom.needs_rebuild(ahead, ni) {
                grid = Grid::new(GridGeometry::for_extent(ahead, ni), n, ni, &s)?;
            }
            if verbosity.normal_verbosity() && ep % 50 == 0 {
                println!(
                    " Epoch {}/{} | extent = {:.1} | n_boxes = {}",
                    ep, params.n_epochs, span, grid.geom.n_boxes
                );
            }
        }

        let exag = if ep < params.early_exag_iter {
            params.early_exag_factor
        } else {
            params.get_late_exag_factor()
        };
        fp[0] = f(exag);
        fp[2] = if ep < TSNE_MOMENTUM_SWITCH_ITER {
            TSNE_INITIAL_MOMENTUM
        } else {
            TSNE_FINAL_MOMENTUM
        } as f32;
        fp[8] = dreams.map_or(0.0, |d| f(d.step_coef(exag)));
        let fp_arr = Array::from_f32(&fp, &[N_STEP_PARAMS as i32]);

        let next = epoch(&k, &grid, &c, &st, n, ni, dreams.is_some(), &fp_arr, &s)?;
        // stay at most one epoch ahead of the device
        eval_all(&[&next.pos, &next.vel, &next.gains], true)?;
        eval_all(&[&st.pos], false)?;
        st = next;
    }

    eval_all(&[&st.pos], false)?;
    let out = st.pos.as_f32()?;
    for (i, p) in embd.iter_mut().enumerate() {
        p[0] = T::from(out[2 * i]).unwrap();
        p[1] = T::from(out[2 * i + 1]).unwrap();
    }
    drop(st);
    drop(grid);
    clear_cache()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten clusters in a square of half-width `radius`, recentred, from a
    /// fixed LCG. Same fixture as the cubecl tests.
    fn clustered(n: usize, radius: f64) -> Vec<f32> {
        let mut state = 7u64;
        let mut unif = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let centres: Vec<(f64, f64)> = (0..10)
            .map(|_| ((unif() * 2.0 - 1.0) * radius, (unif() * 2.0 - 1.0) * radius))
            .collect();
        let mut pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let (cx, cy) = centres[i % 10];
                (cx + (unif() - 0.5) * 4.0, cy + (unif() - 0.5) * 4.0)
            })
            .collect();
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n as f64;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n as f64;
        pts.iter_mut().for_each(|p| {
            p.0 -= mx;
            p.1 -= my;
        });
        pts.iter()
            .flat_map(|&(x, y)| [x as f32, y as f32])
            .collect()
    }

    fn empty_graph(n: usize) -> CoordinateList<f32> {
        CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: n,
        }
    }

    #[test]
    fn test_mlx_repulsion_matches_direct() {
        let n = 2_000;
        let pos = clustered(n, 30.0);
        let half_span = pos.iter().fold(0.0f64, |a, &v| a.max(v.abs() as f64));
        let s = Stream::default_gpu();
        let k = Kernels::new();
        let grid = Grid::new(GridGeometry::for_extent(half_span, 3), n, 3, &s).unwrap();
        let (rep, z) =
            repulsion(&k, &grid, &Array::from_f32(&pos, &[n as i32, 2]), n, 3, &s).unwrap();
        eval_all(&[&rep, &z], false).unwrap();
        let rep = rep.as_f32().unwrap();
        let z = z.as_f32().unwrap()[0] as f64;

        let (mut z_ex, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        let mut f_ex = vec![(0.0f64, 0.0f64); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = (pos[2 * i] - pos[2 * j]) as f64;
                    let dy = (pos[2 * i + 1] - pos[2 * j + 1]) as f64;
                    let q = 1.0 / (1.0 + dx * dx + dy * dy);
                    z_ex += q;
                    f_ex[i].0 += q * q * dx;
                    f_ex[i].1 += q * q * dy;
                }
            }
        }
        for i in 0..n {
            let (rx, ry) = (f_ex[i].0 / z_ex, f_ex[i].1 / z_ex);
            let (gx, gy) = (rep[2 * i] as f64 / z, rep[2 * i + 1] as f64 / z);
            num += (gx - rx).powi(2) + (gy - ry).powi(2);
            den += rx * rx + ry * ry;
        }
        let z_err = (z - z_ex).abs() / z_ex;
        let f_err = (num / den).sqrt();
        assert!(z_err < 1e-2, "Z relative error {z_err:.3e}");
        assert!(f_err < 0.05, "force relative RMS {f_err:.3e}");
    }

    #[test]
    fn test_mlx_optimiser_rejects_bad_input() {
        let params = TsneOptimParams::<f32>::default();
        let mut empty: Vec<Vec<f32>> = vec![];
        let res = optimise_fft3k_tsne_mlx(&mut empty, &params, &empty_graph(0), None, 0);
        assert!(matches!(res, Err(ManifoldsError::NoData)));

        let mut three_d = vec![vec![0.0f32; 3]; 4];
        let res = optimise_fft3k_tsne_mlx(&mut three_d, &params, &empty_graph(4), None, 0);
        assert!(matches!(
            res,
            Err(ManifoldsError::IncorrectDim { n_dim: 3 })
        ));

        let mut two_d = vec![vec![0.0f32; 2]; 4];
        let res = optimise_fft3k_tsne_mlx(&mut two_d, &params, &empty_graph(5), None, 0);
        assert!(matches!(
            res,
            Err(ManifoldsError::GraphSizeMismatch {
                n_graph: 5,
                n_embd: 4
            })
        ));
    }
}

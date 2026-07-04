//! FFT implementation for the acceleration tSNE form

use fftw::array::AlignedVec;
use fftw::plan::{C2RPlan, C2RPlan32, C2RPlan64, R2CPlan, R2CPlan32, R2CPlan64};
use fftw::types::{c32, c64, Flag};
use num_traits::{Float, FromPrimitive, Signed, ToPrimitive};
use rayon::prelude::*;
use std::fmt::Debug;

/// FFTW planning rigour. `Flag::MEASURE` plans slower but can execute faster;
/// grid sizes persist for many epochs, so it may amortise. Benchmark to decide.
const PLAN_FLAG: Flag = Flag::ESTIMATE;

////////////
// Traits //
////////////

/// Trait defining floating-point types compatible with FFTW operations.
///
/// Provides abstractions for f32/f64 with their corresponding complex types,
/// FFT plans, and operations. Enables generic FFT-based algorithms.
pub trait FftwFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Debug + Signed + 'static
{
    /// Complex number type (c32 for f32, c64 for f64)
    type Complex: Copy + Send + Sync;

    /// Real-to-complex FFT plan type
    type R2CPlan: Send;

    /// Complex-to-real FFT plan type
    type C2RPlan: Send;

    /// Construct complex number from real and imaginary parts
    fn new_complex(re: Self, im: Self) -> Self::Complex;

    /// Zero complex value
    fn complex_zero() -> Self::Complex;

    /// Extract real component from complex number
    fn complex_re(c: Self::Complex) -> Self;

    /// Extract imaginary component from complex number
    fn complex_im(c: Self::Complex) -> Self;

    /// Allocate FFTW-aligned array of real values
    fn aligned_real(size: usize) -> AlignedVec<Self>;

    /// Allocate FFTW-aligned array of complex values
    fn aligned_complex(size: usize) -> AlignedVec<Self::Complex>;

    /// Create 2D real-to-complex FFT plan for square array of dimension n
    fn plan_r2c_2d(n: usize) -> Self::R2CPlan;

    /// Create 2D complex-to-real FFT plan for square array of dimension n
    fn plan_c2r_2d(n: usize) -> Self::C2RPlan;

    /// Execute real-to-complex FFT transform
    fn execute_r2c(
        plan: &mut Self::R2CPlan,
        input: &mut AlignedVec<Self>,
        output: &mut AlignedVec<Self::Complex>,
    );

    /// Execute complex-to-real inverse FFT transform
    fn execute_c2r(
        plan: &mut Self::C2RPlan,
        input: &mut AlignedVec<Self::Complex>,
        output: &mut AlignedVec<Self>,
    );
}

/////////
// f64 //
/////////

impl FftwFloat for f64 {
    type Complex = c64;
    type R2CPlan = R2CPlan64;
    type C2RPlan = C2RPlan64;

    #[inline]
    fn new_complex(re: Self, im: Self) -> Self::Complex {
        c64::new(re, im)
    }
    #[inline]
    fn complex_zero() -> Self::Complex {
        c64::new(0.0, 0.0)
    }
    #[inline]
    fn complex_re(c: Self::Complex) -> Self {
        c.re
    }
    #[inline]
    fn complex_im(c: Self::Complex) -> Self {
        c.im
    }
    fn aligned_real(size: usize) -> AlignedVec<Self> {
        AlignedVec::new(size)
    }
    fn aligned_complex(size: usize) -> AlignedVec<Self::Complex> {
        AlignedVec::new(size)
    }
    fn plan_r2c_2d(n: usize) -> Self::R2CPlan {
        R2CPlan::aligned(&[n, n], PLAN_FLAG).expect("Failed to create R2C plan")
    }
    fn plan_c2r_2d(n: usize) -> Self::C2RPlan {
        C2RPlan::aligned(&[n, n], PLAN_FLAG).expect("Failed to create C2R plan")
    }
    fn execute_r2c(
        plan: &mut Self::R2CPlan,
        input: &mut AlignedVec<Self>,
        output: &mut AlignedVec<Self::Complex>,
    ) {
        plan.r2c(input, output).expect("R2C FFT failed");
    }
    fn execute_c2r(
        plan: &mut Self::C2RPlan,
        input: &mut AlignedVec<Self::Complex>,
        output: &mut AlignedVec<Self>,
    ) {
        plan.c2r(input, output).expect("C2R IFFT failed");
    }
}

/////////
// f32 //
/////////

impl FftwFloat for f32 {
    type Complex = c32;
    type R2CPlan = R2CPlan32;
    type C2RPlan = C2RPlan32;

    #[inline]
    fn new_complex(re: Self, im: Self) -> Self::Complex {
        c32::new(re, im)
    }
    #[inline]
    fn complex_zero() -> Self::Complex {
        c32::new(0.0, 0.0)
    }
    #[inline]
    fn complex_re(c: Self::Complex) -> Self {
        c.re
    }
    #[inline]
    fn complex_im(c: Self::Complex) -> Self {
        c.im
    }
    fn aligned_real(size: usize) -> AlignedVec<Self> {
        AlignedVec::new(size)
    }
    fn aligned_complex(size: usize) -> AlignedVec<Self::Complex> {
        AlignedVec::new(size)
    }
    fn plan_r2c_2d(n: usize) -> Self::R2CPlan {
        R2CPlan::aligned(&[n, n], PLAN_FLAG).expect("Failed to create R2C plan")
    }
    fn plan_c2r_2d(n: usize) -> Self::C2RPlan {
        C2RPlan::aligned(&[n, n], PLAN_FLAG).expect("Failed to create C2R plan")
    }
    fn execute_r2c(
        plan: &mut Self::R2CPlan,
        input: &mut AlignedVec<Self>,
        output: &mut AlignedVec<Self::Complex>,
    ) {
        plan.r2c(input, output).expect("R2C FFT failed");
    }
    fn execute_c2r(
        plan: &mut Self::C2RPlan,
        input: &mut AlignedVec<Self::Complex>,
        output: &mut AlignedVec<Self>,
    ) {
        plan.c2r(input, output).expect("C2R IFFT failed");
    }
}

//////////////////
// FftWorkspace //
//////////////////

/// Per-term FFTW plans and aligned buffers.
///
/// Each expansion term owns its own plans and buffers so the `n_terms`
/// convolutions run in parallel (the `fftw` crate exposes no threaded or
/// batched plans, so parallelism across independent transforms is the
/// available axis).
pub struct TermSlot<T: FftwFloat> {
    /// Aligned real input buffer, dimensions `n_fft x n_fft`. Zeroed once at
    /// creation; only the top-left quadrant is rewritten per call (the
    /// out-of-place R2C transform preserves its input, so the zero padding
    /// survives across calls).
    pub fft_input: AlignedVec<T>,
    /// Aligned complex buffer, dimensions `n_fft x (n_fft/2 + 1)`
    pub fft_output: AlignedVec<T::Complex>,
    /// Aligned real C2R output, dimensions `n_fft x n_fft`. Holds the final
    /// convolved (and kernel-normalised) grid the gather step reads.
    pub fft_scratch: AlignedVec<T>,
    /// Pre-built real-to-complex FFT plan
    pub plan_r2c: T::R2CPlan,
    /// Pre-built complex-to-real FFT plan
    pub plan_c2r: T::C2RPlan,
}

impl<T: FftwFloat> TermSlot<T> {
    /// Create one term's plans and buffers for a given FFT dimension.
    fn new(n_fft: usize) -> Self {
        let mut fft_input = T::aligned_real(n_fft * n_fft);
        for v in fft_input.iter_mut() {
            *v = T::zero();
        }
        Self {
            fft_input,
            fft_output: T::aligned_complex(n_fft * (n_fft / 2 + 1)),
            fft_scratch: T::aligned_real(n_fft * n_fft),
            plan_r2c: T::plan_r2c_2d(n_fft),
            plan_c2r: T::plan_c2r_2d(n_fft),
        }
    }
}

/// Reusable FFTW plans and aligned buffers for FFT convolution.
///
/// Creating FFTW plans is expensive (even with `ESTIMATE`), and aligned
/// memory allocation has non-trivial overhead. This workspace holds both,
/// allowing them to persist across calls to `n_body_fft_2d` and across
/// optimisation epochs. Rebuild only when `n_fft` changes, i.e. when the
/// grid resizes.
pub struct FftWorkspace<T: FftwFloat> {
    /// FFT array dimension this workspace was built for
    pub n_fft: usize,
    /// One plan/buffer set per expansion term, grown on first use
    pub slots: Vec<TermSlot<T>>,
    /// Per-point box index and intra-box position, length n_points.
    pub box_data: Vec<((usize, usize), T, T)>,
    /// Per-point Lagrange x-weights, length n_points * n_interp_points.
    pub x_weights: Vec<T>,
    /// Per-point Lagrange y-weights, length n_points * n_interp_points.
    pub y_weights: Vec<T>,
    /// Grid charge coefficients, length total_grid_points * n_terms.
    pub w_coefficients: Vec<T>,
    /// Convolved grid values, length total_grid_points * n_terms. Compact
    /// and term-minor so the gather reads all terms of a node from one
    /// cache line, rather than striding across the fat per-term FFT buffers.
    pub y_tilde_values: Vec<T>,
    /// Counting-sort bucket starts over box rows, length n_boxes + 1.
    pub row_starts: Vec<u32>,
    /// Point indices grouped by box row, length n_points.
    pub order: Vec<u32>,
}

impl<T: FftwFloat> FftWorkspace<T> {
    /// Create workspace for a given FFT dimension.
    ///
    /// Per-term plans and aligned buffers are created lazily on the first
    /// `n_body_fft_2d` call, once `n_terms` is known.
    ///
    /// ### Params
    ///
    /// * `n_fft` - FFT array dimension (typically `2 * n_interp_points *
    ///   n_boxes_per_dim`)
    ///
    /// ### Returns
    ///
    /// Workspace ready for use with `n_body_fft_2d`
    pub fn new(n_fft: usize) -> Self {
        Self {
            n_fft,
            slots: Vec::new(),
            box_data: Vec::new(),
            x_weights: Vec::new(),
            y_weights: Vec::new(),
            w_coefficients: Vec::new(),
            y_tilde_values: Vec::new(),
            row_starts: Vec::new(),
            order: Vec::new(),
        }
    }
}

/////////////
// FftGrid //
/////////////

/// Grid structure for FFT-based interpolation.
///
/// Manages 2D uniform grid discretisation and pre-computed FFT kernel for
/// efficient potential field evaluation via convolution. The grid divides
/// coordinate space into boxes, each containing interpolation nodes for
/// Lagrange polynomial interpolation.
pub struct FftGrid<T: FftwFloat> {
    /// Number of boxes per dimension
    pub n_boxes_per_dim: usize,
    /// Number of interpolation nodes per box dimension (typically 3)
    pub n_interp_points: usize,
    /// Width of each box in coordinate space
    pub box_width: T,
    /// Minimum coordinate value (same for x and y; assumes square domain)
    pub coord_min: T,
    /// Interpolation node positions within `[0,1]` normalised box coordinates
    pub interp_spacings: Vec<T>,
    /// Pre-computed Lagrange polynomial denominators for each node
    pub lagrange_denominators: Vec<T>,
    /// Global grid node x-coordinates (length
    /// `n_interp_points * n_boxes_per_dim`)
    pub global_x_coords: Vec<T>,
    /// `global_y_coords` - Global grid node y-coordinates (length
    /// `n_interp_points * n_boxes_per_dim`)
    pub global_y_coords: Vec<T>,
    /// Pre-computed FFT of convolution kernel; dimensions
    /// `n_fft x (n_fft/2 + 1)` for R2C transform
    pub fft_kernel: AlignedVec<T::Complex>,
    /// FFT array dimension: `2 * n_interp_points * n_boxes_per_dim`
    pub n_fft: usize,
}

impl<T: FftwFloat> FftGrid<T> {
    /// Create new FFT grid structure.
    ///
    /// Initialises uniform 2D grid covering square domain
    /// `[coord_min, coord_max]^2`, computes Lagrange interpolation coefficients,
    /// and pre-computes FFT of the kernel `K(r) = 1/(1 + r^2)^2`.
    ///
    /// ### Params
    ///
    /// * `coord_min` - Minimum coordinate (applies to both x and y)
    /// * `coord_max` - Maximum coordinate (applies to both x and y)
    /// * `n_boxes_per_dim` - Number of boxes per dimension
    /// * `n_interp_points` - Number of interpolation nodes per box per
    ///   dimension
    ///
    /// ### Returns
    ///
    /// Initialised grid structure ready for FFT-accelerated convolution
    pub fn new(coord_min: T, coord_max: T, n_boxes_per_dim: usize, n_interp_points: usize) -> Self {
        let box_width = (coord_max - coord_min) / T::from_usize(n_boxes_per_dim).unwrap();

        let h = T::one() / T::from_usize(n_interp_points).unwrap();
        let two = T::one() + T::one();

        let mut interp_spacings = vec![T::zero(); n_interp_points];
        interp_spacings[0] = h / two;
        for i in 1..n_interp_points {
            interp_spacings[i] = interp_spacings[i - 1] + h;
        }

        let mut lagrange_denominators = vec![T::one(); n_interp_points];
        for i in 0..n_interp_points {
            for j in 0..n_interp_points {
                if i != j {
                    lagrange_denominators[i] =
                        lagrange_denominators[i] * (interp_spacings[i] - interp_spacings[j]);
                }
            }
        }

        let n_interp_1d = n_interp_points * n_boxes_per_dim;
        let global_h = h * box_width;

        let mut global_x_coords = vec![T::zero(); n_interp_1d];
        let mut global_y_coords = vec![T::zero(); n_interp_1d];
        global_x_coords[0] = coord_min + global_h / two;
        global_y_coords[0] = coord_min + global_h / two;

        for i in 1..n_interp_1d {
            global_x_coords[i] = global_x_coords[i - 1] + global_h;
            global_y_coords[i] = global_y_coords[i - 1] + global_h;
        }

        let n_fft = 2 * n_interp_1d;
        let fft_kernel = Self::precompute_kernel(&global_x_coords, &global_y_coords, n_fft);

        Self {
            n_boxes_per_dim,
            n_interp_points,
            box_width,
            coord_min,
            interp_spacings,
            lagrange_denominators,
            global_x_coords,
            global_y_coords,
            fft_kernel,
            n_fft,
        }
    }

    /// Pre-compute FFT of convolution kernel.
    ///
    /// Constructs spatial kernel `K(r) = 1/(1 + r^2)^2` on zero-padded grid,
    /// exploiting circular symmetry by populating four quadrants. Transforms
    /// to frequency domain via R2C FFT for efficient convolution.
    ///
    /// ### Params
    ///
    /// * `x_coords` - Grid node x-coordinates
    /// * `y_coords` - Grid node y-coordinates
    /// * `n_fft` - FFT array dimension (typically `2 * len(x_coords)`)
    ///
    /// ### Returns
    ///
    /// Complex-valued FFT of kernel, dimensions `n_fft x (n_fft/2 + 1)`,
    /// pre-scaled by `1 / n_fft^2` so the inverse transform needs no
    /// separate normalisation
    fn precompute_kernel(x_coords: &[T], y_coords: &[T], n_fft: usize) -> AlignedVec<T::Complex> {
        let n_interp = x_coords.len();
        let x_0 = x_coords[0];
        let y_0 = y_coords[0];

        let mut kernel_real: AlignedVec<T> = T::aligned_real(n_fft * n_fft);
        for x in kernel_real.iter_mut() {
            *x = T::zero();
        }

        for i in 0..n_interp {
            for j in 0..n_interp {
                let dx = x_coords[i] - x_0;
                let dy = y_coords[j] - y_0;
                let dist_sq = dx * dx + dy * dy;
                let denom = T::one() + dist_sq;
                let k_val = T::one() / (denom * denom);

                kernel_real[(n_interp + i) * n_fft + (n_interp + j)] = k_val;
                if i > 0 {
                    kernel_real[(n_interp - i) * n_fft + (n_interp + j)] = k_val;
                }
                if j > 0 {
                    kernel_real[(n_interp + i) * n_fft + (n_interp - j)] = k_val;
                }
                if i > 0 && j > 0 {
                    kernel_real[(n_interp - i) * n_fft + (n_interp - j)] = k_val;
                }
            }
        }

        let n_complex = n_fft * (n_fft / 2 + 1);
        let mut kernel_fft: AlignedVec<T::Complex> = T::aligned_complex(n_complex);
        let mut plan = T::plan_r2c_2d(n_fft);
        T::execute_r2c(&mut plan, &mut kernel_real, &mut kernel_fft);

        // Fold the 1 / n_fft^2 inverse-transform normalisation into the
        // kernel once, so the convolved grids come out of the C2R already
        // normalised.
        let norm = T::one() / T::from_usize(n_fft * n_fft).unwrap();
        for k in kernel_fft.iter_mut() {
            *k = T::new_complex(T::complex_re(*k) * norm, T::complex_im(*k) * norm);
        }

        kernel_fft
    }

    /// Map point coordinates to containing box indices.
    ///
    /// ### Params
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    ///
    /// ### Returns
    ///
    /// Tuple `(y_box_index, x_box_index)` clamped to `[0, n_boxes_per_dim)`
    #[inline]
    pub fn point_to_box(&self, x: T, y: T) -> (usize, usize) {
        let x_idx = ((x - self.coord_min) / self.box_width)
            .to_usize()
            .unwrap_or(0)
            .min(self.n_boxes_per_dim - 1);
        let y_idx = ((y - self.coord_min) / self.box_width)
            .to_usize()
            .unwrap_or(0)
            .min(self.n_boxes_per_dim - 1);
        (y_idx, x_idx)
    }

    /// Compute normalised position within box.
    ///
    /// ### Params
    ///
    /// * `coord` - Coordinate value (x or y)
    /// * `box_idx` - Box index along corresponding dimension
    ///
    /// ### Returns
    ///
    /// Position in `[0, 1]` normalised box coordinates
    #[inline]
    pub fn position_in_box(&self, coord: T, box_idx: usize) -> T {
        let box_min = self.coord_min + T::from_usize(box_idx).unwrap() * self.box_width;
        (coord - box_min) / self.box_width
    }

    /// Check if all points lie within grid interior.
    ///
    /// ### Params
    ///
    /// * `xs` - X coordinates of points
    /// * `ys` - Y coordinates of points
    /// * `margin` - Inset distance from grid boundaries
    ///
    /// ### Returns
    ///
    /// `true` if all points satisfy
    /// `coord_min + margin <= {x,y} <= coord_max - margin`
    pub fn contains_points(&self, xs: &[T], ys: &[T], margin: T) -> bool {
        let coord_max =
            self.coord_min + self.box_width * T::from_usize(self.n_boxes_per_dim).unwrap();
        let inner_min = self.coord_min + margin;
        let inner_max = coord_max - margin;

        for (&x, &y) in xs.iter().zip(ys) {
            if x < inner_min || x > inner_max || y < inner_min || y > inner_max {
                return false;
            }
        }
        true
    }

    /// Compute axis-aligned bounding box of embedding with padding.
    ///
    /// ### Params
    ///
    /// * `embd` - Point coordinates, shape `[n_points][2]`
    /// * `padding_fraction` - Fractional padding relative to data spread
    ///
    /// ### Returns
    ///
    /// Tuple `(min_coord, max_coord)` defining square bounding box
    pub fn compute_bounds(embd: &[Vec<T>], padding_fraction: T) -> (T, T) {
        let mut min_val = embd[0][0];
        let mut max_val = embd[0][0];

        for p in embd {
            min_val = min_val.min(p[0]).min(p[1]);
            max_val = max_val.max(p[0]).max(p[1]);
        }

        let spread = max_val - min_val;
        let padding = spread * padding_fraction;
        (min_val - padding, max_val + padding)
    }

    /// Create grid from embedding data with automatic size selection.
    ///
    /// Computes bounding box with 10% padding, determines grid size via
    /// `choose_grid_size` heuristic, then initialises grid structure.
    ///
    /// ### Params
    ///
    /// * `embd` - Point coordinates, shape `[n_points][2]`
    /// * `n_interp_points` - Interpolation nodes per box per dimension
    /// * `intervals_per_integer` - Target box width (typically 1.0)
    /// * `min_intervals` - Minimum number of boxes per dimension (typically 50)
    ///
    /// ### Returns
    ///
    /// Initialised grid structure sized appropriately for data
    pub fn from_embedding(
        embd: &[Vec<T>],
        n_interp_points: usize,
        intervals_per_integer: f64,
        min_intervals: usize,
    ) -> Self {
        let padding = T::from_f64(0.1).unwrap();
        let (coord_min, coord_max) = Self::compute_bounds(embd, padding);
        let n_boxes = choose_grid_size(
            coord_min.to_f64().unwrap(),
            coord_max.to_f64().unwrap(),
            intervals_per_integer,
            min_intervals,
        );
        Self::new(coord_min, coord_max, n_boxes, n_interp_points)
    }
}

/// Compute Lagrange interpolation weights.
///
/// For a point at position `y_in_box` (in `[0, 1]`), computes the weights for all
/// interpolation nodes using Lagrange polynomials.
///
/// ### Params
///
/// * `y_in_box` - Position within box, range `[0, 1]`
/// * `spacings` - Interpolation node positions in `[0, 1]`
/// * `denominators` - Pre-computed denominators: product of
///   (`spacings[i] - spacings[j]`) for j != i
/// * `weights` - Output buffer for weights
#[inline]
pub fn lagrange_weights<T: FftwFloat>(
    y_in_box: T,
    spacings: &[T],
    denominators: &[T],
    weights: &mut [T],
) {
    let n = spacings.len();
    for i in 0..n {
        let mut numerator = T::one();
        for j in 0..n {
            if i != j {
                numerator = numerator * (y_in_box - spacings[j]);
            }
        }
        weights[i] = numerator / denominators[i];
    }
}

/// Determine optimal grid size based on data spread.
///
/// Follows FIt-SNE's heuristic: number of boxes scales with spread,
/// then rounds up to FFT-friendly sizes. Allowed sizes match the C++
/// reference implementation.
///
/// ### Params
///
/// * `coord_min` - Minimum coordinate
/// * `coord_max` - Maximum coordinate
/// * `intervals_per_integer` - Target interval size (typically 1.0)
/// * `min_intervals` - Minimum number of intervals (typically 50)
///
/// ### Returns
///
/// Number of boxes per dimension
pub fn choose_grid_size(
    coord_min: f64,
    coord_max: f64,
    intervals_per_integer: f64,
    min_intervals: usize,
) -> usize {
    let spread = coord_max - coord_min;
    let n_boxes = ((spread / intervals_per_integer).max(min_intervals as f64)) as usize;

    const ALLOWED_SIZES: [usize; 20] = [
        25, 36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140, 150, 175, 200,
    ];

    if n_boxes < ALLOWED_SIZES[19] {
        for &size in &ALLOWED_SIZES {
            if size >= n_boxes {
                return size;
            }
        }
    }
    n_boxes
}

/// Compute repulsive forces/potentials using FFT acceleration.
///
/// Generic implementation supporting f32/f64 via the `FftwFloat` trait.
/// Accepts a pre-allocated workspace to avoid per-call FFTW plan creation
/// and aligned buffer allocation. The caller must also provide the output
/// buffer, which is overwritten.
///
/// The three stages are parallel: charge spreading over box rows (points in
/// different box rows write disjoint grid slabs), the `n_terms` FFT
/// convolutions over per-term plan/buffer slots, and gathering over points
/// (reading each term's convolved grid directly).
///
/// ### Params
///
/// * `xs` - X-coordinates of the points
/// * `ys` - Y-coordinates of the points
/// * `charges` - Charges of the points, shape `[n_points * n_terms]`
/// * `n_terms` - Number of terms in the expansion
/// * `grid` - FFT grid (pre-computed kernel and interpolation data)
/// * `ws` - Reusable FFT workspace (plans and aligned buffers); must have
///   been created with the same `n_fft` as the grid
/// * `potentials_out` - Output buffer, shape `[n_points * n_terms]`; will
///   be overwritten with computed potentials
pub fn n_body_fft_2d<T: FftwFloat>(
    xs: &[T],
    ys: &[T],
    charges: &[T],
    n_terms: usize,
    grid: &FftGrid<T>,
    ws: &mut FftWorkspace<T>,
    potentials_out: &mut [T],
) {
    let n_points = xs.len();
    let n_interp = grid.n_interp_points;
    let n_boxes = grid.n_boxes_per_dim;
    let n_interp_1d = n_interp * n_boxes;
    let total_grid_points = n_interp_1d * n_interp_1d;
    let n_fft = ws.n_fft;
    let n_complex = n_fft * (n_fft / 2 + 1);

    assert_eq!(ws.n_fft, grid.n_fft, "Workspace n_fft does not match grid");
    assert_eq!(
        potentials_out.len(),
        n_points * n_terms,
        "Output buffer size mismatch"
    );

    // Grow scratch buffers if their sizes don't match. In steady state these
    // are all no-ops (sizes are constant once n_boxes stabilises).
    let weights_len = n_points * n_interp;
    let grid_buf_len = total_grid_points * n_terms;
    if ws.box_data.len() != n_points {
        ws.box_data.resize(n_points, ((0, 0), T::zero(), T::zero()));
        ws.order.resize(n_points, 0);
    }
    if ws.x_weights.len() != weights_len {
        ws.x_weights.resize(weights_len, T::zero());
        ws.y_weights.resize(weights_len, T::zero());
    }
    if ws.w_coefficients.len() != grid_buf_len {
        ws.w_coefficients.resize(grid_buf_len, T::zero());
        ws.y_tilde_values.resize(grid_buf_len, T::zero());
    }
    while ws.slots.len() < n_terms {
        ws.slots.push(TermSlot::new(n_fft));
    }

    // step 1: box assignment and relative coords, written in place
    ws.box_data
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            let (box_y, box_x) = grid.point_to_box(xs[i], ys[i]);
            let x_in_box = grid.position_in_box(xs[i], box_x);
            let y_in_box = grid.position_in_box(ys[i], box_y);
            *slot = ((box_y, box_x), x_in_box, y_in_box);
        });

    // step 1b: Lagrange interpolation weights into flat per-axis buffers
    {
        let box_data = &ws.box_data;
        let interp_spacings = &grid.interp_spacings;
        let lagrange_denominators = &grid.lagrange_denominators;
        ws.x_weights
            .par_chunks_mut(n_interp)
            .zip(ws.y_weights.par_chunks_mut(n_interp))
            .enumerate()
            .for_each(|(i, (x_w, y_w))| {
                let (_, x_pos, y_pos) = box_data[i];
                lagrange_weights(x_pos, interp_spacings, lagrange_denominators, x_w);
                lagrange_weights(y_pos, interp_spacings, lagrange_denominators, y_w);
            });
    }

    // step 1c: counting sort of point indices by box row, so spreading can
    // run one thread per box row without write conflicts.
    ws.row_starts.clear();
    ws.row_starts.resize(n_boxes + 1, 0);
    for &((box_y, _), _, _) in ws.box_data.iter() {
        ws.row_starts[box_y + 1] += 1;
    }
    for b in 0..n_boxes {
        ws.row_starts[b + 1] += ws.row_starts[b];
    }
    let mut cursor = ws.row_starts.clone();
    for i in 0..n_points {
        let ((box_y, _), _, _) = ws.box_data[i];
        ws.order[cursor[box_y] as usize] = i as u32;
        cursor[box_y] += 1;
    }

    // step 1d: charge spreading, parallel over box rows. A point writes only
    // to its own box's n_interp x n_interp nodes, and a box row `by` owns the
    // contiguous w_coefficients slab of grid rows [by * n_interp,
    // (by + 1) * n_interp), so slabs are disjoint across threads.
    {
        let box_data = &ws.box_data;
        let x_weights = &ws.x_weights;
        let y_weights = &ws.y_weights;
        let order = &ws.order;
        let row_starts = &ws.row_starts;
        let slab_len = n_interp * n_interp_1d * n_terms;

        ws.w_coefficients
            .par_chunks_mut(slab_len)
            .enumerate()
            .for_each(|(by, slab)| {
                slab.fill(T::zero());
                let (lo, hi) = (row_starts[by] as usize, row_starts[by + 1] as usize);
                for &pi in &order[lo..hi] {
                    let i = pi as usize;
                    let ((_, box_x), _, _) = box_data[i];
                    let w_start_idx = i * n_interp;
                    for interp_y in 0..n_interp {
                        let wy = y_weights[w_start_idx + interp_y];
                        let row = interp_y * n_interp_1d + box_x * n_interp;
                        for interp_x in 0..n_interp {
                            let weight = wy * x_weights[w_start_idx + interp_x];
                            let base = (row + interp_x) * n_terms;
                            for term in 0..n_terms {
                                slab[base + term] =
                                    slab[base + term] + weight * charges[i * n_terms + term];
                            }
                        }
                    }
                }
            });
    }

    // step 2: FFT convolution, parallel over the independent per-term slots.
    {
        let w_coefficients = &ws.w_coefficients;
        let fft_kernel = &grid.fft_kernel;

        ws.slots[..n_terms]
            .par_iter_mut()
            .enumerate()
            .for_each(|(term, slot)| {
                // embed grid into the FFT input's top-left quadrant. The
                // padding is zeroed once at slot creation and preserved by
                // the out-of-place R2C, so only the quadrant is rewritten.
                for i in 0..n_interp_1d {
                    let src = i * n_interp_1d;
                    let dst = i * n_fft;
                    for j in 0..n_interp_1d {
                        slot.fft_input[dst + j] = w_coefficients[(src + j) * n_terms + term];
                    }
                }

                T::execute_r2c(
                    &mut slot.plan_r2c,
                    &mut slot.fft_input,
                    &mut slot.fft_output,
                );

                // kernel multiply (Hadamard product); the kernel spectrum
                // already carries the 1 / n_fft^2 normalisation.
                for i in 0..n_complex {
                    let val = slot.fft_output[i];
                    let kern = fft_kernel[i];

                    let val_re = T::complex_re(val);
                    let val_im = T::complex_im(val);
                    let kern_re = T::complex_re(kern);
                    let kern_im = T::complex_im(kern);

                    let new_re = val_re * kern_re - val_im * kern_im;
                    let new_im = val_re * kern_im + val_im * kern_re;

                    slot.fft_output[i] = T::new_complex(new_re, new_im);
                }

                T::execute_c2r(
                    &mut slot.plan_c2r,
                    &mut slot.fft_output,
                    &mut slot.fft_scratch,
                );
            });
    }

    // step 2b: extraction of each slot's bottom-right quadrant into the
    // compact term-minor buffer, parallel over grid rows. Plain slice views:
    // the fftw plans are Send but not Sync, so the parallel passes must not
    // capture the slots themselves.
    {
        let grids: Vec<&[T]> = ws.slots[..n_terms]
            .iter()
            .map(|slot| &slot.fft_scratch[..])
            .collect();

        ws.y_tilde_values
            .par_chunks_mut(n_interp_1d * n_terms)
            .enumerate()
            .for_each(|(gy, row_out)| {
                let fft_row = (n_interp_1d + gy) * n_fft + n_interp_1d;
                for gx in 0..n_interp_1d {
                    for term in 0..n_terms {
                        row_out[gx * n_terms + term] = grids[term][fft_row + gx];
                    }
                }
            });
    }

    // step 3: gathering (parallel over points, pre-allocated output)
    {
        let box_data = &ws.box_data;
        let x_weights = &ws.x_weights;
        let y_weights = &ws.y_weights;
        let y_tilde_values = &ws.y_tilde_values;

        potentials_out
            .par_chunks_mut(n_terms)
            .enumerate()
            .for_each(|(i, out)| {
                for term in out.iter_mut() {
                    *term = T::zero();
                }
                let ((box_y, box_x), _, _) = box_data[i];
                let w_start_idx = i * n_interp;

                for interp_y in 0..n_interp {
                    let gy = box_y * n_interp + interp_y;
                    let wy = y_weights[w_start_idx + interp_y];
                    for interp_x in 0..n_interp {
                        let gx = box_x * n_interp + interp_x;
                        let weight = wy * x_weights[w_start_idx + interp_x];
                        let grid_idx = gy * n_interp_1d + gx;

                        for term in 0..n_terms {
                            out[term] =
                                out[term] + weight * y_tilde_values[grid_idx * n_terms + term];
                        }
                    }
                }
            });
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrange_weights() {
        let spacings = vec![0.25, 0.5, 0.75];
        let mut denominators = vec![0.0; 3];
        let mut weights = vec![0.0; 3];

        for i in 0..spacings.len() {
            let mut denom = 1.0;
            for j in 0..spacings.len() {
                if i != j {
                    denom *= spacings[i] - spacings[j];
                }
            }
            denominators[i] = denom;
        }

        lagrange_weights(0.5, &spacings, &denominators, &mut weights);
        assert!((weights[0] - 0.0).abs() < 1e-10);
        assert!((weights[1] - 1.0).abs() < 1e-10);
        assert!((weights[2] - 0.0).abs() < 1e-10);

        lagrange_weights(0.6, &spacings, &denominators, &mut weights);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_box_assignment() {
        let grid = FftGrid::<f64>::new(-10.0, 10.0, 4, 3);

        let (by, bx) = grid.point_to_box(-10.0, -10.0);
        assert_eq!((by, bx), (0, 0));

        let (by, bx) = grid.point_to_box(9.9, 9.9);
        assert_eq!((by, bx), (3, 3));

        let (by, bx) = grid.point_to_box(0.0, 0.0);
        assert_eq!((by, bx), (2, 2));
    }

    fn direct_repulsive_forces(embd: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, f64) {
        let n = embd.len();
        let mut sum_q = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = embd[i][0] - embd[j][0];
                    let dy = embd[i][1] - embd[j][1];
                    let dist_sq = dx * dx + dy * dy;
                    sum_q += 1.0 / (1.0 + dist_sq);
                }
            }
        }

        let mut forces_x = vec![0.0; n];
        let mut forces_y = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = embd[i][0] - embd[j][0];
                    let dy = embd[i][1] - embd[j][1];
                    let dist_sq = dx * dx + dy * dy;
                    let q_ij = 1.0 / (1.0 + dist_sq);
                    let weight = q_ij * q_ij / sum_q;
                    forces_x[i] += weight * dx;
                    forces_y[i] += weight * dy;
                }
            }
        }

        (forces_x, forces_y, sum_q)
    }

    #[test]
    fn test_repulsive_forces_match_direct() {
        let embd: Vec<Vec<f64>> = vec![
            vec![-2.1, 0.3],
            vec![-0.8, 1.2],
            vec![0.2, -0.9],
            vec![1.5, 0.4],
            vec![2.3, -0.6],
            vec![-1.2, -1.1],
            vec![0.7, 1.8],
            vec![1.9, -1.4],
        ];
        let n = embd.len();
        let n_terms = 4;

        let (direct_fx, direct_fy, direct_z) = direct_repulsive_forces(&embd);

        let coord_min = -5.0;
        let coord_max = 5.0;
        let grid = FftGrid::<f64>::new(coord_min, coord_max, 20, 3);
        let mut ws = FftWorkspace::new(grid.n_fft);

        let xs: Vec<f64> = embd.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = embd.iter().map(|p| p[1]).collect();

        let mut charges = vec![0.0; n * n_terms];
        for i in 0..n {
            charges[i * n_terms] = 1.0;
            charges[i * n_terms + 1] = xs[i];
            charges[i * n_terms + 2] = ys[i];
            charges[i * n_terms + 3] = xs[i] * xs[i] + ys[i] * ys[i];
        }

        let mut potentials = vec![0.0; n * n_terms];
        n_body_fft_2d(&xs, &ys, &charges, n_terms, &grid, &mut ws, &mut potentials);

        let mut fft_fx = vec![0.0; n];
        let mut fft_fy = vec![0.0; n];

        let mut fft_z = 0.0;
        for i in 0..n {
            let phi1 = potentials[i * n_terms];
            let phi2 = potentials[i * n_terms + 1];
            let phi3 = potentials[i * n_terms + 2];
            let phi4 = potentials[i * n_terms + 3];
            let r2 = xs[i] * xs[i] + ys[i] * ys[i];
            fft_z += (1.0 + r2) * phi1 - 2.0 * (xs[i] * phi2 + ys[i] * phi3) + phi4;
        }
        fft_z -= n as f64;

        let z_rel_err = (fft_z - direct_z).abs() / direct_z.abs();
        assert!(
            z_rel_err < 0.05,
            "Z mismatch: FFT={:.6}, Direct={:.6}, RelErr={:.6}",
            fft_z,
            direct_z,
            z_rel_err
        );

        for i in 0..n {
            let phi1 = potentials[i * n_terms];
            let phi2 = potentials[i * n_terms + 1];
            let phi3 = potentials[i * n_terms + 2];
            fft_fx[i] = (xs[i] * phi1 - phi2) / fft_z;
            fft_fy[i] = (ys[i] * phi1 - phi3) / fft_z;
        }

        for i in 0..n {
            let fx_err = (fft_fx[i] - direct_fx[i]).abs();
            let fy_err = (fft_fy[i] - direct_fy[i]).abs();
            let force_mag = (direct_fx[i].powi(2) + direct_fy[i].powi(2))
                .sqrt()
                .max(1e-10);

            assert!(
                fx_err / force_mag < 0.15 || fx_err < 1e-5,
                "Point {} Fx mismatch. FFT={:.6}, Direct={:.6}",
                i,
                fft_fx[i],
                direct_fx[i]
            );
            assert!(
                fy_err / force_mag < 0.15 || fy_err < 1e-5,
                "Point {} Fy mismatch. FFT={:.6}, Direct={:.6}",
                i,
                fft_fy[i],
                direct_fy[i]
            );
        }
    }

    #[test]
    fn test_grid_refinement_improves_accuracy() {
        let embd: Vec<Vec<f64>> = vec![
            vec![-1.0, 0.5],
            vec![0.5, -0.8],
            vec![1.2, 1.0],
            vec![-0.3, -0.5],
        ];
        let n = embd.len();
        let (direct_fx, direct_fy, _) = direct_repulsive_forces(&embd);

        let xs: Vec<f64> = embd.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = embd.iter().map(|p| p[1]).collect();

        let mut charges = vec![0.0; n * 4];
        for i in 0..n {
            charges[i * 4] = 1.0;
            charges[i * 4 + 1] = xs[i];
            charges[i * 4 + 2] = ys[i];
            charges[i * 4 + 3] = xs[i] * xs[i] + ys[i] * ys[i];
        }

        let run_fft_err = |n_boxes: usize| -> f64 {
            let grid = FftGrid::new(-5.0, 5.0, n_boxes, 3);
            let mut ws = FftWorkspace::new(grid.n_fft);
            let mut pots = vec![0.0; n * 4];
            n_body_fft_2d(&xs, &ys, &charges, 4, &grid, &mut ws, &mut pots);

            let mut z = 0.0;
            for i in 0..n {
                let r2 = xs[i] * xs[i] + ys[i] * ys[i];
                z += (1.0 + r2) * pots[i * 4]
                    - 2.0 * (xs[i] * pots[i * 4 + 1] + ys[i] * pots[i * 4 + 2])
                    + pots[i * 4 + 3];
            }
            z -= n as f64;

            let mut tot_err = 0.0;
            for i in 0..n {
                let fx = (xs[i] * pots[i * 4] - pots[i * 4 + 1]) / z;
                let fy = (ys[i] * pots[i * 4] - pots[i * 4 + 2]) / z;
                tot_err += (fx - direct_fx[i]).abs() + (fy - direct_fy[i]).abs();
            }
            tot_err
        };

        let err_coarse = run_fft_err(8);
        let err_fine = run_fft_err(32);

        assert!(
            err_fine < err_coarse,
            "Finer grid should reduce error: Coarse={}, Fine={}",
            err_coarse,
            err_fine
        );
    }

    #[test]
    fn test_forces_sum_to_zero() {
        let embd: Vec<Vec<f64>> = vec![
            vec![-2.0, 1.0],
            vec![1.0, -1.5],
            vec![0.5, 2.0],
            vec![-1.0, -0.5],
            vec![2.5, 0.0],
        ];
        let n = embd.len();
        let xs: Vec<f64> = embd.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = embd.iter().map(|p| p[1]).collect();

        let mut charges = vec![0.0; n * 4];
        for i in 0..n {
            charges[i * 4] = 1.0;
            charges[i * 4 + 1] = xs[i];
            charges[i * 4 + 2] = ys[i];
            charges[i * 4 + 3] = xs[i] * xs[i] + ys[i] * ys[i];
        }

        let grid = FftGrid::new(-5.0, 5.0, 20, 3);
        let mut ws = FftWorkspace::new(grid.n_fft);
        let mut pots = vec![0.0; n * 4];
        n_body_fft_2d(&xs, &ys, &charges, 4, &grid, &mut ws, &mut pots);

        let mut z = 0.0;
        for i in 0..n {
            let r2 = xs[i] * xs[i] + ys[i] * ys[i];
            z += (1.0 + r2) * pots[i * 4]
                - 2.0 * (xs[i] * pots[i * 4 + 1] + ys[i] * pots[i * 4 + 2])
                + pots[i * 4 + 3];
        }
        z -= n as f64;

        let mut sum_fx = 0.0;
        let mut sum_fy = 0.0;
        for i in 0..n {
            let fx = (xs[i] * pots[i * 4] - pots[i * 4 + 1]) / z;
            let fy = (ys[i] * pots[i * 4] - pots[i * 4 + 2]) / z;
            sum_fx += fx;
            sum_fy += fy;
        }

        assert!(
            sum_fx.abs() < 1e-4,
            "Net Force X should be ~0, got {}",
            sum_fx
        );
        assert!(
            sum_fy.abs() < 1e-4,
            "Net Force Y should be ~0, got {}",
            sum_fy
        );
    }
}

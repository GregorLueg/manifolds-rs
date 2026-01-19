use fftw::array::AlignedVec;
use fftw::plan::{C2RPlan, C2RPlan32, C2RPlan64, R2CPlan, R2CPlan32, R2CPlan64};
use fftw::types::{c32, c64, Flag};
use num_traits::{Float, FromPrimitive, Signed, ToPrimitive};
use std::fmt::Debug;

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
        R2CPlan::aligned(&[n, n], Flag::ESTIMATE).expect("Failed to create R2C plan")
    }

    fn plan_c2r_2d(n: usize) -> Self::C2RPlan {
        C2RPlan::aligned(&[n, n], Flag::ESTIMATE).expect("Failed to create C2R plan")
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
        R2CPlan::aligned(&[n, n], Flag::ESTIMATE).expect("Failed to create R2C plan")
    }

    fn plan_c2r_2d(n: usize) -> Self::C2RPlan {
        C2RPlan::aligned(&[n, n], Flag::ESTIMATE).expect("Failed to create C2R plan")
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

/////////////
// FftGrid //
/////////////

/// Grid structure for FFT-based interpolation.
///
/// Manages 2D uniform grid discretisation and pre-computed FFT kernel for
/// efficient potential field evaluation via convolution. The grid divides
/// coordinate space into boxes, each containing interpolation nodes for
/// Lagrange polynomial interpolation.
///
/// ### Fields
///
/// * `n_boxes_per_dim` - Number of boxes per dimension
/// * `n_interp_points` - Number of interpolation nodes per box dimension
///   (typically 3)
/// * `box_width` - Width of each box in coordinate space
/// * `coord_min` - Minimum coordinate value (same for x and y; assumes square
///   domain)
/// * `interp_spacings` - Interpolation node positions within [0,1] normalised
///   box coordinates
/// * `lagrange_denominators` - Pre-computed Lagrange polynomial denominators
///   for each node
/// * `global_x_coords` - Global grid node x-coordinates
///   (length `n_interp_points * n_boxes_per_dim`)
/// * `global_y_coords` - Global grid node y-coordinates
///   (length `n_interp_points * n_boxes_per_dim`)
/// * `fft_kernel` - Pre-computed FFT of convolution kernel;
///   dimensions `n_fft × (n_fft/2 + 1)` for R2C transform
/// * `n_fft` - FFT array dimension: `2 * n_interp_points * n_boxes_per_dim`
pub struct FftGrid<T: FftwFloat> {
    pub n_boxes_per_dim: usize,
    pub n_interp_points: usize,
    pub box_width: T,
    pub coord_min: T,
    pub interp_spacings: Vec<T>,
    pub lagrange_denominators: Vec<T>,
    pub global_x_coords: Vec<T>,
    pub global_y_coords: Vec<T>,
    pub fft_kernel: AlignedVec<T::Complex>,
    pub n_fft: usize,
}

impl<T: FftwFloat> FftGrid<T> {
    /// Create new FFT grid structure.
    ///
    /// Initialises uniform 2D grid covering square domain
    /// `[coord_min, coord_max]²`, computes Lagrange interpolation coefficients,
    /// and pre-computes FFT of the kernel `K(r) = 1/(1 + r²)²`.
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
    /// Constructs spatial kernel `K(r) = 1/(1 + r²)²` on zero-padded grid,
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
    /// Complex-valued FFT of kernel, dimensions `n_fft × (n_fft/2 + 1)`
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

                // Centre and three mirrored quadrants
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
    /// Position in [0,1] normalised box coordinates
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
    /// `coord_min + margin ≤ {x,y} ≤ coord_max - margin`
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

/// Compute Lagrange interpolation weights
///
/// For a point at position `y_in_box` (in [0,1]), computes the weights for all
/// interpolation nodes using Lagrange polynomials.
///
/// ### Params
///
/// * `y_in_box` - Position within box, range [0,1]
/// * `spacings` - Interpolation node positions in [0,1]
/// * `denominators` - Pre-computed denominators: product of (spacings[i] -
///   spacings[j]) for j != i
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

//////////////////
// FftWorkspace //
//////////////////

/// Workspace for FFT-based convolution computations.
///
/// Pre-allocates all working buffers and FFT plans required for repeated
/// convolution operations. Reusing workspace across iterations avoids
/// allocation overhead and plan creation costs.
///
/// ### Fields
///
/// * `w_coefficients` - Grid charges interpolated from points, shape
///   `[n_grid_points * n_terms]`
/// * `potentials` - Grid potentials after convolution, shape
///   `[n_grid_points * n_terms]`
/// * `point_boxes` - Box indices for each point, shape `[n_points]`
/// * `x_weights` - Lagrange weights for x-interpolation, shape
///   `[n_points * n_interp_points]`
/// * `y_weights` - Lagrange weights for y-interpolation, shape
///   `[n_points * n_interp_points]`
/// * `fft_input` - FFTW-aligned real buffer for forward transform, shape
///   `[n_fft * n_fft]`
/// * `fft_output` - FFTW-aligned complex buffer for transform output, shape
///   `[n_fft * (n_fft/2 + 1)]`
/// * `fft_scratch` - FFTW-aligned real buffer for inverse transform, shape
///   `[n_fft * n_fft]`
/// * `plan_r2c` - Reusable R2C FFT plan
/// * `plan_c2r` - Reusable C2R inverse FFT plan
/// * `n_fft` - FFT array dimension
pub struct FftWorkspace<T: FftwFloat> {
    pub w_coefficients: Vec<T>,
    pub potentials: Vec<T>,
    pub point_boxes: Vec<(usize, usize)>,
    pub x_weights: Vec<T>,
    pub y_weights: Vec<T>,
    fft_input: AlignedVec<T>,
    fft_output: AlignedVec<T::Complex>,
    fft_scratch: AlignedVec<T>,
    plan_r2c: T::R2CPlan,
    plan_c2r: T::C2RPlan,
    n_fft: usize,
}

impl<T: FftwFloat> FftWorkspace<T> {
    /// Allocate workspace for FFT computations.
    ///
    /// Pre-allocates all buffers and creates FFT plans. Buffer sizes are
    /// determined by grid configuration and number of charge terms.
    ///
    /// ### Params
    ///
    /// * `n_points` - Number of points to process
    /// * `n_terms` - Number of charge/potential terms (typically 4)
    /// * `grid` - Grid configuration defining array dimensions
    ///
    /// ### Returns
    ///
    /// Workspace with zero-initialised buffers and configured FFT plans
    pub fn new(n_points: usize, n_terms: usize, grid: &FftGrid<T>) -> Self {
        let n_interp_1d = grid.n_interp_points * grid.n_boxes_per_dim;
        let n_fft = 2 * n_interp_1d;
        let total_grid_points = n_interp_1d * n_interp_1d;
        let n_complex = n_fft * (n_fft / 2 + 1);

        Self {
            w_coefficients: vec![T::zero(); total_grid_points * n_terms],
            potentials: vec![T::zero(); total_grid_points * n_terms],
            point_boxes: vec![(0, 0); n_points],
            x_weights: vec![T::zero(); grid.n_interp_points * n_points],
            y_weights: vec![T::zero(); grid.n_interp_points * n_points],
            fft_input: T::aligned_real(n_fft * n_fft),
            fft_output: T::aligned_complex(n_complex),
            fft_scratch: T::aligned_real(n_fft * n_fft),
            plan_r2c: T::plan_r2c_2d(n_fft),
            plan_c2r: T::plan_c2r_2d(n_fft),
            n_fft,
        }
    }

    /// Reset charge and potential buffers to zero.
    ///
    /// Called between iterations to clear accumulated values. Does not
    /// re-allocate buffers or reset FFT plans.
    pub fn clear(&mut self) {
        for x in &mut self.w_coefficients {
            *x = T::zero();
        }
        for x in &mut self.potentials {
            *x = T::zero();
        }
    }
}

/////////////
// Helpers //
/////////////

/// Interpolate point charges onto grid using Lagrange polynomials.
///
/// Distributes each point's charge values to nearby grid nodes via tensor-
/// product Lagrange interpolation. Each point affects `n_interp × n_interp`
/// grid nodes within its containing box.
///
/// ### Params
///
/// * `xs` - X coordinates of points
/// * `ys` - Y coordinates of points
/// * `charges` - Charge values, shape `[n_points * n_terms]`
/// * `n_terms` - Number of charge components per point (typically 4)
/// * `grid` - Grid configuration
/// * `workspace` - Pre-allocated workspace; `w_coefficients` is written to
fn interpolate_charges_to_grid<T: FftwFloat>(
    xs: &[T],
    ys: &[T],
    charges: &[T],
    n_terms: usize,
    grid: &FftGrid<T>,
    workspace: &mut FftWorkspace<T>,
) {
    let n = xs.len();
    let n_interp = grid.n_interp_points;
    let n_boxes = grid.n_boxes_per_dim;
    let n_interp_1d = n_interp * n_boxes;

    for i in 0..n {
        workspace.point_boxes[i] = grid.point_to_box(xs[i], ys[i]);

        let (_, box_x) = workspace.point_boxes[i];
        let x_in_box = grid.position_in_box(xs[i], box_x);
        lagrange_weights(
            x_in_box,
            &grid.interp_spacings,
            &grid.lagrange_denominators,
            &mut workspace.x_weights[i * n_interp..(i + 1) * n_interp],
        );

        let (box_y, _) = workspace.point_boxes[i];
        let y_in_box = grid.position_in_box(ys[i], box_y);
        lagrange_weights(
            y_in_box,
            &grid.interp_spacings,
            &grid.lagrange_denominators,
            &mut workspace.y_weights[i * n_interp..(i + 1) * n_interp],
        );
    }

    for i in 0..n {
        let (box_y, box_x) = workspace.point_boxes[i];

        for interp_y in 0..n_interp {
            for interp_x in 0..n_interp {
                let grid_y = box_y * n_interp + interp_y;
                let grid_x = box_x * n_interp + interp_x;
                let grid_idx = grid_y * n_interp_1d + grid_x;

                let weight = workspace.y_weights[i * n_interp + interp_y]
                    * workspace.x_weights[i * n_interp + interp_x];

                for term in 0..n_terms {
                    workspace.w_coefficients[grid_idx * n_terms + term] = workspace.w_coefficients
                        [grid_idx * n_terms + term]
                        + weight * charges[i * n_terms + term];
                }
            }
        }
    }
}

/// Compute grid potentials via FFT-accelerated convolution.
///
/// For each charge term independently:
///
/// 1. Embed `n_interp_1d × n_interp_1d` charges into zero-padded
///    `n_fft × n_fft` array
/// 2. Compute R2C FFT of embedded charges
/// 3. Element-wise product with pre-computed kernel FFT
/// 4. Compute C2R inverse FFT
/// 5. Extract and normalise `n_interp_1d × n_interp_1d` potential values
///
/// ### Params
///
/// * `n_terms` - Number of charge/potential terms
/// * `grid` - Grid with pre-computed `fft_kernel`
/// * `workspace` - Workspace with input `w_coefficients`, output `potentials`
fn fft_convolution<T: FftwFloat>(
    n_terms: usize,
    grid: &FftGrid<T>,
    workspace: &mut FftWorkspace<T>,
) {
    let n_interp_1d = grid.n_interp_points * grid.n_boxes_per_dim;
    let n_fft = workspace.n_fft;
    let n_complex = n_fft * (n_fft / 2 + 1);

    for term in 0..n_terms {
        for x in workspace.fft_input.iter_mut() {
            *x = T::zero();
        }

        // Embed in top-left quadrant
        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let w_idx = (i * n_interp_1d + j) * n_terms + term;
                let fft_idx = i * n_fft + j;
                workspace.fft_input[fft_idx] = workspace.w_coefficients[w_idx];
            }
        }

        // R2C FFT
        T::execute_r2c(
            &mut workspace.plan_r2c,
            &mut workspace.fft_input,
            &mut workspace.fft_output,
        );

        // Hadamard product
        for i in 0..n_complex {
            let a_re = T::complex_re(workspace.fft_output[i]);
            let a_im = T::complex_im(workspace.fft_output[i]);
            let b_re = T::complex_re(grid.fft_kernel[i]);
            let b_im = T::complex_im(grid.fft_kernel[i]);
            workspace.fft_output[i] =
                T::new_complex(a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re);
        }

        // C2R IFFT
        T::execute_c2r(
            &mut workspace.plan_c2r,
            &mut workspace.fft_output,
            &mut workspace.fft_scratch,
        );

        // Extract potentials with normalisation
        let norm = T::from_usize(n_fft * n_fft).unwrap();
        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let fft_idx = (n_interp_1d + i) * n_fft + (n_interp_1d + j);
                let pot_idx = (i * n_interp_1d + j) * n_terms + term;
                workspace.potentials[pot_idx] = workspace.fft_scratch[fft_idx] / norm;
            }
        }
    }
}

/// Interpolate grid potentials back to point locations.
///
/// Gathers potential values from `n_interp × n_interp` nearby grid nodes for
/// each point, using the same Lagrange weights computed during charge
/// interpolation.
///
/// ### Params
///
/// * `potentials_out` - Output buffer, shape `[n_points * n_terms]`
/// * `n_terms` - Number of potential components
/// * `grid` - Grid configuration
/// * `workspace` - Workspace containing `potentials` and pre-computed interpolation weights
fn interpolate_potentials_to_points<T: FftwFloat>(
    potentials_out: &mut [T],
    n_terms: usize,
    grid: &FftGrid<T>,
    workspace: &FftWorkspace<T>,
) {
    let n = workspace.point_boxes.len();
    let n_interp = grid.n_interp_points;
    let n_interp_1d = n_interp * grid.n_boxes_per_dim;

    for x in potentials_out.iter_mut() {
        *x = T::zero();
    }

    for i in 0..n {
        let (box_y, box_x) = workspace.point_boxes[i];

        for interp_y in 0..n_interp {
            for interp_x in 0..n_interp {
                let grid_y = box_y * n_interp + interp_y;
                let grid_x = box_x * n_interp + interp_x;
                let grid_idx = grid_y * n_interp_1d + grid_x;

                let weight = workspace.y_weights[i * n_interp + interp_y]
                    * workspace.x_weights[i * n_interp + interp_x];

                for term in 0..n_terms {
                    potentials_out[i * n_terms + term] = potentials_out[i * n_terms + term]
                        + weight * workspace.potentials[grid_idx * n_terms + term];
                }
            }
        }
    }
}

/// Determine optimal grid size based on data spread
///
/// Follows FIt-SNE's heuristic: number of boxes scales with spread,
/// then rounds up to FFT-friendly sizes
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
        25, 27, 32, 36, 40, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125,
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

/// Compute repulsive forces using FFT acceleration
///
/// Computes the "negative" or repulsive forces for t-SNE using the FFT-based
/// interpolation method. This is much faster than Barnes-Hut for large N and
/// based on the works of Linderman, et al., Nat Med (2019).
///
/// The method uses 4 charge terms to efficiently compute:
///
/// - F_rep,x(i) = (x_i·φ₁ - φ₂) / Z
/// - F_rep,y(i) = (y_i·φ₁ - φ₃) / Z
/// - Z = Σᵢ[(1 + x²ᵢ + y²ᵢ)φ₁ - 2(xᵢφ₂ + yᵢφ₃) + φ₄] - N
///
/// ### Params
///
/// * `embd` - Current embedding positions [n_points][2]
/// * `grid` - Pre-configured FFT grid
/// * `workspace` - Pre-allocated workspace buffers
///
/// ### Returns
///
/// Tuple of (forces_x, forces_y, sum_Q) where:
/// - forces_x: repulsive forces in x direction
/// - forces_y: repulsive forces in y direction
/// - sum_Q: normalization constant Z
pub fn compute_repulsive_forces_fft<T: FftwFloat>(
    embd: &[Vec<T>],
    grid: &FftGrid<T>,
    workspace: &mut FftWorkspace<T>,
) -> (Vec<T>, Vec<T>, T) {
    let n = embd.len();
    let n_terms = 4;

    let xs: Vec<T> = embd.iter().map(|p| p[0]).collect();
    let ys: Vec<T> = embd.iter().map(|p| p[1]).collect();

    let mut charges = vec![T::zero(); n * n_terms];
    for i in 0..n {
        charges[i * n_terms] = T::one();
        charges[i * n_terms + 1] = xs[i];
        charges[i * n_terms + 2] = ys[i];
        charges[i * n_terms + 3] = xs[i] * xs[i] + ys[i] * ys[i];
    }

    workspace.clear();

    interpolate_charges_to_grid(&xs, &ys, &charges, n_terms, grid, workspace);
    fft_convolution(n_terms, grid, workspace);

    let mut potentials = vec![T::zero(); n * n_terms];
    interpolate_potentials_to_points(&mut potentials, n_terms, grid, workspace);

    // Compute Z
    let mut sum_q = T::zero();
    let two = T::one() + T::one();
    for i in 0..n {
        let phi1 = potentials[i * n_terms];
        let phi2 = potentials[i * n_terms + 1];
        let phi3 = potentials[i * n_terms + 2];
        let phi4 = potentials[i * n_terms + 3];

        sum_q = sum_q + (T::one() + xs[i] * xs[i] + ys[i] * ys[i]) * phi1
            - two * (xs[i] * phi2 + ys[i] * phi3)
            + phi4;
    }
    sum_q = sum_q - T::from_usize(n).unwrap();

    // Repulsive forces (un-normalised)
    let mut forces_x = vec![T::zero(); n];
    let mut forces_y = vec![T::zero(); n];

    for i in 0..n {
        let phi1 = potentials[i * n_terms];
        let phi2 = potentials[i * n_terms + 1];
        let phi3 = potentials[i * n_terms + 2];

        forces_x[i] = xs[i] * phi1 - phi2;
        forces_y[i] = ys[i] * phi1 - phi3;
    }

    (forces_x, forces_y, sum_q)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrange_weights() {
        // Test on 3-point interpolation
        let spacings = vec![0.25, 0.5, 0.75];
        let mut denominators = vec![0.0; 3];
        let mut weights = vec![0.0; 3];

        // Compute denominators
        for i in 0..spacings.len() {
            let mut denom = 1.0;
            for j in 0..spacings.len() {
                if i != j {
                    denom *= spacings[i] - spacings[j];
                }
            }
            denominators[i] = denom;
        }

        // At the middle point, should get [0, 1, 0]
        lagrange_weights(0.5, &spacings, &denominators, &mut weights);
        assert!((weights[0] - 0.0).abs() < 1e-10);
        assert!((weights[1] - 1.0).abs() < 1e-10);
        assert!((weights[2] - 0.0).abs() < 1e-10);

        // Weights should sum to 1
        lagrange_weights(0.6, &spacings, &denominators, &mut weights);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_box_assignment() {
        let grid = FftGrid::<f64>::new(-10.0, 10.0, 4, 3);

        // Point at (-10, -10) should be in box (0, 0)
        let (by, bx) = grid.point_to_box(-10.0, -10.0);
        assert_eq!((by, bx), (0, 0));

        // Point at (9.9, 9.9) should be in box (3, 3)
        let (by, bx) = grid.point_to_box(9.9, 9.9);
        assert_eq!((by, bx), (3, 3));

        // Point at (0, 0) should be in box (2, 2) (middle)
        let (by, bx) = grid.point_to_box(0.0, 0.0);
        assert_eq!((by, bx), (2, 2));
    }

    /// Direct computation of repulsive forces for validation
    fn direct_repulsive_forces(embd: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, f64) {
        let n = embd.len();
        let mut sum_q = 0.0;

        // Compute normalisation constant
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

        // Compute forces
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
    fn test_fft_convolution_on_grid() {
        // Test the FFT convolution directly on grid points
        let n_boxes = 8;
        let n_interp = 3;
        let grid = FftGrid::<f64>::new(-4.0, 4.0, n_boxes, n_interp);

        let n_interp_1d = n_boxes * n_interp;
        let n_terms = 1;

        // Put unit charges on a few grid points
        let mut w_coefficients = vec![0.0; n_interp_1d * n_interp_1d * n_terms];

        // Place charges at grid points (5,5), (10,10), (15,8)
        let charge_positions = [(5, 5), (10, 10), (15, 8)];
        let charge_values = [1.0, 2.0, 1.5];

        for (&(gy, gx), &q) in charge_positions.iter().zip(&charge_values) {
            let idx = (gy * n_interp_1d + gx) * n_terms;
            w_coefficients[idx] = q;
        }

        // Run FFT convolution
        let mut workspace = FftWorkspace::new(1, n_terms, &grid);
        workspace.w_coefficients = w_coefficients;

        fft_convolution(n_terms, &grid, &mut workspace);

        // Direct computation on grid
        let mut direct_potentials = vec![0.0; n_interp_1d * n_interp_1d];

        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let xi = grid.global_x_coords[j];
                let yi = grid.global_y_coords[i];

                for (&(gy, gx), &q) in charge_positions.iter().zip(&charge_values) {
                    let xj = grid.global_x_coords[gx];
                    let yj = grid.global_y_coords[gy];

                    let dx = xi - xj;
                    let dy = yi - yj;
                    let dist_sq = dx * dx + dy * dy;
                    let kernel = 1.0 / ((1.0 + dist_sq) * (1.0 + dist_sq));

                    direct_potentials[i * n_interp_1d + j] += kernel * q;
                }
            }
        }

        // Compare
        let mut max_rel_err = 0.0;
        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let fft_val = workspace.potentials[(i * n_interp_1d + j) * n_terms];
                let direct_val = direct_potentials[i * n_interp_1d + j];

                if direct_val.abs() > 1e-10 {
                    let rel_err = (fft_val - direct_val).abs() / direct_val.abs();
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
        }

        assert!(
            max_rel_err < 1e-10,
            "FFT convolution mismatch: max relative error = {:.2e}",
            max_rel_err
        );
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

        let (direct_fx, direct_fy, direct_z) = direct_repulsive_forces(&embd);

        let coord_min = -5.0;
        let coord_max = 5.0;
        let grid = FftGrid::<f64>::new(coord_min, coord_max, 10, 3);
        let mut workspace = FftWorkspace::new(n, 4, &grid);

        let (mut fft_fx, mut fft_fy, fft_z) =
            compute_repulsive_forces_fft(&embd, &grid, &mut workspace);

        // Normalise FFT forces to match direct computation
        for i in 0..n {
            fft_fx[i] /= fft_z;
            fft_fy[i] /= fft_z;
        }

        // Compare Z
        let z_rel_err = (fft_z - direct_z).abs() / direct_z.abs();
        assert!(
            z_rel_err < 0.1,
            "Z mismatch: FFT={:.6}, direct={:.6}",
            fft_z,
            direct_z
        );

        // Compare forces
        for i in 0..n {
            let fx_err = (fft_fx[i] - direct_fx[i]).abs();
            let fy_err = (fft_fy[i] - direct_fy[i]).abs();
            let force_mag = (direct_fx[i].powi(2) + direct_fy[i].powi(2))
                .sqrt()
                .max(1e-10);

            assert!(
                fx_err / force_mag < 0.15,
                "Point {} Fx: FFT={:.6}, direct={:.6}",
                i,
                fft_fx[i],
                direct_fx[i]
            );
            assert!(
                fy_err / force_mag < 0.15,
                "Point {} Fy: FFT={:.6}, direct={:.6}",
                i,
                fft_fy[i],
                direct_fy[i]
            );
        }
    }

    #[test]
    fn test_charge_conservation() {
        // Total charge spread to grid should equal total charge gathered back
        let xs = vec![-1.0, 0.0, 1.0, 0.5];
        let ys = vec![0.0, 1.0, -0.5, 0.5];
        let n = xs.len();
        let n_terms = 1;

        let charges: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let total_charge: f64 = charges.iter().sum();

        let grid = FftGrid::<f64>::new(-5.0, 5.0, 8, 3);
        let mut workspace = FftWorkspace::new(n, n_terms, &grid);

        interpolate_charges_to_grid(&xs, &ys, &charges, n_terms, &grid, &mut workspace);

        // Sum charges on grid
        let grid_total: f64 = workspace.w_coefficients.iter().sum();

        let rel_err = (grid_total - total_charge).abs() / total_charge;
        assert!(
            rel_err < 1e-10,
            "Charge not conserved: input={}, grid={}",
            total_charge,
            grid_total
        );
    }

    #[test]
    fn test_forces_sum_to_zero() {
        // Newton's third law: total repulsive force should be zero
        let embd: Vec<Vec<f64>> = vec![
            vec![-2.0, 1.0],
            vec![1.0, -1.5],
            vec![0.5, 2.0],
            vec![-1.0, -0.5],
            vec![2.5, 0.0],
        ];
        let n = embd.len();

        let grid = FftGrid::<f64>::new(-5.0, 5.0, 10, 3);
        let mut workspace = FftWorkspace::new(n, 4, &grid);

        let (fx, fy, _) = compute_repulsive_forces_fft(&embd, &grid, &mut workspace);

        let total_fx: f64 = fx.iter().sum();
        let total_fy: f64 = fy.iter().sum();

        // Should be very close to zero (within interpolation error)
        assert!(
            total_fx.abs() < 0.01,
            "Total Fx should be ~0, got {}",
            total_fx
        );
        assert!(
            total_fy.abs() < 0.01,
            "Total Fy should be ~0, got {}",
            total_fy
        );
    }

    #[test]
    fn test_grid_refinement_improves_accuracy() {
        // Finer grid should give better approximation
        let embd: Vec<Vec<f64>> = vec![
            vec![-1.0, 0.5],
            vec![0.5, -0.8],
            vec![1.2, 1.0],
            vec![-0.3, -0.5],
        ];
        let n = embd.len();

        let (direct_fx, direct_fy, _) = direct_repulsive_forces(&embd);

        let mut errors_coarse = Vec::new();
        let mut errors_fine = Vec::new();

        // Coarse grid
        let grid_coarse = FftGrid::<f64>::new(-5.0, 5.0, 6, 3);
        let mut ws_coarse = FftWorkspace::new(n, 4, &grid_coarse);
        let (fx_c, fy_c, _) = compute_repulsive_forces_fft(&embd, &grid_coarse, &mut ws_coarse);

        for i in 0..n {
            errors_coarse.push((fx_c[i] - direct_fx[i]).abs() + (fy_c[i] - direct_fy[i]).abs());
        }

        // Fine grid
        let grid_fine = FftGrid::<f64>::new(-5.0, 5.0, 12, 3);
        let mut ws_fine = FftWorkspace::new(n, 4, &grid_fine);
        let (fx_f, fy_f, _) = compute_repulsive_forces_fft(&embd, &grid_fine, &mut ws_fine);

        for i in 0..n {
            errors_fine.push((fx_f[i] - direct_fx[i]).abs() + (fy_f[i] - direct_fy[i]).abs());
        }

        let avg_coarse: f64 = errors_coarse.iter().sum::<f64>() / n as f64;
        let avg_fine: f64 = errors_fine.iter().sum::<f64>() / n as f64;

        assert!(
            avg_fine < avg_coarse,
            "Finer grid should reduce error: coarse={:.6}, fine={:.6}",
            avg_coarse,
            avg_fine
        );
    }

    #[test]
    fn test_interpolation_round_trip() {
        // If we spread a delta function and gather at the same point,
        // we should recover approximately the original value
        let grid = FftGrid::<f64>::new(-5.0, 5.0, 10, 3);

        // Single point at a non-grid location
        let xs = vec![0.73];
        let ys = vec![-0.42];
        let n_terms = 1;
        let charges = vec![1.0];

        let mut workspace = FftWorkspace::new(1, n_terms, &grid);

        // Spread to grid
        interpolate_charges_to_grid(&xs, &ys, &charges, n_terms, &grid, &mut workspace);

        // Check total charge is conserved
        let grid_total: f64 = workspace.w_coefficients.iter().sum();
        assert!(
            (grid_total - 1.0).abs() < 1e-10,
            "Charge not conserved: got {}",
            grid_total
        );

        // Copy charges to potentials (identity "convolution")
        workspace
            .potentials
            .copy_from_slice(&workspace.w_coefficients);

        // Gather back
        let mut recovered = vec![0.0; n_terms];
        interpolate_potentials_to_points(&mut recovered, n_terms, &grid, &workspace);

        // For Lagrange interpolation, spreading then gathering with identity
        // should give sum of squared weights, not 1.0
        // But the value should be positive and bounded
        assert!(
            recovered[0] > 0.0 && recovered[0] <= 1.0,
            "Round-trip value out of range: {}",
            recovered[0]
        );
    }

    #[test]
    fn test_interpolation_at_grid_node() {
        // Point exactly at a grid node should have cleaner interpolation
        let grid = FftGrid::<f64>::new(-4.0, 4.0, 8, 3);

        // Place point exactly at first grid node
        let xs = vec![grid.global_x_coords[0]];
        let ys = vec![grid.global_y_coords[0]];
        let n_terms = 1;
        let charges = vec![1.0];

        let mut workspace = FftWorkspace::new(1, n_terms, &grid);

        interpolate_charges_to_grid(&xs, &ys, &charges, n_terms, &grid, &mut workspace);

        // Should place all charge at grid point (0,0)
        let charge_at_origin = workspace.w_coefficients[0];
        let total: f64 = workspace.w_coefficients.iter().sum();

        assert!(
            (total - 1.0).abs() < 1e-10,
            "Total charge should be 1.0, got {}",
            total
        );

        // Most charge should be at the target node
        assert!(
            charge_at_origin > 0.5,
            "Expected most charge at origin, got {}",
            charge_at_origin
        );
    }
}

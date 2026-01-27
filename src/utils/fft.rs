use fftw::array::AlignedVec;
use fftw::plan::{C2RPlan, C2RPlan32, C2RPlan64, R2CPlan, R2CPlan32, R2CPlan64};
use fftw::types::{c32, c64, Flag};
use num_traits::{Float, FromPrimitive, Signed, ToPrimitive};
use rayon::prelude::*;
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

/// Compute repulsive forces/potentials using FFT acceleration.
///
/// Generic implementation supporting f32/f64 via the `FftwFloat` trait.
/// Modelled after the much more efficient C++ code. Note to self: do not be
/// smart with structures in Rust, just translate working C++ into Rust.
///
/// ### Params
///
/// * `xs`: x-coordinates of the points
/// * `ys`: y-coordinates of the points
/// * `charges`: charges of the points
/// * `n_terms`: number of terms in the expansion
/// * `grid`: FFT grid
///
/// ### Returns
///
/// Forces/potentials of the points
pub fn n_body_fft_2d<T: FftwFloat>(
    xs: &[T],
    ys: &[T],
    charges: &[T],
    n_terms: usize,
    grid: &FftGrid<T>,
) -> Vec<T> {
    let n_points = xs.len();
    let n_interp = grid.n_interp_points;
    let n_boxes = grid.n_boxes_per_dim;
    let n_interp_1d = n_interp * n_boxes;
    let total_grid_points = n_interp_1d * n_interp_1d;

    // step 1: box assignment and relative coords
    let box_data: Vec<((usize, usize), T, T)> = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let (box_y, box_x) = grid.point_to_box(xs[i], ys[i]);
            let x_in_box = grid.position_in_box(xs[i], box_x);
            let y_in_box = grid.position_in_box(ys[i], box_y);
            ((box_y, box_x), x_in_box, y_in_box)
        })
        .collect();

    // step 2: interpolation weights (math only)
    let (x_weights_all, y_weights_all): (Vec<Vec<T>>, Vec<Vec<T>>) = (0..n_points)
        .into_par_iter()
        .map(|i| {
            let (_, x_pos, y_pos) = box_data[i];

            let mut x_w = vec![T::zero(); n_interp];
            let mut y_w = vec![T::zero(); n_interp];

            lagrange_weights(
                x_pos,
                &grid.interp_spacings,
                &grid.lagrange_denominators,
                &mut x_w,
            );
            lagrange_weights(
                y_pos,
                &grid.interp_spacings,
                &grid.lagrange_denominators,
                &mut y_w,
            );

            (x_w, y_w)
        })
        .unzip();

    // flatten for cache locality in the serial loop
    let x_weights_flat: Vec<T> = x_weights_all.into_iter().flatten().collect();
    let y_weights_flat: Vec<T> = y_weights_all.into_iter().flatten().collect();

    // step 1b: grid assembly
    let mut w_coefficients = vec![T::zero(); total_grid_points * n_terms];

    for i in 0..n_points {
        let ((box_y, box_x), _, _) = box_data[i];
        let w_start_idx = i * n_interp;

        for interp_y in 0..n_interp {
            for interp_x in 0..n_interp {
                let gy = box_y * n_interp + interp_y;
                let gx = box_x * n_interp + interp_x;
                let grid_idx = gy * n_interp_1d + gx;

                let weight =
                    y_weights_flat[w_start_idx + interp_y] * x_weights_flat[w_start_idx + interp_x];

                for term in 0..n_terms {
                    w_coefficients[grid_idx * n_terms + term] = w_coefficients
                        [grid_idx * n_terms + term]
                        + weight * charges[i * n_terms + term];
                }
            }
        }
    }

    // step 2: FFT convolution
    let n_fft = grid.n_fft;
    let n_complex = n_fft * (n_fft / 2 + 1);

    // allocate aligned buffers using the trait methods
    let mut fft_input = T::aligned_real(n_fft * n_fft);
    let mut fft_output = T::aligned_complex(n_complex);
    let mut fft_scratch = T::aligned_real(n_fft * n_fft);

    // create plans using trait methods
    let mut plan_r2c = T::plan_r2c_2d(n_fft);
    let mut plan_c2r = T::plan_c2r_2d(n_fft);

    let mut y_tilde_values = vec![T::zero(); total_grid_points * n_terms];

    for term in 0..n_terms {
        for x in fft_input.iter_mut() {
            *x = T::zero();
        }

        // embed grid into FFT input (top-left quadrant)
        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let w_idx = (i * n_interp_1d + j) * n_terms + term;
                let fft_idx = i * n_fft + j;
                fft_input[fft_idx] = w_coefficients[w_idx];
            }
        }

        // 1. R2C
        T::execute_r2c(&mut plan_r2c, &mut fft_input, &mut fft_output);

        // 2. kernel multiply (Hadamard Product)
        for i in 0..n_complex {
            let val = fft_output[i];
            let kern = grid.fft_kernel[i];

            // extract components via trait
            let val_re = T::complex_re(val);
            let val_im = T::complex_im(val);
            let kern_re = T::complex_re(kern);
            let kern_im = T::complex_im(kern);

            // complex mul: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            let new_re = val_re * kern_re - val_im * kern_im;
            let new_im = val_re * kern_im + val_im * kern_re;

            fft_output[i] = T::new_complex(new_re, new_im);
        }

        // 3. C2R
        T::execute_c2r(&mut plan_c2r, &mut fft_output, &mut fft_scratch);

        // 4. normalise
        let norm_factor = T::from_usize(n_fft * n_fft).unwrap();

        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                // extract from the specific quadrant used in C++ logic
                // the C++ logic effectively reads from the wrapped/mirrored position
                let fft_idx = (n_interp_1d + i) * n_fft + (n_interp_1d + j);

                let val = fft_scratch[fft_idx] / norm_factor;
                let pot_idx = (i * n_interp_1d + j) * n_terms + term;
                y_tilde_values[pot_idx] = val;
            }
        }
    }

    // step 3: gathering
    let potentials_flat: Vec<T> = (0..n_points)
        .into_par_iter()
        .flat_map(|i| {
            let ((box_y, box_x), _, _) = box_data[i];
            let w_start_idx = i * n_interp;

            let mut point_potentials = vec![T::zero(); n_terms];

            for interp_y in 0..n_interp {
                for interp_x in 0..n_interp {
                    let gy = box_y * n_interp + interp_y;
                    let gx = box_x * n_interp + interp_x;
                    let grid_idx = gy * n_interp_1d + gx;

                    let weight = y_weights_flat[w_start_idx + interp_y]
                        * x_weights_flat[w_start_idx + interp_x];

                    for term in 0..n_terms {
                        point_potentials[term] = point_potentials[term]
                            + weight * y_tilde_values[grid_idx * n_terms + term];
                    }
                }
            }
            point_potentials
        })
        .collect();

    potentials_flat
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

        // Compute denominators manually for test
        for i in 0..spacings.len() {
            let mut denom = 1.0;
            for j in 0..spacings.len() {
                if i != j {
                    denom *= spacings[i] - spacings[j];
                }
            }
            denominators[i] = denom;
        }

        // At the middle point (0.5), should be exactly [0, 1, 0]
        lagrange_weights(0.5, &spacings, &denominators, &mut weights);
        assert!((weights[0] - 0.0).abs() < 1e-10);
        assert!((weights[1] - 1.0).abs() < 1e-10);
        assert!((weights[2] - 0.0).abs() < 1e-10);

        // Weights should sum to 1.0 anywhere
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

    /// Helper: Brute force computation of repulsive forces
    /// Returns (fx, fy, Z)
    fn direct_repulsive_forces(embd: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, f64) {
        let n = embd.len();
        let mut sum_q = 0.0;

        // 1. Compute Z (normalization constant)
        // Z = Sum_{k!=l} 1 / (1 + ||y_k - y_l||^2)
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

        // 2. Compute Forces
        // F_i = Sum_{j!=i} (y_i - y_j) * Q_ij^2 / Z
        // Note: Q_ij = 1 / (1 + dist^2)
        let mut forces_x = vec![0.0; n];
        let mut forces_y = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = embd[i][0] - embd[j][0];
                    let dy = embd[i][1] - embd[j][1];
                    let dist_sq = dx * dx + dy * dy;
                    let q_ij = 1.0 / (1.0 + dist_sq);

                    // The repulsive gradient term in t-SNE is q_ij^2 / Z (un-normalized q_ij squared)
                    // actually it's q_ij * Q_ij ... Q_ij = q_ij/Z.
                    // Repulsive force term: q_ij^2 / Z
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
        // A small random-ish embedding
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

        // 1. Calculate Truth via Brute Force
        let (direct_fx, direct_fy, direct_z) = direct_repulsive_forces(&embd);

        // 2. Calculate via FFT
        // Define grid loose enough to capture points
        let coord_min = -5.0;
        let coord_max = 5.0;
        // 10 boxes, 3 interp points is standard-ish for accuracy
        let grid = FftGrid::<f64>::new(coord_min, coord_max, 20, 3);

        // Prepare inputs
        let xs: Vec<f64> = embd.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = embd.iter().map(|p| p[1]).collect();

        // Construct charges manually as done in the optimizer
        // Terms: 1, x, y, x^2+y^2
        let mut charges = vec![0.0; n * n_terms];
        for i in 0..n {
            charges[i * n_terms] = 1.0;
            charges[i * n_terms + 1] = xs[i];
            charges[i * n_terms + 2] = ys[i];
            charges[i * n_terms + 3] = xs[i] * xs[i] + ys[i] * ys[i];
        }

        // Run FFT
        let potentials = n_body_fft_2d(&xs, &ys, &charges, n_terms, &grid);

        // Decode Potentials to Forces (matching optimizer logic)
        let mut fft_fx = vec![0.0; n];
        let mut fft_fy = vec![0.0; n];

        // Calculate Z from potentials
        // Z = Sum [ (1+r^2)phi1 - 2(x*phi2 + y*phi3) + phi4 ] - N
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

        // Check Z accuracy
        let z_rel_err = (fft_z - direct_z).abs() / direct_z.abs();
        assert!(
            z_rel_err < 0.05,
            "Z mismatch: FFT={:.6}, Direct={:.6}, RelErr={:.6}",
            fft_z,
            direct_z,
            z_rel_err
        );

        // Calculate Forces from potentials
        // Fx = (x*phi1 - phi2) / Z
        // Fy = (y*phi1 - phi3) / Z
        for i in 0..n {
            let phi1 = potentials[i * n_terms];
            let phi2 = potentials[i * n_terms + 1];
            let phi3 = potentials[i * n_terms + 2];

            fft_fx[i] = (xs[i] * phi1 - phi2) / fft_z;
            fft_fy[i] = (ys[i] * phi1 - phi3) / fft_z;
        }

        // Compare Forces
        for i in 0..n {
            let fx_err = (fft_fx[i] - direct_fx[i]).abs();
            let fy_err = (fft_fy[i] - direct_fy[i]).abs();

            // Allow larger error for small forces
            let force_mag = (direct_fx[i].powi(2) + direct_fy[i].powi(2))
                .sqrt()
                .max(1e-10);

            // 15% relative error tolerance for approximation is acceptable for t-SNE
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

        // Helper to run fft and get avg error
        let run_fft_err = |n_boxes: usize| -> f64 {
            let grid = FftGrid::new(-5.0, 5.0, n_boxes, 3);
            let pots = n_body_fft_2d(&xs, &ys, &charges, 4, &grid);

            // Compute Z
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

        let err_coarse = run_fft_err(8); // Coarse grid
        let err_fine = run_fft_err(32); // Fine grid

        assert!(
            err_fine < err_coarse,
            "Finer grid should reduce error: Coarse={}, Fine={}",
            err_coarse,
            err_fine
        );
    }

    #[test]
    fn test_forces_sum_to_zero() {
        // Newton's third law check
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

        // 4 terms setup
        let mut charges = vec![0.0; n * 4];
        for i in 0..n {
            charges[i * 4] = 1.0;
            charges[i * 4 + 1] = xs[i];
            charges[i * 4 + 2] = ys[i];
            charges[i * 4 + 3] = xs[i] * xs[i] + ys[i] * ys[i];
        }

        let grid = FftGrid::new(-5.0, 5.0, 20, 3);
        let pots = n_body_fft_2d(&xs, &ys, &charges, 4, &grid);

        // Calc Z
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

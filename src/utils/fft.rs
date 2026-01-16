use num_traits::{Float, FromPrimitive, Signed};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::fmt::Debug;
use std::sync::Arc;

////////////
// Traits //
////////////

/// Trait for floating-point types used in FFT computations
pub trait FftFloat: Float + FromPrimitive + Send + Sync + Debug + Signed + 'static {}
impl<T: Float + FromPrimitive + Send + Sync + Debug + Signed + 'static> FftFloat for T {}

/////////////
// FftGrid //
/////////////

/// Grid-structure for FFT-based interpolation
///
/// Manages the 2D grid discretisation and pre-computed FFT kernel
///
/// ### Fields
///
/// * `n_boxes_per_dim` - Number of boxes per dimension
/// * `n_interp_points` - Number of interpolation points per dimension
///   (typically 3).
/// * `box_width` - Width of each box in coordinate space
/// * `coord_min` - Minimum coordinate value (same for x and y)
/// * `interp_spacings` - Interpolation node positions within [0,1] box
/// * `lagrange_denominators` - Lagrange polynomial denominators.
/// * `global_x_coords` - Global grid coordinates for x dimension
/// * `global_y_coords` - Global grid coordinates for y dimension
/// * `fft_kernel` - Pre-computed FFT kernel; size:
///   `(2*n_interp_1d) × (2*n_interp_1d/2 + 1)` for R2C FFT
/// * `fft_forword` - FFT plan for forward transform (reused each iteration)
/// * `fft_inverse` - FFT plan for inverse transform
pub struct FftGrid<T> {
    pub n_boxes_per_dim: usize,
    pub n_interp_points: usize,
    pub box_width: T,
    pub coord_min: T,
    pub interp_spacings: Vec<T>,
    pub lagrange_denominators: Vec<T>,
    pub global_x_coords: Vec<T>,
    pub global_y_coords: Vec<T>,
    pub fft_kernel: Vec<Complex<T>>,
    fft_forward: Arc<dyn Fft<T>>,
    fft_inverse: Arc<dyn Fft<T>>,
}

impl<T> FftGrid<T>
where
    T: FftFloat,
{
    /// Create new FFT grid and precompute kernel
    ///
    /// ### Params
    ///
    /// * `coord_min` - Minimum coordinate value
    /// * `coord_max` - Maximum coordinate value
    /// * `n_boxes_per_dim` - Number of boxes per dimension
    /// * `n_interp_points` - Interpolation points per box (typically 3)
    ///
    /// ### Returns
    ///
    /// * `Self` - New FFT grid
    pub fn new(coord_min: T, coord_max: T, n_boxes_per_dim: usize, n_interp_points: usize) -> Self {
        let box_width = (coord_max - coord_min) / T::from_usize(n_boxes_per_dim).unwrap();

        // equispaced interpolation nodes within [0, 1] box
        let h = T::one() / T::from_usize(n_interp_points).unwrap();
        let mut interp_spacings = vec![T::zero(); n_interp_points];
        interp_spacings[0] = h / (T::one() + T::one()); // h/2
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

        // global grid coords
        let n_interp_1d = n_interp_points * n_boxes_per_dim;
        let global_h = h * box_width;
        let two = T::one() + T::one();

        let mut global_x_coords = vec![T::zero(); n_interp_1d];
        let mut global_y_coords = vec![T::zero(); n_interp_1d];
        global_x_coords[0] = coord_min + global_h / two;
        global_y_coords[0] = coord_min + global_h / two;

        for i in 1..n_interp_1d {
            global_x_coords[i] = global_x_coords[i - 1] + global_h;
            global_y_coords[i] = global_y_coords[i - 1] + global_h;
        }

        // pre-compute circulant embedded kernel
        let n_fft = 2 * n_interp_1d;
        let fft_kernel = Self::precompute_kernel(&global_x_coords, &global_y_coords, n_fft);

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(n_fft);
        let fft_inverse = planner.plan_fft_inverse(n_fft);

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
            fft_forward,
            fft_inverse,
        }
    }

    /// Precompute and FFT the circulant kernel matrix
    ///
    /// Kernel: K(r) = (1 + ||r||²)⁻² (squared Student-t)
    ///
    /// ### Params
    ///
    /// * `x_coords` - X coordinates of the grid points
    /// * `y_coords` - Y coordinates of the grid points
    /// * `n_fft` - Number of FFT points
    ///
    /// ### Returns
    ///
    /// * `Vec<Complex<T>>` - FFT of the circulant kernel matrix
    fn precompute_kernel(x_coords: &[T], y_coords: &[T], n_fft: usize) -> Vec<Complex<T>> {
        let n_interp = x_coords.len();

        // evaluate kernel centred at (0, 0) position
        let x_0 = x_coords[0];
        let y_0 = y_coords[0];

        // build circulant embedding kernel
        let mut kernel_real = vec![T::zero(); n_fft * n_fft];

        for i in 0..n_interp {
            for j in 0..n_interp {
                let dx = x_coords[i] - x_0;
                let dy = y_coords[j] - y_0;
                let dist_sq = dx * dx + dy * dy;
                let k_val = T::one() / ((T::one() + dist_sq) * (T::one() + dist_sq));

                // circulant embedding: mirror in all directions
                kernel_real[(n_interp + i) * n_fft + (n_interp + j)] = k_val;
                kernel_real[(n_interp - i) * n_fft + (n_interp + j)] = k_val;
                kernel_real[(n_interp + i) * n_fft + (n_interp - j)] = k_val;
                kernel_real[(n_interp - i) * n_fft + (n_interp - j)] = k_val;
            }
        }

        // convert to complex
        let mut kernel_complex: Vec<Complex<T>> = kernel_real
            .iter()
            .map(|&x| Complex::new(x, T::zero()))
            .collect();

        // fft each row
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        for row in 0..n_fft {
            let start = row * n_fft;
            let end = start + n_fft;
            fft.process(&mut kernel_complex[start..end]);
        }

        // fft each column
        let mut col_buffer = vec![Complex::new(T::zero(), T::zero()); n_fft];
        for col in 0..n_fft {
            for row in 0..n_fft {
                col_buffer[row] = kernel_complex[row * n_fft + col];
            }
            fft.process(&mut col_buffer);
            for row in 0..n_fft {
                kernel_complex[row * n_fft + col] = col_buffer[row];
            }
        }

        kernel_complex
    }

    /// Determine which box a point belongs to
    ///
    /// ### Params
    ///
    /// * `x`: The x-coordinate of the point
    /// * `y`: The y-coordinate of the point
    ///
    /// ### Returns
    ///
    /// A tuple of the form `(y_idx, x_idx)` where `y_idx` is the index of the
    /// box in the y-direction and `x_idx` is the index of the box in the
    /// x-direction.
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

    /// Compute position within box in [0,1]
    ///
    /// ### Params
    ///
    /// * `coord`: The coordinate of the point
    /// * `box_idx`: The index of the box
    ///
    /// ### Returns
    ///
    /// The position of the point within the box in the range [0,1].
    #[inline]
    pub fn position_in_box(&self, coord: T, box_idx: usize) -> T {
        let box_min = self.coord_min + T::from_usize(box_idx).unwrap() * self.box_width;
        (coord - box_min) / self.box_width
    }

    /// Check if points are within grid bounds with margin
    ///
    /// Returns true if grid is valid, false if rebuild needed
    ///
    /// ### Params
    ///
    /// * `xs` - X coordinates of points to check
    /// * `ys` - Y coordinates of points to check
    /// * `margin` - Safety margin from grid edges
    ///
    /// ### Returns
    ///
    /// * `bool` - True if all points within bounds, false otherwise
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

    /// Compute bounds from embedding with padding
    ///
    /// ### Params
    ///
    /// * `embd` - Embedding points as [x, y] vectors
    /// * `padding_fraction` - Fraction of spread to add as padding
    ///
    /// ### Returns
    ///
    /// * `(T, T)` - Tuple of (min_coord, max_coord) with padding applied
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

    /// Create grid sized for given embedding
    ///
    /// ### Params
    ///
    /// * `embd` - Embedding points as [x, y] vectors
    /// * `n_interp_points` - Interpolation points per box (typically 3)
    /// * `intervals_per_integer` - Target intervals per integer coordinate
    /// * `min_intervals` - Minimum number of intervals
    ///
    /// ### Returns
    ///
    /// * `Self` - New FFT grid fitted to embedding bounds
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
pub fn lagrange_weights<T>(y_in_box: T, spacings: &[T], denominators: &[T], weights: &mut [T])
where
    T: Float,
{
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

/// Workspace for FFT computations.
///
/// Holds all temporary buffers needed for a given iteration.
///
/// ### Fields
///
/// * `w_coefficients`: Charge values on grid (before embedding). Size:
///   `n_interp_1d × n_interp_1d × n_terms`.
/// * `fft_input`: Embedded charges for FFT (zero-padded). Size:
///   `n_fft × n_fft (per term)`.
/// * `fft_output`: FFT output (complex). Size: `n_fft × n_fft`.
/// * `potentials`: Potential values on grid (after IFFT). Size:
///   `n_interp_1d × n_interp_1d × n_terms`.
/// * `point_boxes`: Point-to-box mapping.
/// * `x_weights`: Lagrange weight buffers (x direction).
/// * `y_weights`: Lagrange weight buffers (y direction).
pub struct FftWorkspace<T> {
    pub w_coefficients: Vec<T>,
    pub fft_input: Vec<T>,
    pub fft_output: Vec<Complex<T>>,
    pub potentials: Vec<T>,
    pub point_boxes: Vec<(usize, usize)>,
    pub x_weights: Vec<T>,
    pub y_weights: Vec<T>,
}

impl<T> FftWorkspace<T>
where
    T: Float + FromPrimitive,
{
    /// Create workspace for given grid and problem size
    ///
    /// ### Params
    ///
    /// * `n_points`: Number of points.
    /// * `n_terms`: Number of terms.
    /// * `grid`: FFT grid.
    ///
    /// ### Returns
    ///
    /// `Self` - FftWorkspace<T>
    pub fn new(n_points: usize, n_terms: usize, grid: &FftGrid<T>) -> Self {
        let n_interp_1d = grid.n_interp_points * grid.n_boxes_per_dim;
        let n_fft = 2 * n_interp_1d;
        let total_grid_points = n_interp_1d * n_interp_1d;

        Self {
            w_coefficients: vec![T::zero(); total_grid_points * n_terms],
            fft_input: vec![T::zero(); n_fft * n_fft],
            fft_output: vec![Complex::new(T::zero(), T::zero()); n_fft * n_fft],
            potentials: vec![T::zero(); total_grid_points * n_terms],
            point_boxes: vec![(0, 0); n_points],
            x_weights: vec![T::zero(); grid.n_interp_points * n_points],
            y_weights: vec![T::zero(); grid.n_interp_points * n_points],
        }
    }

    /// Clear workspace for next iteration
    ///
    /// Sets the w_coefficients and potentials to zero.
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

/// Interpolate point charges onto grid
///
/// Step 1; spreads each point's charges to nearby grid nodes using Lagrange
/// interpolation.
///
/// ### Params
///
/// * `xs` - X coordinates of points
/// * `ys` - Y coordinates of points
/// * `charges` - Charge values per point (size: n_points × n_terms)
/// * `n_terms` - Number of charge terms (typically 4)
/// * `grid` - Grid structure
/// * `workspace` - Pre-allocated workspace
fn interpolate_charges_to_grid<T>(
    xs: &[T],
    ys: &[T],
    charges: &[T],
    n_terms: usize,
    grid: &FftGrid<T>,
    workspace: &mut FftWorkspace<T>,
) where
    T: FftFloat,
{
    let n = xs.len();
    let n_interp = grid.n_interp_points;
    let n_boxes = grid.n_boxes_per_dim;

    // compute box assignments and positions within boxes
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

    // spread charges to grid
    let n_interp_1d = n_interp * n_boxes;

    for i in 0..n {
        let (box_y, box_x) = workspace.point_boxes[i];

        for interp_y in 0..n_interp {
            for interp_x in 0..n_interp {
                // Global grid index
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

/// Compute potentials via FFT convolution
///
/// Step 2 of the FFT convolution algorithm
///
/// For each charge term, performs:
/// 1. Embed charges into zero-padded array
/// 2. FFT(charges)
/// 3. Hadamard product with FFT(kernel)
/// 4. IFFT to get potentials
///
/// ### Params
///
/// * `n_terms` - Number of charge terms
/// * `grid` - Grid structure with precomputed kernel FFT
/// * `workspace` - Workspace with charges and output buffers
fn fft_convolution<T>(n_terms: usize, grid: &FftGrid<T>, workspace: &mut FftWorkspace<T>)
where
    T: FftFloat,
{
    let n_interp_1d = grid.n_interp_points * grid.n_boxes_per_dim;
    let n_fft = 2 * n_interp_1d;

    for term in 0..n_terms {
        // zero out FTT input
        for x in workspace.fft_input.iter_mut() {
            *x = T::zero();
        }

        // embed w_coefs
        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let w_idx = (i * n_interp_1d + j) * n_terms + term;
                let fft_idx = (n_interp_1d + i) * n_fft + (n_interp_1d + j);
                workspace.fft_input[fft_idx] = workspace.w_coefficients[w_idx];
            }
        }

        // conversion into complex numbers
        for (i, &val) in workspace.fft_input.iter().enumerate() {
            workspace.fft_output[i] = Complex::new(val, T::zero());
        }

        // fft rows
        for row in 0..n_fft {
            let start = row * n_fft;
            let end = start + n_fft;
            grid.fft_forward
                .process(&mut workspace.fft_output[start..end]);
        }

        // fft columns
        let mut col_buffer = vec![Complex::new(T::zero(), T::zero()); n_fft];
        for col in 0..n_fft {
            for row in 0..n_fft {
                col_buffer[row] = workspace.fft_output[row * n_fft + col];
            }
            grid.fft_forward.process(&mut col_buffer);
            for row in 0..n_fft {
                workspace.fft_output[row * n_fft + col] = col_buffer[row];
            }
        }

        // Hadamard product with kernel FFT
        for i in 0..n_fft * n_fft {
            workspace.fft_output[i] = workspace.fft_output[i] * grid.fft_kernel[i];
        }

        // IFFT columns
        for col in 0..n_fft {
            for row in 0..n_fft {
                col_buffer[row] = workspace.fft_output[row * n_fft + col];
            }
            grid.fft_inverse.process(&mut col_buffer);
            for row in 0..n_fft {
                workspace.fft_output[row * n_fft + col] = col_buffer[row];
            }
        }

        // IFFT rows
        for row in 0..n_fft {
            let start = row * n_fft;
            let end = start + n_fft;
            grid.fft_inverse
                .process(&mut workspace.fft_output[start..end]);
        }

        // Extract potentials (first quadrant only, with normalization)
        let norm = T::from_usize(n_fft * n_fft).unwrap();
        for i in 0..n_interp_1d {
            for j in 0..n_interp_1d {
                let fft_idx = i * n_fft + j;
                let pot_idx = (i * n_interp_1d + j) * n_terms + term;
                workspace.potentials[pot_idx] = workspace.fft_output[fft_idx].re / norm;
            }
        }
    }
}

/// Interpolate potentials from grid back to points
///
/// For step 3 for each point, gathers potential values from nearby grid nodes
/// using the same Lagrange weights as forward interpolation
///
/// ### Params
///
/// * `xs` - X coordinates of points
/// * `potentials_out` - Output buffer (size: n_points × n_terms)
/// * `n_terms` - Number of potential terms
/// * `grid` - Grid structure
/// * `workspace` - Workspace with precomputed potentials
fn interpolate_potentials_to_points<T>(
    potentials_out: &mut [T],
    n_terms: usize,
    grid: &FftGrid<T>,
    workspace: &FftWorkspace<T>,
) where
    T: FftFloat,
{
    let n = workspace.point_boxes.len();
    let n_interp = grid.n_interp_points;
    let n_interp_1d = n_interp * grid.n_boxes_per_dim;

    // zero output
    for x in potentials_out.iter_mut() {
        *x = T::zero();
    }

    // gather from grid
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

    // Round up to FFT-friendly sizes (powers of 2, 3, 5, 7)
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
pub fn compute_repulsive_forces_fft<T>(
    embd: &[Vec<T>],
    grid: &FftGrid<T>,
    workspace: &mut FftWorkspace<T>,
) -> (Vec<T>, Vec<T>, T)
where
    T: FftFloat,
{
    let n = embd.len();
    let n_terms = 4;

    // extract the coordinates
    let xs: Vec<T> = embd.iter().map(|p| p[0]).collect();
    let ys: Vec<T> = embd.iter().map(|p| p[1]).collect();

    // prepare charge terms: [1, x, y, x² + y²]
    let mut charges = vec![T::zero(); n * n_terms];
    for i in 0..n {
        charges[i * n_terms] = T::one();
        charges[i * n_terms + 1] = xs[i];
        charges[i * n_terms + 2] = ys[i];
        charges[i * n_terms + 3] = xs[i] * xs[i] + ys[i] * ys[i];
    }

    // clear workspace from previous iteration
    workspace.clear();

    // step 1: Interpolate charges to grid
    interpolate_charges_to_grid(&xs, &ys, &charges, n_terms, grid, workspace);

    // step 2: FFT convolution
    fft_convolution(n_terms, grid, workspace);

    // step 3: Interpolate potentials back to points
    let mut potentials = vec![T::zero(); n * n_terms];
    interpolate_potentials_to_points(&mut potentials, n_terms, grid, workspace);

    // compute normalisation constant Z
    let mut sum_q = T::zero();
    for i in 0..n {
        let phi1 = potentials[i * n_terms];
        let phi2 = potentials[i * n_terms + 1];
        let phi3 = potentials[i * n_terms + 2];
        let phi4 = potentials[i * n_terms + 3];

        sum_q = sum_q + (T::one() + xs[i] * xs[i] + ys[i] * ys[i]) * phi1
            - (T::one() + T::one()) * (xs[i] * phi2 + ys[i] * phi3)
            + phi4;
    }
    sum_q = sum_q - T::from_usize(n).unwrap();

    // compute repulsive forces
    let mut forces_x = vec![T::zero(); n];
    let mut forces_y = vec![T::zero(); n];

    let z_inv = if sum_q.abs() > T::epsilon() {
        T::one() / sum_q
    } else {
        T::zero()
    };

    for i in 0..n {
        let phi1 = potentials[i * n_terms];
        let phi2 = potentials[i * n_terms + 1];
        let phi3 = potentials[i * n_terms + 2];

        forces_x[i] = (xs[i] * phi1 - phi2) * z_inv;
        forces_y[i] = (ys[i] * phi1 - phi3) * z_inv;
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

    /// Direct O(N²) computation of kernel sums for validation
    fn direct_kernel_potentials(
        xs: &[f64],
        ys: &[f64],
        charges: &[f64],
        n_terms: usize,
    ) -> Vec<f64> {
        let n = xs.len();
        let mut potentials = vec![0.0; n * n_terms];

        for i in 0..n {
            for j in 0..n {
                let dx = xs[i] - xs[j];
                let dy = ys[i] - ys[j];
                let dist_sq = dx * dx + dy * dy;
                let kernel = 1.0 / ((1.0 + dist_sq) * (1.0 + dist_sq));

                for term in 0..n_terms {
                    potentials[i * n_terms + term] += kernel * charges[j * n_terms + term];
                }
            }
        }

        potentials
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
        // Random-ish embedding
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

        // Direct computation
        let (direct_fx, direct_fy, direct_z) = direct_repulsive_forces(&embd);

        // FFT computation
        let coord_min = -5.0;
        let coord_max = 5.0;
        let grid = FftGrid::<f64>::new(coord_min, coord_max, 10, 3);
        let mut workspace = FftWorkspace::new(n, 4, &grid);

        let (fft_fx, fft_fy, fft_z) = compute_repulsive_forces_fft(&embd, &grid, &mut workspace);

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
    fn test_kernel_symmetry() {
        // Kernel should be symmetric: K(i,j) = K(j,i)
        let grid = FftGrid::<f64>::new(-5.0, 5.0, 4, 3);
        let n_fft = 2 * grid.n_interp_points * grid.n_boxes_per_dim;

        for i in 0..n_fft {
            for j in 0..i {
                let k_ij = grid.fft_kernel[i * n_fft + j];
                let k_ji = grid.fft_kernel[j * n_fft + i];
                let diff = (k_ij - k_ji).norm();
                assert!(
                    diff < 1e-10,
                    "Kernel asymmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    k_ij,
                    k_ji
                );
            }
        }
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

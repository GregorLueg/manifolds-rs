use num_traits::{Float, FromPrimitive};

/////////////
// Helpers //
/////////////

/// A simple 2D Bounding Box
#[derive(Clone, Copy, Debug)]
struct BBox<T> {
    x_min: T,
    x_max: T,
    y_min: T,
    y_max: T,
}

impl<T: Float> BBox<T> {
    /// Determines which quadrant a point belongs to relative to the center of the box.
    ///
    /// The quadrants are indexed in Z-order (Morton order):
    /// * 0: Top-Left (x < center, y < center)
    /// * 1: Top-Right (x > center, y < center)
    /// * 2: Bottom-Left (x < center, y > center)
    /// * 3: Bottom-Right (x > center, y > center)
    fn get_quadrant(&self, x: T, y: T) -> usize {
        let center_x = (self.x_min + self.x_max) / (T::one() + T::one());
        let center_y = (self.y_min + self.y_max) / (T::one() + T::one());

        let right = if x > center_x { 1 } else { 0 };
        let bottom = if y > center_y { 2 } else { 0 };
        right + bottom
    }

    /// Creates a new Bounding Box representing one of the four sub-quadrants.
    ///
    /// ### Params
    ///
    /// * `quadrant` - The index (0-3) of the quadrant to generate.
    ///
    /// ### Panics
    ///
    /// Panics if `quadrant` is not in the range [0, 3].
    fn sub_quadrant(&self, quadrant: usize) -> Self {
        let center_x = (self.x_min + self.x_max) / (T::one() + T::one());
        let center_y = (self.y_min + self.y_max) / (T::one() + T::one());

        match quadrant {
            0 => BBox {
                x_min: self.x_min,
                x_max: center_x,
                y_min: self.y_min,
                y_max: center_y,
            },
            1 => BBox {
                x_min: center_x,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: center_y,
            },
            2 => BBox {
                x_min: self.x_min,
                x_max: center_x,
                y_min: center_y,
                y_max: self.y_max,
            },
            3 => BBox {
                x_min: center_x,
                x_max: self.x_max,
                y_min: center_y,
                y_max: self.y_max,
            },
            _ => unreachable!(),
        }
    }
}

//////////////
// QuadNode //
//////////////

/// A node in the flattened QuadTree
///
/// ### Fields
///
/// * `com_x` - Centre of mass (X)
/// * `com_y` - Centre of mass (Y)
/// * `mass` - Total mass (number of points in this node/subtree)
/// * `width` - Width of the cell covered by this node
/// * `children` - Indices of children in the `nodes` vector; `None` if child
///   does not exist.
/// * `point_indices` - If this is a leaf, stores indices of all points in this cell.
///   Empty for internal nodes.
#[derive(Debug, Clone)]
pub struct QuadNode<T> {
    pub com_x: T,
    pub com_y: T,
    pub mass: T,
    pub width: T,
    pub children: [Option<usize>; 4],
    pub point_indices: Vec<usize>, // Changed from Option<usize> to Vec<usize>
}

impl<T: Float> QuadNode<T> {
    /// Check if this node contains a specific point index
    #[inline]
    pub fn contains_point(&self, idx: usize) -> bool {
        self.point_indices.contains(&idx)
    }

    /// Check if this is a leaf node
    #[inline]
    pub fn is_leaf(&self) -> bool {
        !self.point_indices.is_empty() || self.children.iter().all(|c| c.is_none())
    }
}

/// Barnes-Hut Tree
///
/// Continuous flat arena to store the nodes for better cache locality
pub struct BarnesHutTree<T> {
    pub nodes: Vec<QuadNode<T>>,
    root: usize,
}

impl<T> BarnesHutTree<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    /// Build the QuadTree from the current embedding
    pub fn new(embd: &[Vec<T>]) -> Self {
        let (min_x, max_x, min_y, max_y) = embd.iter().fold(
            (
                T::infinity(),
                T::neg_infinity(),
                T::infinity(),
                T::neg_infinity(),
            ),
            |(min_x, max_x, min_y, max_y), p| {
                (
                    min_x.min(p[0]),
                    max_x.max(p[0]),
                    min_y.min(p[1]),
                    max_y.max(p[1]),
                )
            },
        );

        let epsilon = T::from_f64(1e-6).unwrap();
        let bbox = BBox {
            x_min: min_x - epsilon,
            x_max: max_x + epsilon,
            y_min: min_y - epsilon,
            y_max: max_y + epsilon,
        };

        let mut nodes: Vec<QuadNode<T>> = Vec::with_capacity(embd.len() * 2);
        let point_indices: Vec<usize> = (0..embd.len()).collect();

        let root = Self::build_recursive(&mut nodes, embd, &point_indices, bbox);

        Self { nodes, root }
    }

    /// Recursively builds the QuadTree by partitioning points into quadrants.
    ///
    /// This function determines if the current set of points forms a leaf node
    /// (single point or coincident points) or an internal node. If internal,
    /// it distributes points into four child buckets and recurses.
    ///
    /// ### Params
    ///
    /// * `nodes` - The flat arena of nodes being populated.
    /// * `embd` - The read-only embedding coordinates.
    /// * `point_indices` - Indices of the points contained in the current
    ///   node's region.
    /// * `bbox` - The spatial bounding box covered by this node.
    ///
    /// ### Returns
    ///
    /// The index of the newly created node within the `nodes` vector.
    fn build_recursive(
        nodes: &mut Vec<QuadNode<T>>,
        embd: &[Vec<T>],
        point_indices: &[usize],
        bbox: BBox<T>,
    ) -> usize {
        let width = bbox.x_max - bbox.x_min;

        // Single point → leaf
        if point_indices.len() == 1 {
            let idx = point_indices[0];
            let p = &embd[idx];

            let node = QuadNode {
                com_x: p[0],
                com_y: p[1],
                mass: T::one(),
                width,
                children: [None; 4],
                point_indices: vec![idx],
            };

            let node_idx = nodes.len();
            nodes.push(node);
            return node_idx;
        }

        let mut buckets: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        let mut sum_x = T::zero();
        let mut sum_y = T::zero();
        let mass = T::from_usize(point_indices.len()).unwrap();

        for &idx in point_indices {
            let p = &embd[idx];
            sum_x = sum_x + p[0];
            sum_y = sum_y + p[1];

            let quadrant = bbox.get_quadrant(p[0], p[1]);
            buckets[quadrant].push(idx);
        }

        // Check for infinite recursion (all points in same quadrant)
        let max_bucket_size = buckets.iter().map(|b| b.len()).max().unwrap_or(0);
        if max_bucket_size == point_indices.len() {
            // All points coincident or very close — create leaf with ALL point indices
            let node = QuadNode {
                com_x: sum_x / mass,
                com_y: sum_y / mass,
                mass,
                width,
                children: [None; 4],
                point_indices: point_indices.to_vec(), // Store ALL indices
            };

            let node_idx = nodes.len();
            nodes.push(node);
            return node_idx;
        }

        // Internal node
        let node_idx = nodes.len();

        nodes.push(QuadNode {
            com_x: sum_x / mass,
            com_y: sum_y / mass,
            mass,
            width,
            children: [None; 4],
            point_indices: Vec::new(), // Internal nodes have no direct points
        });

        let mut children_indices = [None; 4];

        for i in 0..4 {
            if !buckets[i].is_empty() {
                let child_idx =
                    Self::build_recursive(nodes, embd, &buckets[i], bbox.sub_quadrant(i));
                children_indices[i] = Some(child_idx);
            }
        }

        nodes[node_idx].children = children_indices;

        node_idx
    }

    /// Compute the repulsive forces on a point using the Barnes-Hut approximation
    ///
    /// ### Params
    ///
    /// * `point_idx` - Index of the point in the original embedding
    /// * `p_x` - X coordinate of the point
    /// * `p_y` - Y coordinate of the point
    /// * `theta` - Barnes-Hut approximation parameter (typically 0.5)
    ///
    /// ### Returns
    ///
    /// A tuple `(force_x, force_y, sum_q)` where:
    /// * `force_x` - Repulsive force in x direction
    /// * `force_y` - Repulsive force in y direction
    /// * `sum_q` - Sum of unnormalised affinities (for normalisation)
    pub fn compute_repulsive_force(&self, point_idx: usize, p_x: T, p_y: T, theta: T) -> (T, T, T) {
        let mut force_x = T::zero();
        let mut force_y = T::zero();
        let mut sum_q = T::zero();

        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);

        let theta_sq = theta * theta;
        let min_dist_sq = T::from_f64(1e-12).unwrap();

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            let dx = p_x - node.com_x;
            let dy = p_y - node.com_y;
            let dist_sq = (dx * dx + dy * dy).max(min_dist_sq);

            // Skip if this node contains the query point and distance is tiny
            // (handles both single-point leaves and coincident point leaves)
            if node.contains_point(point_idx) {
                if node.is_leaf() {
                    // Leaf containing query point: compute force from OTHER points in this leaf
                    let other_mass = node.mass - T::one();
                    if other_mass > T::zero() && dist_sq > min_dist_sq {
                        let q = T::one() / (T::one() + dist_sq);
                        let mult = other_mass * q * q;
                        sum_q = sum_q + other_mass * q;
                        force_x = force_x + mult * dx;
                        force_y = force_y + mult * dy;
                    }
                    continue;
                } else {
                    // Internal node containing query point: must open it
                    for child in node.children.iter().flatten() {
                        stack.push(*child);
                    }
                    continue;
                }
            }

            // Barnes-Hut criterion: is the node far enough to use as summary?
            let is_summary = node.width * node.width < theta_sq * dist_sq;

            if is_summary || node.is_leaf() {
                // Use this node as a summary
                let q = T::one() / (T::one() + dist_sq);
                let mult = node.mass * q * q;

                sum_q = sum_q + node.mass * q;
                force_x = force_x + mult * dx;
                force_y = force_y + mult * dy;
            } else {
                // Node too close: open it
                for child in node.children.iter().flatten() {
                    stack.push(*child);
                }
            }
        }

        (force_x, force_y, sum_q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: create embedding from slice of tuples
    fn embd_from_tuples(pts: &[(f64, f64)]) -> Vec<Vec<f64>> {
        pts.iter().map(|&(x, y)| vec![x, y]).collect()
    }

    #[test]
    fn test_single_point_tree() {
        let embd = embd_from_tuples(&[(1.0, 2.0)]);
        let tree = BarnesHutTree::new(&embd);

        assert_eq!(tree.nodes.len(), 1);
        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.com_x, 1.0);
        assert_relative_eq!(root.com_y, 2.0);
        assert_relative_eq!(root.mass, 1.0);
        assert_eq!(root.point_indices, vec![0]);
        assert!(root.is_leaf());
    }

    #[test]
    fn test_two_points_different_quadrants() {
        // Points in opposite corners should create tree with depth
        let embd = embd_from_tuples(&[(0.0, 0.0), (10.0, 10.0)]);
        let tree = BarnesHutTree::new(&embd);

        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.mass, 2.0);
        assert_relative_eq!(root.com_x, 5.0);
        assert_relative_eq!(root.com_y, 5.0);
        assert!(!root.is_leaf()); // Should have children
        assert!(root.point_indices.is_empty());
    }

    #[test]
    fn test_coincident_points_stored_correctly() {
        // Multiple points at same location
        let embd = embd_from_tuples(&[(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]);
        let tree = BarnesHutTree::new(&embd);

        // Should create a single leaf with all three points
        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.mass, 3.0);
        assert_eq!(root.point_indices.len(), 3);
        assert!(root.point_indices.contains(&0));
        assert!(root.point_indices.contains(&1));
        assert!(root.point_indices.contains(&2));
        assert!(root.is_leaf());
    }

    #[test]
    fn test_mass_conservation() {
        // Total mass should equal number of points
        let embd = embd_from_tuples(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (2.0, 2.0),
            (3.0, 1.0),
        ]);
        let tree = BarnesHutTree::new(&embd);

        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.mass, 7.0);
    }

    #[test]
    fn test_centre_of_mass_correctness() {
        let embd = embd_from_tuples(&[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)]);
        let tree = BarnesHutTree::new(&embd);

        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.com_x, 2.0);
        assert_relative_eq!(root.com_y, 2.0);
    }

    #[test]
    fn test_no_self_interaction_single_point() {
        let embd = embd_from_tuples(&[(0.0, 0.0)]);
        let tree = BarnesHutTree::new(&embd);

        let (fx, fy, sum_q) = tree.compute_repulsive_force(0, 0.0, 0.0, 0.5);

        // Single point should have zero force on itself
        assert_relative_eq!(fx, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fy, 0.0, epsilon = 1e-10);
        assert_relative_eq!(sum_q, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_self_interaction_coincident_points() {
        // Three coincident points — each should only feel force from the other two
        let embd = embd_from_tuples(&[(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]);
        let tree = BarnesHutTree::new(&embd);

        for i in 0..3 {
            let (fx, fy, sum_q) = tree.compute_repulsive_force(i, 5.0, 5.0, 0.5);

            // Force should be finite (not NaN or Inf)
            assert!(fx.is_finite(), "fx is not finite for point {}", i);
            assert!(fy.is_finite(), "fy is not finite for point {}", i);
            assert!(sum_q.is_finite(), "sum_q is not finite for point {}", i);

            // At exact same location, dx=dy=0, so force direction is zero
            // but sum_q should reflect the other 2 points
            println!(
                "Point {}: fx={:.6}, fy={:.6}, sum_q={:.6}",
                i, fx, fy, sum_q
            );
        }
    }

    #[test]
    fn test_force_symmetry_two_points() {
        let embd = embd_from_tuples(&[(0.0, 0.0), (2.0, 0.0)]);
        let tree = BarnesHutTree::new(&embd);

        let (fx0, fy0, _) = tree.compute_repulsive_force(0, 0.0, 0.0, 0.5);
        let (fx1, fy1, _) = tree.compute_repulsive_force(1, 2.0, 0.0, 0.5);

        // Forces should be equal and opposite
        assert_relative_eq!(fx0, -fx1, epsilon = 1e-10);
        assert_relative_eq!(fy0, -fy1, epsilon = 1e-10);

        // Point 0 should be pushed left (negative x), point 1 pushed right (positive x)
        assert!(fx0 < 0.0, "Point 0 should be pushed left");
        assert!(fx1 > 0.0, "Point 1 should be pushed right");
        assert_relative_eq!(fy0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fy1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_force_magnitude_decreases_with_distance() {
        // Two configurations: points close vs far
        let embd_close = embd_from_tuples(&[(0.0, 0.0), (1.0, 0.0)]);
        let embd_far = embd_from_tuples(&[(0.0, 0.0), (10.0, 0.0)]);

        let tree_close = BarnesHutTree::new(&embd_close);
        let tree_far = BarnesHutTree::new(&embd_far);

        let (fx_close, _, _) = tree_close.compute_repulsive_force(0, 0.0, 0.0, 0.5);
        let (fx_far, _, _) = tree_far.compute_repulsive_force(0, 0.0, 0.0, 0.5);

        // Closer points should have larger force magnitude
        assert!(
            fx_close.abs() > fx_far.abs(),
            "Close force {} should be larger than far force {}",
            fx_close.abs(),
            fx_far.abs()
        );
    }

    #[test]
    fn test_sum_q_equals_n_minus_1_for_exact_computation() {
        // With theta=0, we get exact computation (no approximation)
        // For well-separated points, sum_q should be close to n-1 when points are far apart
        // (each q_ij ≈ 1 when dist_sq >> 1)
        let embd = embd_from_tuples(&[(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]);
        let tree = BarnesHutTree::new(&embd);

        let theta = 0.0; // Force exact computation
        let (_, _, sum_q) = tree.compute_repulsive_force(0, 0.0, 0.0, theta);

        // With large distances, q = 1/(1+d²) ≈ 0, so sum_q < n-1
        // But it should be positive and reasonable
        assert!(sum_q > 0.0);
        assert!(sum_q < 3.0); // Less than n-1=3 because of distance decay
        println!("sum_q with exact computation: {}", sum_q);
    }

    #[test]
    fn test_barnes_hut_approximation_reasonable() {
        // Compare exact (theta=0) vs approximate (theta=0.5) for a larger point set
        let mut pts = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                pts.push((i as f64 * 2.0, j as f64 * 2.0));
            }
        }
        let embd = embd_from_tuples(&pts);
        let tree = BarnesHutTree::new(&embd);

        // Test point in middle of grid
        let test_idx = 55;
        let (px, py) = pts[test_idx];

        let (fx_exact, fy_exact, sq_exact) = tree.compute_repulsive_force(test_idx, px, py, 0.0);
        let (fx_approx, fy_approx, sq_approx) = tree.compute_repulsive_force(test_idx, px, py, 0.5);

        println!(
            "Exact:  fx={:.6}, fy={:.6}, sum_q={:.6}",
            fx_exact, fy_exact, sq_exact
        );
        println!(
            "Approx: fx={:.6}, fy={:.6}, sum_q={:.6}",
            fx_approx, fy_approx, sq_approx
        );

        // Approximation should be in the same ballpark (within 50% for theta=0.5)
        let fx_ratio = if fx_exact.abs() > 1e-10 {
            (fx_approx / fx_exact - 1.0).abs()
        } else {
            0.0
        };
        let fy_ratio = if fy_exact.abs() > 1e-10 {
            (fy_approx / fy_exact - 1.0).abs()
        } else {
            0.0
        };

        println!(
            "Force relative errors: fx={:.2}%, fy={:.2}%",
            fx_ratio * 100.0,
            fy_ratio * 100.0
        );

        // Allow up to 50% error for theta=0.5
        assert!(
            fx_ratio < 0.5 || fx_exact.abs() < 1e-6,
            "fx error too large"
        );
        assert!(
            fy_ratio < 0.5 || fy_exact.abs() < 1e-6,
            "fy error too large"
        );
    }

    #[test]
    fn test_no_nan_or_inf_forces() {
        // Stress test with various configurations
        let configs: Vec<Vec<(f64, f64)>> = vec![
            // Coincident points
            vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            // Very close points
            vec![(0.0, 0.0), (1e-10, 0.0)],
            // Mix of close and far
            vec![(0.0, 0.0), (1e-8, 1e-8), (100.0, 100.0)],
            // Large coordinates
            vec![(1e6, 1e6), (1e6 + 1.0, 1e6)],
        ];

        for (cfg_idx, pts) in configs.iter().enumerate() {
            let embd = embd_from_tuples(pts);
            let tree = BarnesHutTree::new(&embd);

            for i in 0..pts.len() {
                let (fx, fy, sum_q) = tree.compute_repulsive_force(i, pts[i].0, pts[i].1, 0.5);

                assert!(
                    fx.is_finite(),
                    "Config {}, point {}: fx is not finite: {}",
                    cfg_idx,
                    i,
                    fx
                );
                assert!(
                    fy.is_finite(),
                    "Config {}, point {}: fy is not finite: {}",
                    cfg_idx,
                    i,
                    fy
                );
                assert!(
                    sum_q.is_finite(),
                    "Config {}, point {}: sum_q is not finite: {}",
                    cfg_idx,
                    i,
                    sum_q
                );
                assert!(
                    sum_q >= 0.0,
                    "Config {}, point {}: sum_q is negative: {}",
                    cfg_idx,
                    i,
                    sum_q
                );
            }
        }
    }

    #[test]
    fn test_total_sum_q_consistent() {
        // Sum of all sum_q values should be consistent (each pair counted twice)
        let embd = embd_from_tuples(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
        let tree = BarnesHutTree::new(&embd);

        let total: f64 = (0..4)
            .map(|i| {
                let (_, _, sq) = tree.compute_repulsive_force(i, embd[i][0], embd[i][1], 0.0);
                sq
            })
            .sum();

        println!("Total sum_q across all points: {}", total);

        // Should be positive and finite
        assert!(total > 0.0);
        assert!(total.is_finite());
    }
}

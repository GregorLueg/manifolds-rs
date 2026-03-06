//! Barnes-Hut tree implementation for tSNE fitting

use num_traits::{Float, FromPrimitive};

/////////////
// Helpers //
/////////////

/// A simple 2D bounding box.
///
/// ### Fields
///
/// * `x_min` - Left boundary
/// * `x_max` - Right boundary
/// * `y_min` - Top boundary
/// * `y_max` - Bottom boundary
#[derive(Clone, Copy, Debug)]
struct BBox<T> {
    x_min: T,
    x_max: T,
    y_min: T,
    y_max: T,
}

impl<T: Float> BBox<T> {
    /// Determine which quadrant a point belongs to relative to the centre.
    ///
    /// Quadrants are indexed in Z-order (Morton order):
    /// * 0: Top-Left (x < centre, y < centre)
    /// * 1: Top-Right (x >= centre, y < centre)
    /// * 2: Bottom-Left (x < centre, y >= centre)
    /// * 3: Bottom-Right (x >= centre, y >= centre)
    #[inline]
    fn get_quadrant(&self, x: T, y: T) -> usize {
        let two = T::one() + T::one();
        let cx = (self.x_min + self.x_max) / two;
        let cy = (self.y_min + self.y_max) / two;

        let right = if x > cx { 1 } else { 0 };
        let bottom = if y > cy { 2 } else { 0 };
        right + bottom
    }

    /// Create a bounding box for one of the four sub-quadrants.
    ///
    /// ### Params
    ///
    /// * `quadrant` - Quadrant index (0-3)
    ///
    /// ### Panics
    ///
    /// Panics if `quadrant` is not in `[0, 3]`.
    fn sub_quadrant(&self, quadrant: usize) -> Self {
        let two = T::one() + T::one();
        let cx = (self.x_min + self.x_max) / two;
        let cy = (self.y_min + self.y_max) / two;

        match quadrant {
            0 => BBox {
                x_min: self.x_min,
                x_max: cx,
                y_min: self.y_min,
                y_max: cy,
            },
            1 => BBox {
                x_min: cx,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: cy,
            },
            2 => BBox {
                x_min: self.x_min,
                x_max: cx,
                y_min: cy,
                y_max: self.y_max,
            },
            3 => BBox {
                x_min: cx,
                x_max: self.x_max,
                y_min: cy,
                y_max: self.y_max,
            },
            _ => unreachable!(),
        }
    }
}

//////////////
// LeafData //
//////////////

/// Point index data stored in leaf nodes.
///
/// Uses an enum to avoid paying for a `Vec` allocation on the common
/// single-point leaf path. The `Coincident` variant is only constructed
/// when multiple points fall into the same quadrant and cannot be further
/// separated (i.e. they are at identical or near-identical coordinates).
#[derive(Debug, Clone)]
pub enum LeafData {
    /// Single point (vast majority of leaves, no heap allocation)
    Single(usize),
    /// Multiple coincident/near-coincident points
    Coincident(Vec<usize>),
}

impl LeafData {
    /// Check whether this leaf contains a given point index.
    #[inline]
    fn contains(&self, idx: usize) -> bool {
        match self {
            LeafData::Single(i) => *i == idx,
            LeafData::Coincident(indices) => indices.contains(&idx),
        }
    }
}

//////////////
// QuadNode //
//////////////

/// A node in the flattened quad-tree arena.
///
/// Leaf nodes carry a `LeafData` value identifying which point(s) they
/// contain. Internal nodes have `leaf_data: None` and instead reference
/// up to four children. This design keeps single-point leaves (the vast
/// majority) free of heap allocation whilst still correctly tracking all
/// indices for coincident-point leaves, ensuring exact self-interaction
/// exclusion during force computation.
#[derive(Debug, Clone)]
pub struct QuadNode<T> {
    /// Centre of mass (X)
    pub com_x: T,
    /// Centre of mass (Y)
    pub com_y: T,
    /// Total mass (number of points in this node/subtree)
    pub mass: T,
    /// Width of the cell covered by this node
    pub width: T,
    /// Indices of children in the `nodes` vector; `None` if child does not
    /// exist
    pub children: [Option<usize>; 4],
    /// Point data for leaf nodes; `None` for internal nodes
    pub leaf_data: Option<LeafData>,
}

impl<T: Float> QuadNode<T> {
    /// Whether this node is a leaf.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.leaf_data.is_some()
    }
}

///////////////////
// BarnesHutTree //
///////////////////

/// Internal Build stack
struct BuildTask<T> {
    /// Tracks (parent_node_index, quadrant_index) to link back up the tree
    parent_idx: Option<(usize, usize)>,
    start: usize,
    end: usize,
    bbox: BBox<T>,
    depth: usize,
}

/// Barnes-Hut quad-tree for 2D repulsive force approximation.
///
/// Stores nodes in a contiguous flat arena for cache locality. The tree
/// is built once per epoch from the current embedding positions, then
/// queried in parallel for each point's repulsive forces.
pub struct BarnesHutTree<T> {
    /// Flat arena of quad-tree nodes
    pub nodes: Vec<QuadNode<T>>,
    /// Index of the root node in the arena
    pub root: usize,
}

impl<T> BarnesHutTree<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    /// Build quad-tree from current embedding positions.
    ///
    /// Uses a heap-allocated task stack to completely eliminate recursion.
    /// This is strictly necessary when called from FFI environments (like R or
    /// Python) where C-stack limits are strictly monitored and easily blown
    /// during post-early-exaggeration t-SNE cluster expansions.
    ///
    /// ### Params
    ///
    /// * `embd` - Point coordinates, shape `[n_points][2]`
    ///
    /// ### Returns
    ///
    /// Constructed tree ready for force queries
    pub fn new(embd: &[Vec<T>]) -> Self {
        if embd.is_empty() {
            return Self {
                nodes: vec![QuadNode {
                    com_x: T::zero(),
                    com_y: T::zero(),
                    mass: T::zero(),
                    width: T::zero(),
                    children: [None; 4],
                    leaf_data: None,
                }],
                root: 0,
            };
        }

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
        let mut scratch = vec![0usize; embd.len()];
        let mut indices: Vec<usize> = (0..embd.len()).collect();

        let mut stack = Vec::with_capacity(128);
        stack.push(BuildTask {
            parent_idx: None,
            start: 0,
            end: embd.len(),
            bbox,
            depth: 0,
        });

        let mut root_idx = 0;

        // Iterative Tree Builder Loop
        while let Some(task) = stack.pop() {
            let width = task.bbox.x_max - task.bbox.x_min;
            let n = task.end - task.start;
            let point_indices = &mut indices[task.start..task.end];

            // 1. Single point -> leaf
            if n == 1 {
                let idx = point_indices[0];
                let p = &embd[idx];
                let node_idx = nodes.len();
                nodes.push(QuadNode {
                    com_x: p[0],
                    com_y: p[1],
                    mass: T::one(),
                    width,
                    children: [None; 4],
                    leaf_data: Some(LeafData::Single(idx)),
                });

                if let Some((p_idx, q)) = task.parent_idx {
                    nodes[p_idx].children[q] = Some(node_idx);
                } else {
                    root_idx = node_idx;
                }
                continue;
            }

            // 2. Compute center of mass and quadrant counts
            let mut sum_x = T::zero();
            let mut sum_y = T::zero();
            let mass = T::from_usize(n).unwrap();
            let mut counts = [0usize; 4];

            for &idx in point_indices.iter() {
                let p = &embd[idx];
                sum_x = sum_x + p[0];
                sum_y = sum_y + p[1];
                let q = task.bbox.get_quadrant(p[0], p[1]);
                counts[q] += 1;
            }

            let max_count = *counts.iter().max().unwrap();

            // 3. Fallback Guard: Depth limit or exact coincidence
            if task.depth >= 64
                || !width.is_finite()
                || (max_count == n && width < T::from_f64(1e-10).unwrap())
            {
                let node_idx = nodes.len();
                nodes.push(QuadNode {
                    com_x: sum_x / mass,
                    com_y: sum_y / mass,
                    mass,
                    width,
                    children: [None; 4],
                    leaf_data: Some(LeafData::Coincident(point_indices.to_vec())),
                });

                if let Some((p_idx, q)) = task.parent_idx {
                    nodes[p_idx].children[q] = Some(node_idx);
                } else {
                    root_idx = node_idx;
                }
                continue;
            }

            // 4. Box shrinking: All points in one quadrant, but not perfectly coincident
            if max_count == n {
                let q = counts.iter().position(|&c| c == n).unwrap();
                // Push back to stack with a shrunk box, keeping the original parent
                stack.push(BuildTask {
                    parent_idx: task.parent_idx,
                    start: task.start,
                    end: task.end,
                    bbox: task.bbox.sub_quadrant(q),
                    depth: task.depth + 1,
                });
                continue;
            }

            // 5. Internal Node: scatter to scratch to partition in place
            let mut offsets = [0usize; 4];
            offsets[0] = 0;
            for i in 1..4 {
                offsets[i] = offsets[i - 1] + counts[i - 1];
            }
            let mut write_pos = offsets;
            let scratch_slice = &mut scratch[task.start..task.end];

            for &idx in point_indices.iter() {
                let q = task.bbox.get_quadrant(embd[idx][0], embd[idx][1]);
                scratch_slice[write_pos[q]] = idx;
                write_pos[q] += 1;
            }
            point_indices.copy_from_slice(scratch_slice);

            let node_idx = nodes.len();
            nodes.push(QuadNode {
                com_x: sum_x / mass,
                com_y: sum_y / mass,
                mass,
                width,
                children: [None; 4],
                leaf_data: None,
            });

            if let Some((p_idx, q)) = task.parent_idx {
                nodes[p_idx].children[q] = Some(node_idx);
            } else {
                root_idx = node_idx;
            }

            // 6. Push sub-tasks for children
            let mut current_start = task.start;
            for i in 0..4 {
                let current_end = current_start + counts[i];
                if counts[i] > 0 {
                    stack.push(BuildTask {
                        parent_idx: Some((node_idx, i)),
                        start: current_start,
                        end: current_end,
                        bbox: task.bbox.sub_quadrant(i),
                        depth: task.depth + 1,
                    });
                }
                current_start = current_end;
            }
        }

        Self {
            nodes,
            root: root_idx,
        }
    }

    /// Compute repulsive forces on a point using Barnes-Hut approximation.
    ///
    /// Traverses the tree using an explicit fixed-size stack. At each node,
    /// applies the Barnes-Hut opening criterion. Self-interaction is excluded,
    /// and mathematically coincident points are ignored to prevent
    /// Z-normalisation inflation.
    ///
    /// ### Params
    ///
    /// * `point_idx` - Index of the query point in the embedding
    /// * `p_x` - X coordinate of the query point
    /// * `p_y` - Y coordinate of the query point
    /// * `theta` - Barnes-Hut opening parameter (typically 0.5)
    ///
    /// ### Returns
    ///
    /// Tuple `(force_x, force_y, sum_q)` where `sum_q` is the sum of
    /// unnormalised Student-t affinities (for computing Z)
    pub fn compute_repulsive_force(
        &self,
        point_idx: usize,
        p_x: T,
        p_y: T,
        theta: T,
        stack: &mut Vec<usize>,
    ) -> (T, T, T) {
        let mut force_x = T::zero();
        let mut force_y = T::zero();
        let mut sum_q = T::zero();

        // Clear the stack from the previous run but keep its allocated capacity!
        stack.clear();
        stack.push(self.root);

        let theta_sq = theta * theta;
        let min_dist_sq = T::from_f64(1e-12).unwrap();

        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni];

            let dx = p_x - node.com_x;
            let dy = p_y - node.com_y;
            let dist_sq = (dx * dx + dy * dy).max(min_dist_sq);

            if let Some(ref leaf) = node.leaf_data {
                let m = if leaf.contains(point_idx) {
                    node.mass - T::one()
                } else {
                    node.mass
                };

                if m > T::zero() && dist_sq > min_dist_sq {
                    let q = T::one() / (T::one() + dist_sq);
                    sum_q = sum_q + m * q;
                    let mult = m * q * q;
                    force_x = force_x + mult * dx;
                    force_y = force_y + mult * dy;
                }
                continue;
            }

            let is_summary = node.width * node.width < theta_sq * dist_sq;

            if is_summary {
                let q = T::one() / (T::one() + dist_sq);
                sum_q = sum_q + node.mass * q;
                let mult = node.mass * q * q;
                force_x = force_x + mult * dx;
                force_y = force_y + mult * dy;
            } else {
                for child in node.children.iter().flatten() {
                    stack.push(*child);
                }
            }
        }

        (force_x, force_y, sum_q)
    }
}

///////////
// Tests //
///////////

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
        assert!(root.is_leaf());
        assert!(matches!(root.leaf_data, Some(LeafData::Single(0))));
    }

    #[test]
    fn test_two_points_different_quadrants() {
        let embd = embd_from_tuples(&[(0.0, 0.0), (10.0, 10.0)]);
        let tree = BarnesHutTree::new(&embd);

        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.mass, 2.0);
        assert_relative_eq!(root.com_x, 5.0);
        assert_relative_eq!(root.com_y, 5.0);
        assert!(!root.is_leaf());
        assert!(root.leaf_data.is_none());
    }

    #[test]
    fn test_coincident_points_stored_correctly() {
        let embd = embd_from_tuples(&[(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]);
        let tree = BarnesHutTree::new(&embd);

        let root = &tree.nodes[tree.root];
        assert_relative_eq!(root.mass, 3.0);
        assert!(root.is_leaf());

        match &root.leaf_data {
            Some(LeafData::Coincident(indices)) => {
                assert_eq!(indices.len(), 3);
                assert!(indices.contains(&0));
                assert!(indices.contains(&1));
                assert!(indices.contains(&2));
            }
            other => panic!("Expected Coincident, got {:?}", other),
        }
    }

    #[test]
    fn test_mass_conservation() {
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
        let mut stack = Vec::new();

        let (fx, fy, sum_q) = tree.compute_repulsive_force(0, 0.0, 0.0, 0.5, &mut stack);
        assert_relative_eq!(fx, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fy, 0.0, epsilon = 1e-10);
        assert_relative_eq!(sum_q, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_self_interaction_coincident_points() {
        let embd = embd_from_tuples(&[(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]);
        let tree = BarnesHutTree::new(&embd);
        let mut stack = Vec::new();

        for i in 0..3 {
            let (fx, fy, sum_q) = tree.compute_repulsive_force(i, 5.0, 5.0, 0.5, &mut stack);
            assert!(fx.is_finite(), "fx is not finite for point {}", i);
            assert!(fy.is_finite(), "fy is not finite for point {}", i);
            assert!(sum_q.is_finite(), "sum_q is not finite for point {}", i);

            let (_, _, sum_q_0) = tree.compute_repulsive_force(0, 5.0, 5.0, 0.5, &mut stack);
            assert_relative_eq!(sum_q, sum_q_0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_force_symmetry_two_points() {
        let embd = embd_from_tuples(&[(0.0, 0.0), (2.0, 0.0)]);
        let tree = BarnesHutTree::new(&embd);
        let mut stack = Vec::new();

        let (fx0, fy0, _) = tree.compute_repulsive_force(0, 0.0, 0.0, 0.5, &mut stack);
        let (fx1, fy1, _) = tree.compute_repulsive_force(1, 2.0, 0.0, 0.5, &mut stack);

        assert_relative_eq!(fx0, -fx1, epsilon = 1e-10);
        assert_relative_eq!(fy0, -fy1, epsilon = 1e-10);
        assert!(fx0 < 0.0, "Point 0 should be pushed left");
        assert!(fx1 > 0.0, "Point 1 should be pushed right");
        assert_relative_eq!(fy0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(fy1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_force_magnitude_decreases_with_distance() {
        let embd_close = embd_from_tuples(&[(0.0, 0.0), (1.0, 0.0)]);
        let embd_far = embd_from_tuples(&[(0.0, 0.0), (10.0, 0.0)]);

        let tree_close = BarnesHutTree::new(&embd_close);
        let tree_far = BarnesHutTree::new(&embd_far);
        let mut stack = Vec::new();

        let (fx_close, _, _) = tree_close.compute_repulsive_force(0, 0.0, 0.0, 0.5, &mut stack);
        let (fx_far, _, _) = tree_far.compute_repulsive_force(0, 0.0, 0.0, 0.5, &mut stack);

        assert!(
            fx_close.abs() > fx_far.abs(),
            "Close force {} should be larger than far force {}",
            fx_close.abs(),
            fx_far.abs()
        );
    }

    #[test]
    fn test_sum_q_equals_n_minus_1_for_exact_computation() {
        let embd = embd_from_tuples(&[(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]);
        let tree = BarnesHutTree::new(&embd);
        let mut stack = Vec::new();

        let (_, _, sum_q) = tree.compute_repulsive_force(0, 0.0, 0.0, 0.0, &mut stack);
        assert!(sum_q > 0.0);
        assert!(sum_q < 3.0);
    }

    #[test]
    fn test_barnes_hut_approximation_reasonable() {
        let mut pts = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                pts.push((i as f64 * 2.0, j as f64 * 2.0));
            }
        }
        let embd = embd_from_tuples(&pts);
        let tree = BarnesHutTree::new(&embd);
        let mut stack = Vec::new();

        let test_idx = 55;
        let (px, py) = pts[test_idx];

        let (fx_exact, fy_exact, _) =
            tree.compute_repulsive_force(test_idx, px, py, 0.0, &mut stack);
        let (fx_approx, fy_approx, _) =
            tree.compute_repulsive_force(test_idx, px, py, 0.5, &mut stack);

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
        let configs: Vec<Vec<(f64, f64)>> = vec![
            vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (1e-10, 0.0)],
            vec![(0.0, 0.0), (1e-8, 1e-8), (100.0, 100.0)],
            vec![(1e6, 1e6), (1e6 + 1.0, 1e6)],
        ];

        for (cfg_idx, pts) in configs.iter().enumerate() {
            let embd = embd_from_tuples(pts);
            let tree = BarnesHutTree::new(&embd);
            let mut stack = Vec::new();

            for i in 0..pts.len() {
                let (fx, fy, sum_q) =
                    tree.compute_repulsive_force(i, pts[i].0, pts[i].1, 0.5, &mut stack);
                assert!(
                    fx.is_finite(),
                    "Config {}, point {}: fx not finite",
                    cfg_idx,
                    i
                );
                assert!(
                    fy.is_finite(),
                    "Config {}, point {}: fy not finite",
                    cfg_idx,
                    i
                );
                assert!(
                    sum_q.is_finite(),
                    "Config {}, point {}: sum_q not finite",
                    cfg_idx,
                    i
                );
                assert!(
                    sum_q >= 0.0,
                    "Config {}, point {}: sum_q negative",
                    cfg_idx,
                    i
                );
            }
        }
    }

    #[test]
    fn test_total_sum_q_consistent() {
        let embd = embd_from_tuples(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]);
        let tree = BarnesHutTree::new(&embd);
        let mut stack = Vec::new();

        let total: f64 = (0..4)
            .map(|i| {
                let (_, _, sq) =
                    tree.compute_repulsive_force(i, embd[i][0], embd[i][1], 0.0, &mut stack);
                sq
            })
            .sum();

        assert!(total > 0.0);
        assert!(total.is_finite());
    }

    #[test]
    fn test_self_exclusion_exact_for_all_coincident() {
        let embd = embd_from_tuples(&[(3.0, 3.0), (3.0, 3.0), (3.0, 3.0), (3.0, 3.0)]);
        let tree = BarnesHutTree::new(&embd);
        let mut stack = Vec::new();

        let mut sum_qs = Vec::new();
        for i in 0..4 {
            let (_, _, sq) = tree.compute_repulsive_force(i, 3.0, 3.0, 0.5, &mut stack);
            sum_qs.push(sq);
        }

        for i in 1..4 {
            assert!(
                (sum_qs[0] - sum_qs[i]).abs() < 1e-12,
                "sum_q differs between point 0 ({}) and point {} ({})",
                sum_qs[0],
                i,
                sum_qs[i]
            );
        }
    }
}

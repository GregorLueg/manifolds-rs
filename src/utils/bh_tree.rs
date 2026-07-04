//! Morton (Z-order) linear quad-tree Barnes-Hut implementation for tSNE
//! fitting.
//!
//! The build quantises the embedding into per-axis 32-bit integer buckets,
//! interleaves them into 64-bit Morton codes, sorts a `(code, index)`
//! permutation, and walks the sorted codes breadth-first to emit nodes whose
//! children occupy a contiguous arena range. Nodes store only what the force
//! traversal reads (centre of mass, count, level); raw coordinates are never
//! stored.

use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;

/////////////
// Helpers //
/////////////

/// `first_child` value marking a leaf.
const SENTINEL: u32 = u32::MAX;

/// Bits per axis in the Morton code. 32 bits is lossless for `f32` coords; for
/// `f64` quantisation only decides tree topology, never force arithmetic
/// (centres of mass are computed from actual coordinates).
const BITS: u32 = 32;

/// Minimum rayon chunk length, so small inputs stay on one thread.
const PAR_MIN_LEN: usize = 1024;

/// Spread the low 32 bits of `x` into the even bit positions (one zero gap
/// between each), the 2D Morton building block.
///
/// ### Params
///
/// * `x` - Value whose low 32 bits are spread; higher bits are ignored
///
/// ### Returns
///
/// `x` with its low 32 bits placed at even bit positions 0, 2, 4, ...
#[inline]
const fn part_1by1(mut x: u64) -> u64 {
    x &= 0x0000_0000_ffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;

    (x | (x << 1)) & 0x5555_5555_5555_5555
}

/// Interleave two 32-bit axis buckets into a 64-bit Morton code, x varying
/// fastest (canonical Z-order).
///
/// ### Params
///
/// * `x` - Quantised X-axis bucket
/// * `y` - Quantised Y-axis bucket
///
/// ### Returns
///
/// 64-bit Morton (Z-order) code interleaving `x` and `y`
#[inline]
const fn encode(x: u32, y: u32) -> u64 {
    part_1by1(x as u64) | (part_1by1(y as u64) << 1)
}

/// Quantise one coordinate into its axis bucket.
///
/// A degenerate zero-width axis has `inv_scale == 0` and collapses every
/// point to bucket zero.
///
/// ### Params
///
/// * `v` - Coordinate to quantise
/// * `min` - Minimum coordinate value along this axis
/// * `inv_scale` - Reciprocal of the per-bucket width along this axis
///
/// ### Returns
///
/// Axis bucket index in `[0, u32::MAX]`
#[inline]
fn quantise<T: Float>(v: T, min: T, inv_scale: T) -> u32 {
    let bucket = ((v - min) * inv_scale).to_u64().unwrap_or(0);
    bucket.min(u32::MAX as u64) as u32
}

//////////
// Node //
//////////

/// A node in the flat quad-tree arena.
///
/// The force traversal reads `com_x`, `com_y`, `count`, and `level` (which
/// sizes the cell through the per-tree width table), and follows
/// `first_child` for the `child_count` children stored contiguously. A leaf
/// is marked by `first_child == SENTINEL`.
///
/// Only non-empty quadrants are emitted as children, stored back to back,
/// rather than four fixed slots.
#[derive(Debug, Clone)]
pub struct Node<T> {
    /// Centre of mass (X)
    pub com_x: T,
    /// Centre of mass (Y)
    pub com_y: T,
    /// Number of points in this subtree
    pub count: u32,
    /// Arena index of the first child; `SENTINEL` for leaves
    pub first_child: u32,
    /// Number of contiguous children (at most 4)
    pub child_count: u8,
    /// Tree level, indexing the squared-width table for the theta test
    pub level: u8,
}

///////////////////
// BarnesHutTree //
///////////////////

/// Morton linear Barnes-Hut quad-tree for 2D repulsive force approximation.
///
/// Holds its build buffers across epochs: the epoch loop keeps one tree and
/// calls [`BarnesHutTree::rebuild`] each epoch, reusing allocations instead
/// of reallocating.
pub struct BarnesHutTree<T> {
    /// Flat arena of nodes; children always follow their parent
    pub nodes: Vec<Node<T>>,
    /// Build scratch: each node's half-open window in `sorted`
    ranges: Vec<(u32, u32)>,
    /// The `(Morton code, point index)` permutation
    sorted: Vec<(u64, u32)>,
    /// Squared full cell width per level for the theta test.
    level_width_sq: Vec<T>,
}

impl<T> BarnesHutTree<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    /// Create an empty tree. The epoch loop holds one of these and rebuilds
    /// it each epoch so the buffers persist.
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            ranges: Vec::new(),
            sorted: Vec::new(),
            level_width_sq: Vec::new(),
        }
    }

    /// Build a fresh tree over `pos`. Convenience for one-shot callers; the
    /// epoch loop should hold one tree and call [`BarnesHutTree::rebuild`].
    ///
    /// ### Params
    ///
    /// * `pos` - Interleaved point coordinates `[x0, y0, x1, y1, ...]`
    pub fn new(pos: &[T]) -> Self {
        let mut tree = Self::empty();
        tree.rebuild(pos);
        tree
    }

    /// Rebuild the tree over `pos` in place, reusing the retained buffers.
    ///
    /// ### Params
    ///
    /// * `pos` - Interleaved point coordinates `[x0, y0, x1, y1, ...]`
    pub fn rebuild(&mut self, pos: &[T]) {
        let n = pos.len() / 2;

        self.nodes.clear();
        self.ranges.clear();
        self.sorted.clear();

        if n == 0 {
            self.level_width_sq.clear();
            return;
        }

        let (min_x, max_x, min_y, max_y) = pos
            .par_chunks_exact(2)
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || {
                    (
                        T::infinity(),
                        T::neg_infinity(),
                        T::infinity(),
                        T::neg_infinity(),
                    )
                },
                |(min_x, max_x, min_y, max_y), p| {
                    (
                        min_x.min(p[0]),
                        max_x.max(p[0]),
                        min_y.min(p[1]),
                        max_y.max(p[1]),
                    )
                },
            )
            .reduce(
                || {
                    (
                        T::infinity(),
                        T::neg_infinity(),
                        T::infinity(),
                        T::neg_infinity(),
                    )
                },
                |a, b| (a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3)),
            );

        let extent_x = max_x - min_x;
        let extent_y = max_y - min_y;
        let scale = T::from_u64(1u64 << BITS).unwrap();
        let inv_scale_x = if extent_x > T::zero() {
            scale / extent_x
        } else {
            T::zero()
        };
        let inv_scale_y = if extent_y > T::zero() {
            scale / extent_y
        } else {
            T::zero()
        };

        let max_extent = extent_x.max(extent_y);
        self.level_width_sq.clear();
        for level in 0..=BITS {
            let width = max_extent / T::from_u64(1u64 << level).unwrap();
            self.level_width_sq.push(width * width);
        }

        self.sorted.par_extend(
            pos.par_chunks_exact(2)
                .with_min_len(PAR_MIN_LEN)
                .enumerate()
                .map(|(i, p)| {
                    let code = encode(
                        quantise(p[0], min_x, inv_scale_x),
                        quantise(p[1], min_y, inv_scale_y),
                    );
                    (code, i as u32)
                }),
        );
        self.sorted.par_sort_unstable_by_key(|&(code, _)| code);

        let sorted = &self.sorted;
        let nodes = &mut self.nodes;
        let ranges = &mut self.ranges;

        nodes.reserve(2 * n - 1);
        ranges.reserve(2 * n - 1);
        nodes.push(Node {
            com_x: T::zero(),
            com_y: T::zero(),
            count: n as u32,
            first_child: SENTINEL,
            child_count: 0,
            level: BITS as u8,
        });
        ranges.push((0, n as u32));

        let mut node = 0usize;
        while node < nodes.len() {
            let (start, end) = ranges[node];
            if end - start <= 1 || sorted[start as usize].0 == sorted[(end - 1) as usize].0 {
                node += 1;
                continue;
            }

            let xor = sorted[start as usize].0 ^ sorted[(end - 1) as usize].0;
            let highest_diff = 63 - xor.leading_zeros();
            let level = (BITS - 1) - highest_diff / 2;
            let shift = 2 * (BITS - 1 - level);
            nodes[node].level = level as u8;

            let first_child = nodes.len() as u32;
            let mut child_count: u8 = 0;
            let mut child_start = start;
            while child_start < end {
                let group = (sorted[child_start as usize].0 >> shift) & 0b11;
                let mut child_end = child_start + 1;
                while child_end < end && (sorted[child_end as usize].0 >> shift) & 0b11 == group {
                    child_end += 1;
                }
                nodes.push(Node {
                    com_x: T::zero(),
                    com_y: T::zero(),
                    count: child_end - child_start,
                    first_child: SENTINEL,
                    child_count: 0,
                    level: BITS as u8,
                });
                ranges.push((child_start, child_end));
                child_count += 1;
                child_start = child_end;
            }
            nodes[node].first_child = first_child;
            nodes[node].child_count = child_count;
            node += 1;
        }

        nodes
            .par_iter_mut()
            .zip(ranges.par_iter())
            .with_min_len(PAR_MIN_LEN)
            .filter(|(node, _)| node.first_child == SENTINEL)
            .for_each(|(node, &(start, end))| {
                let mut sum_x = T::zero();
                let mut sum_y = T::zero();
                for slot in start..end {
                    let idx = sorted[slot as usize].1 as usize;
                    sum_x = sum_x + pos[2 * idx];
                    sum_y = sum_y + pos[2 * idx + 1];
                }
                let inv = T::from_u32(node.count).unwrap().recip();
                node.com_x = sum_x * inv;
                node.com_y = sum_y * inv;
            });

        for i in (0..nodes.len()).rev() {
            if nodes[i].first_child == SENTINEL {
                continue;
            }
            let first = nodes[i].first_child as usize;
            let last = first + nodes[i].child_count as usize;
            let mut sum_x = T::zero();
            let mut sum_y = T::zero();
            for child in &nodes[first..last] {
                let w = T::from_u32(child.count).unwrap();
                sum_x = sum_x + child.com_x * w;
                sum_y = sum_y + child.com_y * w;
            }
            let inv = T::from_u32(nodes[i].count).unwrap().recip();
            nodes[i].com_x = sum_x * inv;
            nodes[i].com_y = sum_y * inv;
        }

        debug_assert_eq!(
            nodes
                .iter()
                .filter(|node| node.first_child == SENTINEL)
                .map(|node| node.count as u64)
                .sum::<u64>(),
            n as u64,
            "tree lost or invented points"
        );
    }

    /// Compute repulsive forces on a point using Barnes-Hut approximation.
    ///
    /// Traverses the arena with an explicit reusable stack. A node is
    /// summarised by its centre of mass when it is a leaf or passes the theta
    /// opening criterion; otherwise its children are pushed.
    ///
    /// ### Params
    ///
    /// * `p_x` - X coordinate of the query point
    /// * `p_y` - Y coordinate of the query point
    /// * `theta` - Barnes-Hut opening parameter (typically 0.5)
    /// * `stack` - Reusable traversal scratch (cleared here, capacity kept)
    ///
    /// ### Returns
    ///
    /// Tuple `(force_x, force_y, sum_q)` where `sum_q` is the sum of
    /// unnormalised Student-t affinities (for computing Z)
    pub fn compute_repulsive_force(
        &self,
        p_x: T,
        p_y: T,
        theta: T,
        stack: &mut Vec<u32>,
    ) -> (T, T, T) {
        let mut force_x = T::zero();
        let mut force_y = T::zero();
        let mut sum_q = T::zero();

        if self.nodes.is_empty() {
            return (force_x, force_y, sum_q);
        }

        let theta_sq = theta * theta;
        let min_dist_sq = T::from_f64(1e-12).unwrap();

        stack.clear();
        stack.push(0);

        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];

            let dx = p_x - node.com_x;
            let dy = p_y - node.com_y;
            let dist_sq = dx * dx + dy * dy;

            if node.first_child == SENTINEL {
                // Self / coincident exclusion.
                if dist_sq <= min_dist_sq {
                    continue;
                }
            } else if self.level_width_sq[node.level as usize] >= theta_sq * dist_sq {
                // Cell subtends too large an angle: descend.
                for child in 0..node.child_count as u32 {
                    stack.push(node.first_child + child);
                }
                continue;
            }

            // Summarise the cell by its centre of mass.
            let q = (T::one() + dist_sq).recip();
            let mass_q = T::from_u32(node.count).unwrap() * q;
            sum_q = sum_q + mass_q;
            let mult = mass_q * q;
            force_x = force_x + mult * dx;
            force_y = force_y + mult * dy;
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

    /// Helper: flat interleaved embedding from a slice of tuples.
    fn pos_from_tuples(pts: &[(f64, f64)]) -> Vec<f64> {
        pts.iter().flat_map(|&(x, y)| [x, y]).collect()
    }

    /// Brute-force reference with the same coincident-point exclusion
    /// (`dist_sq <= 1e-12` contributes nothing) as the tree.
    fn brute_force(pos: &[f64], i: usize) -> (f64, f64, f64) {
        let n = pos.len() / 2;
        let (mut fx, mut fy, mut sq) = (0.0, 0.0, 0.0);
        for j in 0..n {
            let dx = pos[2 * i] - pos[2 * j];
            let dy = pos[2 * i + 1] - pos[2 * j + 1];
            let d = dx * dx + dy * dy;
            if d <= 1e-12 {
                continue;
            }
            let q = 1.0 / (1.0 + d);
            sq += q;
            fx += q * q * dx;
            fy += q * q * dy;
        }
        (fx, fy, sq)
    }

    /// Deterministic LCG cloud, no RNG dependency.
    fn lcg_cloud(n: usize, mut state: u64) -> Vec<f64> {
        (0..2 * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 40) as f64 / (1u64 << 24) as f64
            })
            .collect()
    }

    #[test]
    fn test_single_point_tree() {
        let tree = BarnesHutTree::new(&[1.0f64, 2.0]);
        assert_eq!(tree.nodes.len(), 1);
        let root = &tree.nodes[0];
        assert_relative_eq!(root.com_x, 1.0);
        assert_relative_eq!(root.com_y, 2.0);
        assert_eq!(root.count, 1);
        assert_eq!(root.first_child, SENTINEL);
    }

    #[test]
    fn test_two_points_different_quadrants() {
        let tree = BarnesHutTree::new(&pos_from_tuples(&[(0.0, 0.0), (10.0, 10.0)]));
        let root = &tree.nodes[0];
        assert_eq!(root.count, 2);
        assert_relative_eq!(root.com_x, 5.0);
        assert_relative_eq!(root.com_y, 5.0);
        assert_ne!(root.first_child, SENTINEL);
        assert_eq!(root.child_count, 2);
    }

    #[test]
    fn test_coincident_points_collapse_to_one_leaf() {
        let pos = vec![5.0f64; 1000];
        let tree = BarnesHutTree::new(&pos);
        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.nodes[0].count, 500);
        assert_eq!(tree.nodes[0].first_child, SENTINEL);
    }

    #[test]
    fn test_mass_conservation() {
        let pos = lcg_cloud(2_000, 17);
        let tree = BarnesHutTree::new(&pos);
        assert_eq!(tree.nodes[0].count, 2_000);
        let leaf_sum: u64 = tree
            .nodes
            .iter()
            .filter(|n| n.first_child == SENTINEL)
            .map(|n| n.count as u64)
            .sum();
        assert_eq!(leaf_sum, 2_000);
    }

    #[test]
    fn test_root_centre_of_mass_is_the_mean() {
        let pos = lcg_cloud(1_000, 5);
        let tree = BarnesHutTree::new(&pos);
        let (mut mx, mut my) = (0.0, 0.0);
        for p in pos.chunks_exact(2) {
            mx += p[0];
            my += p[1];
        }
        assert_relative_eq!(tree.nodes[0].com_x, mx / 1_000.0, epsilon = 1e-9);
        assert_relative_eq!(tree.nodes[0].com_y, my / 1_000.0, epsilon = 1e-9);
    }

    #[test]
    fn test_no_self_interaction() {
        // A single point and a coincident cluster both yield zero force and
        // zero sum_q for a query at their position.
        let mut stack = Vec::new();
        for pos in [vec![0.0f64, 0.0], vec![5.0f64; 8]] {
            let tree = BarnesHutTree::new(&pos);
            let (fx, fy, sq) = tree.compute_repulsive_force(pos[0], pos[1], 0.5, &mut stack);
            assert_relative_eq!(fx, 0.0);
            assert_relative_eq!(fy, 0.0);
            assert_relative_eq!(sq, 0.0);
        }
    }

    #[test]
    fn test_force_symmetry_two_points() {
        let tree = BarnesHutTree::new(&pos_from_tuples(&[(0.0, 0.0), (2.0, 0.0)]));
        let mut stack = Vec::new();
        let (fx0, fy0, _) = tree.compute_repulsive_force(0.0, 0.0, 0.5, &mut stack);
        let (fx1, fy1, _) = tree.compute_repulsive_force(2.0, 0.0, 0.5, &mut stack);
        assert_relative_eq!(fx0, -fx1, epsilon = 1e-10);
        assert_relative_eq!(fy0, -fy1, epsilon = 1e-10);
        assert!(fx0 < 0.0, "Point 0 should be pushed left");
        assert!(fx1 > 0.0, "Point 1 should be pushed right");
    }

    #[test]
    fn test_force_magnitude_decreases_with_distance() {
        let close = BarnesHutTree::new(&pos_from_tuples(&[(0.0, 0.0), (1.0, 0.0)]));
        let far = BarnesHutTree::new(&pos_from_tuples(&[(0.0, 0.0), (10.0, 0.0)]));
        let mut stack = Vec::new();
        let (fx_close, _, _) = close.compute_repulsive_force(0.0, 0.0, 0.5, &mut stack);
        let (fx_far, _, _) = far.compute_repulsive_force(0.0, 0.0, 0.5, &mut stack);
        assert!(fx_close.abs() > fx_far.abs());
    }

    #[test]
    fn test_theta_zero_matches_brute_force() {
        let pos = lcg_cloud(400, 23);
        let tree = BarnesHutTree::new(&pos);
        let mut stack = Vec::new();
        for i in 0..400 {
            let (fx, fy, sq) =
                tree.compute_repulsive_force(pos[2 * i], pos[2 * i + 1], 0.0, &mut stack);
            let (bx, by, bq) = brute_force(&pos, i);
            assert_relative_eq!(fx, bx, epsilon = 1e-9);
            assert_relative_eq!(fy, by, epsilon = 1e-9);
            assert_relative_eq!(sq, bq, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_barnes_hut_approximation_reasonable() {
        let pos = lcg_cloud(1_000, 42);
        let tree = BarnesHutTree::new(&pos);
        let mut stack = Vec::new();
        for i in (0..1_000).step_by(97) {
            let (_, _, sq_exact) =
                tree.compute_repulsive_force(pos[2 * i], pos[2 * i + 1], 0.0, &mut stack);
            let (_, _, sq_approx) =
                tree.compute_repulsive_force(pos[2 * i], pos[2 * i + 1], 0.5, &mut stack);
            assert!((sq_approx / sq_exact - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn test_no_nan_or_inf_forces() {
        let configs: Vec<Vec<(f64, f64)>> = vec![
            vec![(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (1e-10, 0.0)],
            vec![(0.0, 0.0), (1e-8, 1e-8), (100.0, 100.0)],
            vec![(1e6, 1e6), (1e6 + 1.0, 1e6)],
        ];
        let mut stack = Vec::new();
        for pts in &configs {
            let pos = pos_from_tuples(pts);
            let tree = BarnesHutTree::new(&pos);
            for i in 0..pts.len() {
                let (fx, fy, sq) =
                    tree.compute_repulsive_force(pos[2 * i], pos[2 * i + 1], 0.5, &mut stack);
                assert!(fx.is_finite() && fy.is_finite() && sq.is_finite());
                assert!(sq >= 0.0);
            }
        }
    }

    #[test]
    fn test_empty_input() {
        let tree = BarnesHutTree::<f64>::new(&[]);
        let mut stack = Vec::new();
        let (fx, fy, sq) = tree.compute_repulsive_force(0.0, 0.0, 0.5, &mut stack);
        assert_eq!((fx, fy, sq), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_rebuild_matches_fresh_tree() {
        let a = lcg_cloud(500, 1);
        let b = lcg_cloud(800, 2);
        let mut reused = BarnesHutTree::new(&a);
        reused.rebuild(&b);
        let fresh = BarnesHutTree::new(&b);
        let mut stack = Vec::new();
        for i in (0..800).step_by(53) {
            let r = reused.compute_repulsive_force(b[2 * i], b[2 * i + 1], 0.5, &mut stack);
            let f = fresh.compute_repulsive_force(b[2 * i], b[2 * i + 1], 0.5, &mut stack);
            assert_eq!(r, f);
        }
    }
}

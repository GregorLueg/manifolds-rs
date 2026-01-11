use num_traits::{Float, FromPrimitive};

/////////////
// Helpers //
/////////////

/// A simple 2D Bounding Box
///
/// ### Fields
///
/// * `x_min` - Min value in x direction
/// * `x_max` - Max value in x direction
/// * `y_min` - Min value in y direction
/// * `y_max` - Max value in y direction
#[derive(Clone, Copy, Debug)]
struct BBox<T> {
    x_min: T,
    x_max: T,
    y_min: T,
    y_max: T,
}

impl<T: Float> BBox<T> {
    /// Check which quadrant a point belongs to relative to the center of this
    /// box
    ///
    /// ### Params
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    ///
    /// ### Returns
    ///
    /// `usize` -> 0 (NW), 1 (NE), 2 (SW), 3 (SE)
    fn get_quadrant(&self, x: T, y: T) -> usize {
        let center_x = (self.x_min + self.x_max) / (T::one() + T::one());
        let center_y = (self.y_min + self.y_max) / (T::one() + T::one());

        let right = if x > center_x { 1 } else { 0 };
        let bottom = if y > center_y { 2 } else { 0 };
        right + bottom
    }

    /// Return the boundaries for a specific quadrant
    ///
    /// ### Params
    ///
    /// * `quadrant` - The idx of the quadrant. Must be one of `0`, `1`, `2` or
    ///   `3`; otherwise, it will panic.
    ///
    /// ### Returns
    ///
    /// The `BBox`.
    fn sub_quadrant(&self, quadrant: usize) -> Self {
        let center_x = (self.x_min + self.x_max) / (T::one() + T::one());
        let center_y = (self.y_min + self.y_max) / (T::one() + T::one());

        match quadrant {
            0 => BBox {
                x_min: self.x_min,
                x_max: center_x,
                y_min: self.y_min,
                y_max: center_y,
            }, // NW
            1 => BBox {
                x_min: center_x,
                x_max: self.x_max,
                y_min: self.y_min,
                y_max: center_y,
            }, // NE
            2 => BBox {
                x_min: self.x_min,
                x_max: center_x,
                y_min: center_y,
                y_max: self.y_max,
            }, // SW
            3 => BBox {
                x_min: center_x,
                x_max: self.x_max,
                y_min: center_y,
                y_max: self.y_max,
            }, // SE
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
/// * `com_x` -  Center of mass (X)
/// * `com_y` - Center of mass (Y)
/// * `mass` - Total mass (number of points in this node/subtree)
/// * `width` - Width of the cell covered by this node
/// * `children` - Indices of children in the `nodes` vector; `None` if child
///   does not exist.
/// * `point_idx` - If this is a leaf, it may store the index of the original
///   data point
#[derive(Debug, Clone)]
pub struct QuadNode<T> {
    pub com_x: T,
    pub com_y: T,
    pub mass: T,
    pub width: T,
    pub children: [Option<usize>; 4],
    pub point_idx: Option<usize>,
}

/// Barnes-Hut Tree
///
/// Continuous flat arena to store the nodes for better cache locality
///
/// ### Fields
///
/// * `nodes` - The flat structure storing the `QuadNode`s
/// * `root` - The index of the root (usually 0).
pub struct BarnesHutTree<T> {
    pub nodes: Vec<QuadNode<T>>,
    root: usize,
}

impl<T> BarnesHutTree<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    /// Build the QuadTree from the current embedding
    ///
    /// ### Params
    ///
    /// * `embd` - The current 2D embeddings.
    ///
    /// ### Returns self
    pub fn new(embd: &[Vec<T>]) -> Self {
        // global bounds
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

        // add a small epsilon to bounds to ensure points don't fall exactly on
        // edge
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

    /// Build the BarnesHutTree recursively
    ///
    /// ### Params
    ///
    /// * `nodes` - Mutable reference to the internal vector storage
    /// * `embd` - The current 2D embeddings.
    /// * `point_indices` - Slice of indices of the samples
    /// * `bbox` - The BBox (in the recursion the subquadrants will be used)
    ///
    /// ### Returns
    ///
    /// The index of the created node in the `nodes` vector
    fn build_recursive(
        nodes: &mut Vec<QuadNode<T>>,
        embd: &[Vec<T>],
        point_indices: &[usize],
        bbox: BBox<T>,
    ) -> usize {
        // assuming roughly square for simplicity -> will go back to this
        let width = bbox.x_max - bbox.x_min;

        // add the first node
        if point_indices.len() == 1 {
            let idx = point_indices[0];
            let p = &embd[idx];

            let node = QuadNode {
                com_x: p[0],
                com_y: p[1],
                mass: T::one(),
                width,
                children: [None; 4],
                point_idx: Some(idx),
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

        let node_idx = nodes.len();

        nodes.push(QuadNode {
            com_x: sum_x / mass,
            com_y: sum_y / mass,
            mass,
            width,
            children: [None; 4],
            point_idx: None,
        });

        let mut children_indices = [None; 4];

        for i in 0..4 {
            if !buckets[i].is_empty() {
                let child_idx =
                    Self::build_recursive(nodes, embd, &buckets[i], bbox.sub_quadrant(i));
                children_indices[i] = Some(child_idx);
            }
        }

        // add the children information
        nodes[node_idx].children = children_indices;

        node_idx
    }

    /// Compute the repulsive forces on a point using the Barnes-Hut
    /// approximation
    ///
    /// ### Params
    ///
    /// * `point_idx` - Index of the point in the original embedding
    /// * `p_x` - X coordinate of the point
    /// * `p_y` - Y coordinate of the point
    /// * `theta` - Barnes-Hut approximation parameter (typically 0.5). Lower
    ///   values are more accurate but slower
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

        // we use an iterative stack to traverse the tree (avoids recursion
        // overhead)
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            let dx = p_x - node.com_x;
            let dy = p_y - node.com_y;
            let dist_sq = dx * dx + dy * dy;

            // 1. check if node is the point itself (ignore)
            if node.point_idx == Some(point_idx) && dist_sq < T::from_f64(1e-8).unwrap() {
                continue;
            }

            // 2. Barnes-Hut condition: is the node far enough to be treated as
            // a summary? width^2 / dist^2 < theta^2
            let is_summary = node.width * node.width < theta * theta * dist_sq;
            let is_leaf = node.point_idx.is_some() || node.children.iter().all(|c| c.is_none());

            if is_summary || is_leaf {
                let q = T::one() / (T::one() + dist_sq);
                let mult = node.mass * q * q;

                sum_q = sum_q + node.mass * q;
                force_x = force_x + mult * dx;
                force_y = force_y + mult * dy;
            } else {
                // node is too close and not a leaf: Open it up (push children to stack)
                for child in node.children.iter().flatten() {
                    stack.push(*child);
                }
            }
        }

        (force_x, force_y, sum_q)
    }
}

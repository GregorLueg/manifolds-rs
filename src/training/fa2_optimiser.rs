//! ForceAtlas2 optimiser (Jacomy et al. 2014). Repulsion runs through the
//! mass-weighted Barnes-Hut tree; attraction, gravity and Gephi's adaptive
//! speed control are exact. Node masses are `1 + degree`, as in Gephi.

use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;

use crate::data::graph::coo_to_adjacency_list;
use crate::data::structures::*;
use crate::prelude::*;
use crate::utils::bh_tree::*;

/////////////
// Globals //
/////////////

/// Floor on the speed efficiency; Gephi stops halving or damping it below
/// this value.
const FA2_MIN_SPEED_EFFICIENCY: f64 = 0.05;

/// Maximum relative rise of the global speed per epoch (Gephi).
const FA2_MAX_RISE: f64 = 0.5;

/// Upper bound on the adaptive jitter tolerance (Gephi).
const FA2_MAX_JITTER: f64 = 10.0;

/// Scale of Gephi's `sqrt(n)` estimate of the optimal jitter tolerance.
const FA2_JITTER_SCALE: f64 = 0.05;

/// Swing-to-traction ratio above which the speed efficiency is halved.
const FA2_SWING_RATIO_LIMIT: f64 = 2.0;

/// Global speed above which the efficiency is no longer raised (Gephi).
const FA2_SPEED_CEILING: f64 = 1000.0;

/// Cut of the speed efficiency when swing dwarfs traction.
const FA2_EFFICIENCY_HALVE: f64 = 0.5;

/// Damping of the speed efficiency when swing exceeds the jitter budget.
const FA2_EFFICIENCY_DAMP: f64 = 0.7;

/// Growth of the speed efficiency when swing stays within budget.
const FA2_EFFICIENCY_GROWTH: f64 = 1.3;

////////////////
// Structures //
////////////////

/// ForceAtlas2 optimisation parameters. Defaults follow Gephi.
#[derive(Clone, Debug)]
pub struct Fa2OptimParams<T> {
    /// Number of epochs
    pub n_epochs: usize,
    /// Repulsion strength (`k_r`); larger values spread the layout
    pub scaling_ratio: T,
    /// Pull towards the origin, keeps disconnected components from drifting
    pub gravity: T,
    /// Distance-independent gravity scaled by `scaling_ratio`
    pub strong_gravity: bool,
    /// Logarithmic attraction (LinLog), tighter communities
    pub lin_log: bool,
    /// Divide attraction by node mass ("dissuade hubs"), pushing hubs to the
    /// periphery
    pub dissuade_hubs: bool,
    /// Exponent applied to edge weights; `0` ignores the weights
    pub edge_weight_influence: T,
    /// Tolerated swinging; larger is faster but less precise
    pub jitter_tolerance: T,
    /// Barnes-Hut opening parameter. The tree compares cell width rather than
    /// Gephi's region size, so equal values do not give equal accuracy.
    pub theta: T,
}

impl<T> Fa2OptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Generate a new instance.
    ///
    /// ### Params
    ///
    /// * `n_epochs` - Number of epochs
    /// * `scaling_ratio` - Repulsion strength
    /// * `gravity` - Pull towards the origin
    /// * `strong_gravity` - Distance-independent gravity
    /// * `lin_log` - Logarithmic attraction
    /// * `dissuade_hubs` - Divide attraction by node mass
    /// * `edge_weight_influence` - Exponent applied to edge weights
    /// * `jitter_tolerance` - Tolerated swinging
    /// * `theta` - Barnes-Hut opening parameter
    ///
    /// ### Returns
    ///
    /// Initialised parameters
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_epochs: usize,
        scaling_ratio: T,
        gravity: T,
        strong_gravity: bool,
        lin_log: bool,
        dissuade_hubs: bool,
        edge_weight_influence: T,
        jitter_tolerance: T,
        theta: T,
    ) -> Self {
        Self {
            n_epochs,
            scaling_ratio,
            gravity,
            strong_gravity,
            lin_log,
            dissuade_hubs,
            edge_weight_influence,
            jitter_tolerance,
            theta,
        }
    }
}

impl<T> Default for Fa2OptimParams<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            n_epochs: 500,
            scaling_ratio: T::from_f64(2.0).unwrap(),
            gravity: T::one(),
            strong_gravity: false,
            lin_log: false,
            dissuade_hubs: false,
            edge_weight_influence: T::one(),
            jitter_tolerance: T::one(),
            theta: T::from_f64(1.2).unwrap(),
        }
    }
}

///////////////
// Optimiser //
///////////////

/// Type of repulsion approximation to use for ForceAtlas2.
#[derive(Default)]
pub enum Fa2Opt {
    /// Barnes-Hut tree over mass-weighted cells
    #[default]
    BarnesHut,
}

/// Parse the ForceAtlas2 repulsion approximation.
///
/// ### Params
///
/// * `s` - Accepts `"barnes hut"`, `"barnes_hut"`, `"barnes-hut"` or `"bh"`
///
/// ### Returns
///
/// `Some(Fa2Opt)` if the string matches, `None` otherwise
pub fn parse_fa2_optimiser(s: &str) -> Option<Fa2Opt> {
    match s.to_lowercase().as_str() {
        "barnes hut" | "barnes_hut" | "barnes-hut" | "bh" => Some(Fa2Opt::BarnesHut),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Build the FA2 adjacency: self-loops dropped, weights raised to
/// `edge_weight_influence`, and the graph checked for symmetry (the force
/// gather visits each edge from both endpoints).
///
/// ### Params
///
/// * `graph` - Sparse graph in COO format
/// * `edge_weight_influence` - Exponent applied to edge weights
///
/// ### Returns
///
/// Adjacency list sorted by neighbour index, or
/// `ManifoldsError::AsymmetricGraph` naming the first unmatched edge
fn fa2_adjacency<T>(
    graph: &CoordinateList<T>,
    edge_weight_influence: T,
) -> Result<Vec<Vec<(usize, T)>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    if let Some((&row, &col)) = graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .find(|(&i, &j)| i >= n || j >= n)
    {
        return Err(ManifoldsError::AsymmetricGraph { row, col });
    }

    let mut adj = coo_to_adjacency_list(graph);
    adj.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.retain(|&(j, _)| j != i);
        row.sort_unstable_by_key(|&(j, _)| j);
    });

    let unmatched = adj.par_iter().enumerate().find_map_any(|(i, row)| {
        row.iter().find_map(
            |&(j, w)| match adj[j].binary_search_by_key(&i, |&(k, _)| k) {
                Ok(pos) if adj[j][pos].1 == w => None,
                _ => Some((i, j)),
            },
        )
    });
    if let Some((row, col)) = unmatched {
        return Err(ManifoldsError::AsymmetricGraph { row, col });
    }

    if edge_weight_influence != T::one() {
        adj.par_iter_mut().flatten().for_each(|(_, w)| {
            *w = if edge_weight_influence == T::zero() {
                T::one()
            } else {
                w.powf(edge_weight_influence)
            };
        });
    }

    Ok(adj)
}

/// Gephi's adaptive speed update from the global swing and traction.
///
/// ### Params
///
/// * `speed` - Current global speed
/// * `efficiency` - Current speed efficiency
/// * `swing` - Total mass-weighted swing, `sum m_i |F_i - F_i_old|`
/// * `traction` - Total effective traction, `sum m_i |F_i + F_i_old| / 2`
/// * `jitter_tolerance` - User jitter tolerance
/// * `n` - Number of nodes
///
/// ### Returns
///
/// Updated `(speed, efficiency)`
fn update_speed(
    speed: f64,
    mut efficiency: f64,
    swing: f64,
    traction: f64,
    jitter_tolerance: f64,
    n: usize,
) -> (f64, f64) {
    let n = n as f64;
    let estimated_jt = FA2_JITTER_SCALE * n.sqrt();
    let min_jt = estimated_jt.sqrt();
    let mut jt = if traction > 0.0 {
        jitter_tolerance * min_jt.max(FA2_MAX_JITTER.min(estimated_jt * traction / (n * n)))
    } else {
        jitter_tolerance * min_jt
    };

    if traction > 0.0 && swing / traction > FA2_SWING_RATIO_LIMIT {
        if efficiency > FA2_MIN_SPEED_EFFICIENCY {
            efficiency *= FA2_EFFICIENCY_HALVE;
        }
        jt = jt.max(jitter_tolerance);
    }

    let target = if swing == 0.0 {
        f64::INFINITY
    } else {
        jt * efficiency * traction / swing
    };

    if swing > jt * traction {
        if efficiency > FA2_MIN_SPEED_EFFICIENCY {
            efficiency *= FA2_EFFICIENCY_DAMP;
        }
    } else if speed < FA2_SPEED_CEILING {
        efficiency *= FA2_EFFICIENCY_GROWTH;
    }

    (
        speed + (target - speed).min(FA2_MAX_RISE * speed),
        efficiency,
    )
}

//////////
// Main //
//////////

/// Optimise a 2D embedding with ForceAtlas2.
///
/// Each epoch runs one parallel pass that fuses Barnes-Hut repulsion,
/// gravity and attraction per node, reduces global swing and traction in
/// `f64`, updates Gephi's adaptive speed, then moves every node by
/// `speed / (1 + sqrt(speed * swing_i))` times its force.
///
/// Attraction under `dissuade_hubs` is divided by each node's own mass and
/// scaled by the mean mass. Gephi divides both endpoints by the mass of the
/// edge's source node, which depends on edge orientation.
///
/// ### Params
///
/// * `embd` - Initial embedding, shape `[n_samples][2]` (modified in place)
/// * `params` - FA2 parameters
/// * `graph` - Symmetric weighted graph in COO format
/// * `verbose` - Verbosity level: `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `Ok(())`, or an error if the embedding is not 2D, is empty, does not
/// match the graph size, or the graph is not symmetric
///
/// ### References
///
/// Jacomy et al., PLoS ONE, 2014 (ForceAtlas2)
pub fn optimise_fa2<T>(
    embd: &mut [Vec<T>],
    params: &Fa2OptimParams<T>,
    graph: &CoordinateList<T>,
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

    let adj = fa2_adjacency(graph, params.edge_weight_influence)?;
    let masses: Vec<T> = adj
        .iter()
        .map(|row| T::from_usize(row.len() + 1).unwrap())
        .collect();

    let att_coef = if params.dissuade_hubs {
        let mean = masses.iter().map(|m| m.to_f64().unwrap()).sum::<f64>() / n as f64;
        T::from_f64(mean).unwrap()
    } else {
        T::one()
    };

    let kr = params.scaling_ratio;
    let gravity = params.gravity;
    let theta = params.theta;
    let jitter_tolerance = params.jitter_tolerance.to_f64().unwrap();

    let mut pos = vec![T::zero(); n * 2];
    let mut force = vec![T::zero(); n * 2];
    let mut old_force = vec![T::zero(); n * 2];
    let mut swing_traction = vec![(0.0_f64, 0.0_f64); n];
    let mut tree = BarnesHutTree::empty();

    let mut speed = 1.0_f64;
    let mut efficiency = 1.0_f64;

    for epoch in 0..params.n_epochs {
        embd.par_iter()
            .zip(pos.par_chunks_mut(2))
            .for_each(|(p, dst)| {
                dst[0] = p[0];
                dst[1] = p[1];
            });

        tree.rebuild(&pos, Some(&masses));

        force.par_chunks_mut(2).enumerate().for_each_init(
            || Vec::with_capacity(128),
            |stack, (i, f)| {
                let px = pos[2 * i];
                let py = pos[2 * i + 1];
                let m = masses[i];

                let (rx, ry) = tree.compute_fa2_repulsion(px, py, theta, stack);
                let mut fx = kr * m * rx;
                let mut fy = kr * m * ry;

                if params.strong_gravity {
                    let g = kr * m * gravity;
                    fx -= px * g;
                    fy -= py * g;
                } else {
                    let d = (px * px + py * py).sqrt();
                    if d > T::zero() {
                        let g = m * gravity / d;
                        fx -= px * g;
                        fy -= py * g;
                    }
                }

                let att_scale = if params.dissuade_hubs {
                    -att_coef / m
                } else {
                    -att_coef
                };
                let mut ax = T::zero();
                let mut ay = T::zero();
                for &(j, e) in &adj[i] {
                    let dx = px - pos[2 * j];
                    let dy = py - pos[2 * j + 1];
                    let w = if params.lin_log {
                        let d = (dx * dx + dy * dy).sqrt();
                        if d > T::zero() {
                            e * d.ln_1p() / d
                        } else {
                            T::zero()
                        }
                    } else {
                        e
                    };
                    ax += dx * w;
                    ay += dy * w;
                }

                f[0] = fx + ax * att_scale;
                f[1] = fy + ay * att_scale;
            },
        );

        // parallel fill, sequential sum: a rayon float reduce is not
        // reproducible across runs.
        swing_traction
            .par_iter_mut()
            .zip(force.par_chunks(2))
            .zip(old_force.par_chunks(2))
            .zip(masses.par_iter())
            .for_each(|(((st, f), o), &m)| {
                let m = m.to_f64().unwrap();
                let (fx, fy) = (f[0].to_f64().unwrap(), f[1].to_f64().unwrap());
                let (ox, oy) = (o[0].to_f64().unwrap(), o[1].to_f64().unwrap());
                let s = ((ox - fx).powi(2) + (oy - fy).powi(2)).sqrt();
                let t = ((ox + fx).powi(2) + (oy + fy).powi(2)).sqrt();
                *st = (m * s, 0.5 * m * t);
            });
        let (swing, traction) = swing_traction
            .iter()
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        (speed, efficiency) = update_speed(speed, efficiency, swing, traction, jitter_tolerance, n);
        let speed_t = T::from_f64(speed).unwrap();

        embd.par_iter_mut()
            .zip(force.par_chunks(2))
            .zip(old_force.par_chunks(2))
            .zip(masses.par_iter())
            .for_each(|(((p, f), o), &m)| {
                let dx = o[0] - f[0];
                let dy = o[1] - f[1];
                let swing_i = m * (dx * dx + dy * dy).sqrt();
                let factor = speed_t / (T::one() + (speed_t * swing_i).sqrt());
                p[0] += f[0] * factor;
                p[1] += f[1] * factor;
            });

        std::mem::swap(&mut force, &mut old_force);

        if verbosity.normal_verbosity() && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            println!(
                " Epoch {}/{} | speed = {:.3e} | swing / traction = {:.3}",
                epoch,
                params.n_epochs,
                speed,
                swing / traction.max(f64::MIN_POSITIVE)
            );
        }
    }

    Ok(())
}

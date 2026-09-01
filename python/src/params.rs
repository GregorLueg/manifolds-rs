//! Building the crate's `*Params<T>` structs out of Python dictionaries.
//!
//! Every parameter struct in `manifolds-rs` is generic over the float type and
//! several nest two deep, so binding them as `#[pyclass]` would mean an `F32`
//! and an `F64` twin of each, ten times over. A dictionary crosses the boundary
//! instead and the struct is assembled here, generic over `T`, once.
//!
//! Two rules hold throughout:
//!
//! - **A missing key means the crate's own default.** The Rust `Default` impl
//!   stays the single source of truth for every value the Python layer does not
//!   deliberately override, so no default is written down twice.
//! - **An unknown key is an error, naming the key.** A dictionary is
//!   stringly-typed, and a silently ignored `min_dst` typo would otherwise cost
//!   somebody an afternoon.

use manifolds_rs::prelude::*;
use manifolds_rs::{
    DensmapParams, DensneParams, DiffusionMapsParams, PacmapParams, PhateParams, TsneParams,
    UmapParams,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "gpu")]
use manifolds_rs::{DensmapParamsGpu, TsneParamsGpu, UmapParamsGpu};

///////////////
// Extractor //
///////////////

/// A dictionary being read into one parameter struct.
///
/// Tracks which keys have been consumed so [`Reader::finish`] can report the
/// ones that were not. Borrowing rather than copying the dictionary keeps the
/// whole read allocation-free apart from the string keys themselves.
pub(crate) struct Reader<'a, 'py> {
    /// The dictionary handed in from Python.
    dict: &'a Bound<'py, PyDict>,
    /// Every key this reader has been asked for, consumed or not.
    seen: Vec<&'static str>,
    /// Name of the struct being built, for error messages.
    what: &'static str,
}

impl<'a, 'py> Reader<'a, 'py> {
    /// Start reading `dict` as the parameter struct named `what`.
    ///
    /// ### Params
    ///
    /// * `dict` - The parameters, as sent from the Python layer.
    /// * `what` - Struct name, used only in error messages.
    ///
    /// ### Returns
    ///
    /// A reader positioned at the start of `dict`.
    pub(crate) fn new(dict: &'a Bound<'py, PyDict>, what: &'static str) -> Self {
        Self {
            dict,
            seen: Vec::new(),
            what,
        }
    }

    /// Read one optional key.
    ///
    /// A key that is absent, or present but `None`, both mean "not set". The
    /// Python layer sends `None` for every group field the caller left alone,
    /// so the two have to behave identically.
    ///
    /// ### Params
    ///
    /// * `key` - Field name, matching the Rust struct field exactly.
    ///
    /// ### Returns
    ///
    /// The extracted value, or `None`. Errors if the value is present but of
    /// the wrong type, with the key named.
    pub(crate) fn get<D>(&mut self, key: &'static str) -> PyResult<Option<D>>
    where
        D: for<'i, 'p> FromPyObject<'i, 'p>,
    {
        self.seen.push(key);
        match self.dict.get_item(key)? {
            None => Ok(None),
            Some(v) if v.is_none() => Ok(None),
            Some(v) => v.extract::<D>().map(Some).map_err(|e| {
                // `FromPyObject::Error` is only `Into<PyErr>`, not `Display`.
                let e: PyErr = e.into();
                PyTypeError::new_err(format!("{}: bad value for {key:?}: {e}", self.what))
            }),
        }
    }

    /// Read one optional float key, converting into the generic float type.
    ///
    /// Python floats arrive as `f64`. Narrowing to `f32` when that is the
    /// element type is exactly what the caller asked for by handing in a
    /// float32 design matrix, so it happens here rather than being refused.
    ///
    /// ### Params
    ///
    /// * `key` - Field name, matching the Rust struct field exactly.
    ///
    /// ### Returns
    ///
    /// The value as `T`, or `None` when unset. Errors if the value is not a
    /// number, or is not finite: a NaN parameter poisons an entire embedding
    /// silently, and every knob here is a real number.
    pub(crate) fn float<T>(&mut self, key: &'static str) -> PyResult<Option<T>>
    where
        T: ManifoldsFloat,
    {
        let Some(v) = self.get::<f64>(key)? else {
            return Ok(None);
        };
        if !v.is_finite() {
            return Err(PyTypeError::new_err(format!(
                "{}: {key:?} must be finite, got {v}",
                self.what
            )));
        }
        T::from_f64(v).map(Some).ok_or_else(|| {
            PyTypeError::new_err(format!("{}: {key:?} = {v} is out of range", self.what))
        })
    }

    /// Read a nested parameter dictionary.
    ///
    /// ### Params
    ///
    /// * `key` - Field name of the nested struct.
    ///
    /// ### Returns
    ///
    /// The nested dictionary, or `None` when the caller left the whole group
    /// alone. Errors if the value is present but not a dictionary.
    pub(crate) fn group(&mut self, key: &'static str) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.seen.push(key);
        match self.dict.get_item(key)? {
            None => Ok(None),
            Some(v) if v.is_none() => Ok(None),
            Some(v) => v.cast_into::<PyDict>().map(Some).map_err(|_| {
                PyTypeError::new_err(format!("{}: {key:?} must be a dict", self.what))
            }),
        }
    }

    /// Whether a key is present at all, an explicit `None` included.
    ///
    /// Almost every field treats an absent key and an explicit `None` alike,
    /// because `None` means "use the crate default". Two do not: PHATE's
    /// `decay` and PaCMAP's `range` have non-`None` defaults and a `None` that
    /// means something (a binary connectivity kernel, and no initialisation
    /// range). Those two ask this first.
    ///
    /// ### Params
    ///
    /// * `key` - Field name.
    ///
    /// ### Returns
    ///
    /// Whether the dictionary carries the key.
    pub(crate) fn present(&self, key: &str) -> bool {
        // `get_item` hands back `Some(py_none)` for a key whose value is None,
        // which is exactly the case this has to tell apart from absence.
        self.dict.get_item(key).ok().flatten().is_some()
    }

    /// Reject any key the reader was never asked for.
    ///
    /// ### Returns
    ///
    /// Nothing, or a `TypeError` naming the first unrecognised key and listing
    /// what this struct does accept.
    pub(crate) fn finish(self) -> PyResult<()> {
        for key in self.dict.keys() {
            let name: String = key.extract()?;
            if !self.seen.iter().any(|&s| s == name) {
                let mut known: Vec<&str> = self.seen.clone();
                known.sort_unstable();
                return Err(PyTypeError::new_err(format!(
                    "{}: unknown parameter {name:?}; this struct takes: {}",
                    self.what,
                    known.join(", ")
                )));
            }
        }
        Ok(())
    }
}

/// Apply an optional override in place.
///
/// Saves a `if let Some(v) = ... { field = v }` at every one of roughly a
/// hundred call sites.
///
/// ### Params
///
/// * `slot` - The field to overwrite.
/// * `value` - The override, or `None` to leave the default in place.
fn set<D>(slot: &mut D, value: Option<D>) {
    if let Some(v) = value {
        *slot = v;
    }
}

/// Read a nested group into an already-defaulted struct.
///
/// ### Params
///
/// * `reader` - Reader for the enclosing struct.
/// * `key` - Field name of the nested group.
/// * `slot` - The nested struct, already holding its defaults.
/// * `fill` - Applies one group dictionary to `slot`.
///
/// ### Returns
///
/// Nothing, or the first extraction error from inside the group.
fn nested<S>(
    reader: &mut Reader<'_, '_>,
    key: &'static str,
    slot: &mut S,
    fill: impl FnOnce(&Bound<'_, PyDict>, &mut S) -> PyResult<()>,
) -> PyResult<()> {
    if let Some(group) = reader.group(key)? {
        fill(&group, slot)?;
    }
    Ok(())
}

////////////////////////
// Nested param types //
////////////////////////

/// Fill [`NearestNeighbourParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The `nn_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_nn<T>(d: &Bound<'_, PyDict>, p: &mut NearestNeighbourParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "nn_params");
    set(&mut p.dist_metric, r.get::<String>("dist_metric")?);
    set(&mut p.n_tree, r.get::<usize>("n_tree")?);
    p.search_budget = r.get::<usize>("search_budget")?.or(p.search_budget);
    set(&mut p.m, r.get::<usize>("m")?);
    set(&mut p.ef_construction, r.get::<usize>("ef_construction")?);
    set(&mut p.ef_search, r.get::<usize>("ef_search")?);
    set(&mut p.diversify_prob, r.float::<T>("diversify_prob")?);
    set(&mut p.delta, r.float::<T>("delta")?);
    p.ef_budget = r.get::<usize>("ef_budget")?.or(p.ef_budget);
    set(&mut p.bt_budget, r.float::<T>("bt_budget")?);
    p.n_list = r.get::<usize>("n_list")?.or(p.n_list);
    p.n_probes = r.get::<usize>("n_probes")?.or(p.n_probes);
    r.finish()
}

/// Fill [`UmapGraphParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The `graph_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_umap_graph<T>(d: &Bound<'_, PyDict>, p: &mut UmapGraphParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "graph_params");
    set(&mut p.bandwidth, r.float::<T>("bandwidth")?);
    set(
        &mut p.local_connectivity,
        r.float::<T>("local_connectivity")?,
    );
    set(&mut p.mix_weight, r.float::<T>("mix_weight")?);
    r.finish()
}

/// Fill [`UmapOptimParams`] from a dictionary.
///
/// `min_dist` is deliberately absent: it lives at the top level, where it and
/// `spread` are fed to the crate's own curve fit before this runs. Accepting it
/// here as well would let a caller move it without refitting `a` and `b`, which
/// is the one combination that is always wrong. Pinning `a` and `b` directly
/// stays available for anyone who means it.
///
/// ### Params
///
/// * `d` - The `optim_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_umap_optim<T>(d: &Bound<'_, PyDict>, p: &mut UmapOptimParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "optim_params");
    set(&mut p.a, r.float::<T>("a")?);
    set(&mut p.b, r.float::<T>("b")?);
    set(&mut p.lr, r.float::<T>("lr")?);
    set(&mut p.gamma, r.float::<T>("gamma")?);
    set(&mut p.n_epochs, r.get::<usize>("n_epochs")?);
    set(&mut p.neg_sample_rate, r.get::<usize>("neg_sample_rate")?);
    set(&mut p.beta1, r.float::<T>("beta1")?);
    set(&mut p.beta2, r.float::<T>("beta2")?);
    set(&mut p.eps, r.float::<T>("eps")?);
    r.finish()
}

/// Fill [`TsneOptimParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The `optim_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_tsne_optim<T>(d: &Bound<'_, PyDict>, p: &mut TsneOptimParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "optim_params");
    set(&mut p.n_epochs, r.get::<usize>("n_epochs")?);
    p.lr = r.float::<T>("lr")?.or(p.lr);
    set(&mut p.early_exag_iter, r.get::<usize>("early_exag_iter")?);
    set(&mut p.early_exag_factor, r.float::<T>("early_exag_factor")?);
    p.late_exag_factor = r.float::<T>("late_exag_factor")?.or(p.late_exag_factor);
    set(&mut p.theta, r.float::<T>("theta")?);
    set(&mut p.n_interp_points, r.get::<usize>("n_interp_points")?);
    r.finish()
}

/// Fill [`PacmapOptimParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The `optim_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_pacmap_optim<T>(
    d: &Bound<'_, PyDict>,
    p: &mut PacmapOptimParams<T>,
) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "optim_params");
    set(&mut p.n_epochs, r.get::<usize>("n_epochs")?);
    set(&mut p.lr, r.float::<T>("lr")?);
    set(&mut p.beta1, r.float::<T>("beta1")?);
    set(&mut p.beta2, r.float::<T>("beta2")?);
    set(&mut p.eps, r.float::<T>("eps")?);
    set(&mut p.phase1_end, r.get::<usize>("phase1_end")?);
    set(&mut p.phase2_end, r.get::<usize>("phase2_end")?);
    r.finish()
}

/// Fill [`DensParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The `dens_params` group.
/// * `p` - Struct holding the densMAP or den-SNE defaults, overwritten in
///   place. The two differ only in `lambda`, which is why the caller picks the
///   starting struct rather than this function.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_dens<T>(d: &Bound<'_, PyDict>, p: &mut DensParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "dens_params");
    set(&mut p.lambda, r.float::<T>("lambda")?);
    set(&mut p.frac, r.float::<T>("frac")?);
    set(&mut p.var_shift, r.float::<T>("var_shift")?);
    r.finish()
}

/// Fill [`PhateDiffusionParams`] from a dictionary.
///
/// `t` is the one field that is not a plain scalar: [`PhateTime`] is either
/// `Auto { t_max }` or `Fixed(t)`, and Python sends `None` for the former and
/// an integer for the latter. `t_max` is read separately and only bites in the
/// `Auto` case, which is how [`PhateDiffusionParams::new`] treats it too.
///
/// ### Params
///
/// * `d` - The `diffusion_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
pub(crate) fn fill_phate_diffusion<T>(
    d: &Bound<'_, PyDict>,
    p: &mut PhateDiffusionParams<T>,
) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "diffusion_params");
    // An explicit `None` here selects the binary connectivity kernel, so it
    // cannot be collapsed into "unset" the way every other field's is.
    let decay_given = r.present("decay");
    let decay = r.float::<T>("decay")?;
    if decay_given {
        p.decay = decay;
    }
    set(&mut p.bandwidth_scale, r.float::<T>("bandwidth_scale")?);
    set(&mut p.thresh, r.float::<T>("thresh")?);
    set(&mut p.graph_symmetry, r.get::<String>("graph_symmetry")?);
    p.n_landmarks = r.get::<usize>("n_landmarks")?.or(p.n_landmarks);
    set(&mut p.landmark_method, r.get::<String>("landmark_method")?);
    p.n_svd = r.get::<usize>("n_svd")?.or(p.n_svd);
    set(&mut p.gamma, r.float::<T>("gamma")?);
    p.t = read_time(&mut r, &p.t)?;
    r.finish()
}

/// Read the `t` / `t_max` pair into a [`PhateTime`].
///
/// ### Params
///
/// * `r` - Reader for the enclosing group, so both keys count as consumed.
/// * `current` - The default this struct already carries. Borrowed, since
///   [`PhateTime`] is not `Copy`.
///
/// ### Returns
///
/// `Fixed(t)` when `t` was given, otherwise `Auto` with `t_max` applied to
/// whatever the default already held.
fn read_time(r: &mut Reader<'_, '_>, current: &PhateTime) -> PyResult<PhateTime> {
    let t = r.get::<usize>("t")?;
    let t_max = r.get::<usize>("t_max")?;
    Ok(match (t, t_max, current) {
        (Some(t), _, _) => PhateTime::Fixed(t),
        (None, Some(t_max), _) => PhateTime::Auto { t_max },
        (None, None, PhateTime::Fixed(t)) => PhateTime::Fixed(*t),
        (None, None, PhateTime::Auto { t_max }) => PhateTime::Auto { t_max: *t_max },
    })
}

/////////////////
// Curve seeds //
/////////////////

/// Read `min_dist` and `spread` and hand back UMAP parameters fitted to them.
///
/// UMAP's repulsion curve is parameterised by `a` and `b`, which nobody sets by
/// hand: they come from a least-squares fit against `min_dist` and `spread`.
/// The crate owns that fit, so this seeds from `new_default_2d` rather than
/// reproducing either the fit or its `0.5` / `1.0` defaults. Anything the
/// caller pinned in `optim_params`, `a` and `b` included, is applied on top
/// afterwards.
///
/// ### Params
///
/// * `r` - Reader for the top-level dictionary.
///
/// ### Returns
///
/// Default UMAP parameters with the curve already fitted.
fn seed_umap<T>(r: &mut Reader<'_, '_>) -> PyResult<UmapParams<T>>
where
    T: ManifoldsFloat,
{
    let min_dist = r.float::<T>("min_dist")?;
    let spread = r.float::<T>("spread")?;
    Ok(UmapParams::new_default_2d(min_dist, spread))
}

//////////////////////////
// Top-level CPU params //
//////////////////////////

/// Build [`UmapParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn umap<T>(d: &Bound<'_, PyDict>) -> PyResult<UmapParams<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "UmapParams");
    let mut p = seed_umap::<T>(&mut r)?;
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.k, r.get::<usize>("k")?);
    set(&mut p.optimiser, r.get::<String>("optimiser")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.initialisation, r.get::<String>("initialisation")?);
    p.init_range = r.float::<T>("init_range")?.or(p.init_range);
    set(&mut p.randomised, r.get::<bool>("randomised")?);
    nested(&mut r, "nn_params", &mut p.nn_params, fill_nn)?;
    nested(
        &mut r,
        "umap_graph_params",
        &mut p.umap_graph_params,
        fill_umap_graph,
    )?;
    nested(&mut r, "optim_params", &mut p.optim_params, fill_umap_optim)?;
    r.finish()?;
    Ok(p)
}

/// Build [`DensmapParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn densmap<T>(d: &Bound<'_, PyDict>) -> PyResult<DensmapParams<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "DensmapParams");
    let min_dist = r.float::<T>("min_dist")?;
    let spread = r.float::<T>("spread")?;
    let lambda = r.float::<T>("lambda")?;
    let mut p = DensmapParams::new_default_2d(min_dist, spread, lambda);
    // The UMAP half arrives inline rather than under its own key, so densMAP
    // reads exactly like UMAP from Python with one extra group.
    umap_into(&mut r, &mut p.umap_params)?;
    nested(&mut r, "dens_params", &mut p.dens_params, fill_dens)?;
    r.finish()?;
    Ok(p)
}

/// Build [`TsneParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn tsne<T>(d: &Bound<'_, PyDict>) -> PyResult<TsneParams<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "TsneParams");
    let mut p = TsneParams::new_default_2d(r.float::<T>("perplexity")?);
    tsne_into(&mut r, &mut p)?;
    r.finish()?;
    Ok(p)
}

/// Build [`DensneParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn densne<T>(d: &Bound<'_, PyDict>) -> PyResult<DensneParams<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "DensneParams");
    let perplexity = r.float::<T>("perplexity")?;
    let lambda = r.float::<T>("lambda")?;
    let mut p = DensneParams::new_default_2d(perplexity, lambda);
    tsne_into(&mut r, &mut p.tsne_params)?;
    nested(&mut r, "dens_params", &mut p.dens_params, fill_dens)?;
    r.finish()?;
    Ok(p)
}

/// Build [`PhateParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn phate<T>(d: &Bound<'_, PyDict>) -> PyResult<PhateParams<T>>
where
    T: ManifoldsFloat,
{
    let mut p = PhateParams::<T>::default();
    let mut r = Reader::new(d, "PhateParams");
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.k, r.get::<usize>("k")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.mds_method, r.get::<String>("mds_method")?);
    p.mds_iter = r.get::<usize>("mds_iter")?.or(p.mds_iter);
    set(&mut p.randomised, r.get::<bool>("randomised")?);
    nested(&mut r, "nn_params", &mut p.ann_params, fill_nn)?;
    nested(
        &mut r,
        "diffusion_params",
        &mut p.diffusion_params,
        fill_phate_diffusion,
    )?;
    r.finish()?;
    Ok(p)
}

/// Build [`PacmapParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn pacmap<T>(d: &Bound<'_, PyDict>) -> PyResult<PacmapParams<T>>
where
    T: ManifoldsFloat,
{
    let mut p = PacmapParams::<T>::default();
    let mut r = Reader::new(d, "PacmapParams");
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.optimiser_type, r.get::<String>("optimiser_type")?);
    set(&mut p.n_near, r.get::<usize>("n_near")?);
    set(&mut p.n_mid_near, r.get::<usize>("n_mid_near")?);
    set(&mut p.n_further, r.get::<usize>("n_further")?);
    set(
        &mut p.mn_candidate_start,
        r.get::<usize>("mn_candidate_start")?,
    );
    set(&mut p.mn_candidate_end, r.get::<usize>("mn_candidate_end")?);
    set(&mut p.initialisation, r.get::<String>("initialisation")?);
    // As for PHATE's `decay`: an explicit `None` means no initialisation
    // range, which is not the same as leaving the default of 0.01 in place.
    let range_given = r.present("range");
    let range = r.float::<T>("range")?;
    if range_given {
        p.range = range;
    }
    nested(&mut r, "nn_params", &mut p.nn_params, fill_nn)?;
    nested(
        &mut r,
        "optim_params",
        &mut p.optim_params,
        fill_pacmap_optim,
    )?;
    r.finish()?;
    Ok(p)
}

/// Build [`DiffusionMapsParams`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
pub(crate) fn diffusion_maps<T>(d: &Bound<'_, PyDict>) -> PyResult<DiffusionMapsParams<T>>
where
    T: ManifoldsFloat,
{
    let mut p = DiffusionMapsParams::<T>::default();
    let mut r = Reader::new(d, "DiffusionMapsParams");
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.k, r.get::<usize>("k")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.bandwidth_scale, r.float::<T>("bandwidth_scale")?);
    set(&mut p.thresh, r.float::<T>("thresh")?);
    set(&mut p.graph_symmetry, r.get::<String>("graph_symmetry")?);
    set(&mut p.alpha_norm, r.float::<T>("alpha_norm")?);
    p.n_landmarks = r.get::<usize>("n_landmarks")?.or(p.n_landmarks);
    set(&mut p.landmark_method, r.get::<String>("landmark_method")?);
    p.n_svd = r.get::<usize>("n_svd")?.or(p.n_svd);
    p.t = read_time(&mut r, &p.t)?;
    nested(&mut r, "nn_params", &mut p.ann_params, fill_nn)?;
    r.finish()?;
    Ok(p)
}

/////////////
// Helpers //
/////////////

/// Read the UMAP fields out of an enclosing dictionary.
///
/// densMAP's parameters are UMAP's plus one group, and the Python layer keeps
/// them flat rather than nesting a `umap_params` key nobody would want to type.
/// The reader is shared so both halves count keys against the same dictionary.
///
/// ### Params
///
/// * `r` - Reader for the enclosing dictionary.
/// * `p` - Struct already seeded with the fitted curve, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad key.
fn umap_into<T>(r: &mut Reader<'_, '_>, p: &mut UmapParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.k, r.get::<usize>("k")?);
    set(&mut p.optimiser, r.get::<String>("optimiser")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.initialisation, r.get::<String>("initialisation")?);
    p.init_range = r.float::<T>("init_range")?.or(p.init_range);
    set(&mut p.randomised, r.get::<bool>("randomised")?);
    nested(r, "nn_params", &mut p.nn_params, fill_nn)?;
    nested(
        r,
        "umap_graph_params",
        &mut p.umap_graph_params,
        fill_umap_graph,
    )?;
    nested(r, "optim_params", &mut p.optim_params, fill_umap_optim)?;
    Ok(())
}

/// Read the t-SNE fields out of an enclosing dictionary.
///
/// Shared by [`tsne`] and [`densne`] for the same reason [`umap_into`] is
/// shared: den-SNE's parameters are t-SNE's plus one group. `perplexity` is
/// absent because the seed already read it.
///
/// ### Params
///
/// * `r` - Reader for the enclosing dictionary.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad key.
fn tsne_into<T>(r: &mut Reader<'_, '_>, p: &mut TsneParams<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.initialisation, r.get::<String>("initialisation")?);
    p.init_range = r.float::<T>("init_range")?.or(p.init_range);
    set(&mut p.randomised_init, r.get::<bool>("randomised_init")?);
    nested(r, "nn_params", &mut p.nn_params, fill_nn)?;
    nested(r, "optim_params", &mut p.optim_params, fill_tsne_optim)?;
    Ok(())
}

//////////////////////////
// Top-level GPU params //
//////////////////////////

/// Fill [`NearestNeighbourParamsGpu`] from a dictionary.
///
/// A different struct from the CPU one, not a subset: the GPU backends take
/// CAGRA build degrees and beam-search budgets that have no CPU counterpart.
///
/// ### Params
///
/// * `d` - The `nn_params` group.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad or unknown key.
#[cfg(feature = "gpu")]
pub(crate) fn fill_nn_gpu<T>(
    d: &Bound<'_, PyDict>,
    p: &mut NearestNeighbourParamsGpu<T>,
) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "nn_params");
    set(&mut p.dist_metric, r.get::<String>("dist_metric")?);
    p.n_list = r.get::<usize>("n_list")?.or(p.n_list);
    p.n_probes = r.get::<usize>("n_probes")?.or(p.n_probes);
    p.k = r.get::<usize>("k")?.or(p.k);
    p.k_build = r.get::<usize>("k_build")?.or(p.k_build);
    p.n_tree = r.get::<usize>("n_tree")?.or(p.n_tree);
    set(&mut p.delta, r.float::<T>("delta")?);
    p.rho = r.float::<T>("rho")?.or(p.rho);
    p.beam_width = r.get::<usize>("beam_width")?.or(p.beam_width);
    p.max_beam_iters = r.get::<usize>("max_beam_iters")?.or(p.max_beam_iters);
    p.n_entry_points = r.get::<usize>("n_entry_points")?.or(p.n_entry_points);
    r.finish()
}

/// Build [`UmapParamsGpu`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
#[cfg(feature = "gpu")]
pub(crate) fn umap_gpu<T>(d: &Bound<'_, PyDict>) -> PyResult<UmapParamsGpu<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "UmapParamsGpu");
    let min_dist = r.float::<T>("min_dist")?;
    let spread = r.float::<T>("spread")?;
    let mut p = UmapParamsGpu::new_default_2d(min_dist, spread);
    umap_gpu_into(&mut r, &mut p)?;
    r.finish()?;
    Ok(p)
}

/// Build [`DensmapParamsGpu`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
#[cfg(feature = "gpu")]
pub(crate) fn densmap_gpu<T>(d: &Bound<'_, PyDict>) -> PyResult<DensmapParamsGpu<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "DensmapParamsGpu");
    let min_dist = r.float::<T>("min_dist")?;
    let spread = r.float::<T>("spread")?;
    let lambda = r.float::<T>("lambda")?;
    let mut p = DensmapParamsGpu::new_default_2d(min_dist, spread, lambda);
    umap_gpu_into(&mut r, &mut p.umap_params)?;
    nested(&mut r, "dens_params", &mut p.dens_params, fill_dens)?;
    r.finish()?;
    Ok(p)
}

/// Build [`TsneParamsGpu`] from a dictionary.
///
/// ### Params
///
/// * `d` - The full parameter dictionary sent by the Python layer.
///
/// ### Returns
///
/// A fully specified parameter struct, or the first bad or unknown key.
#[cfg(feature = "gpu")]
pub(crate) fn tsne_gpu<T>(d: &Bound<'_, PyDict>) -> PyResult<TsneParamsGpu<T>>
where
    T: ManifoldsFloat,
{
    let mut r = Reader::new(d, "TsneParamsGpu");
    let mut p = TsneParamsGpu::new_default_2d(r.float::<T>("perplexity")?);
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.initialisation, r.get::<String>("initialisation")?);
    p.init_range = r.float::<T>("init_range")?.or(p.init_range);
    set(&mut p.randomised_init, r.get::<bool>("randomised_init")?);
    nested(&mut r, "nn_params", &mut p.nn_params, fill_nn_gpu)?;
    nested(&mut r, "optim_params", &mut p.optim_params, fill_tsne_optim)?;
    r.finish()?;
    Ok(p)
}

/// Read the GPU UMAP fields, shared by `umap_gpu` and `densmap_gpu`.
///
/// ### Params
///
/// * `r` - Reader for the enclosing dictionary.
/// * `p` - Struct holding the crate defaults, overwritten in place.
///
/// ### Returns
///
/// Nothing, or the first bad key.
#[cfg(feature = "gpu")]
fn umap_gpu_into<T>(r: &mut Reader<'_, '_>, p: &mut UmapParamsGpu<T>) -> PyResult<()>
where
    T: ManifoldsFloat,
{
    set(&mut p.n_dim, r.get::<usize>("n_dim")?);
    set(&mut p.k, r.get::<usize>("k")?);
    set(&mut p.optimiser, r.get::<String>("optimiser")?);
    set(&mut p.ann_type, r.get::<String>("ann_type")?);
    set(&mut p.initialisation, r.get::<String>("initialisation")?);
    p.init_range = r.float::<T>("init_range")?.or(p.init_range);
    set(&mut p.randomised, r.get::<bool>("randomised")?);
    nested(r, "nn_params", &mut p.nn_params, fill_nn_gpu)?;
    nested(
        r,
        "umap_graph_params",
        &mut p.umap_graph_params,
        fill_umap_graph,
    )?;
    nested(r, "optim_params", &mut p.optim_params, fill_umap_optim)?;
    Ok(())
}

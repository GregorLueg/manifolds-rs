//! Raw mlx-c bindings plus a thin RAII layer for the MLX optimisers.
//!
//! Only the functions actually called are declared; signatures are copied
//! from `mlx/c/{array,ops,stream,error,fast}.h` (mlx-c 0.7.0). No `#[link]`
//! here: ann-search-rs owns `links = "mlxc"` and links the library.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::Once;

use crate::prelude::*;

///////////
// Types //
///////////

/// `mlx_array`: an opaque handle around a pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct mlx_array {
    /// Opaque pointer to the C++ array
    ctx: *mut c_void,
}

/// `mlx_stream`: an opaque handle around a pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct mlx_stream {
    /// Opaque pointer to the C++ stream
    ctx: *mut c_void,
}

/// `mlx_vector_array`: an opaque handle around a pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct mlx_vector_array {
    /// Opaque pointer to the C++ vector
    ctx: *mut c_void,
}

/// `mlx_vector_string`: an opaque handle around a pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct mlx_vector_string {
    /// Opaque pointer to the C++ vector
    ctx: *mut c_void,
}

/// `mlx_fast_metal_kernel`: an opaque handle around a pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct mlx_fast_metal_kernel {
    /// Opaque pointer to the C++ kernel
    ctx: *mut c_void,
}

/// `mlx_fast_metal_kernel_config`: an opaque handle around a pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct mlx_fast_metal_kernel_config {
    /// Opaque pointer to the C++ config
    ctx: *mut c_void,
}

/// `mlx_dtype` is a C enum, so an `int` on the ABI.
pub type mlx_dtype = c_int;

/// `MLX_UINT32` in the `mlx_dtype` enum.
pub const MLX_UINT32: mlx_dtype = 3;

/// `MLX_FLOAT32` in the `mlx_dtype` enum.
pub const MLX_FLOAT32: mlx_dtype = 10;

/// Signature of the callback `mlx_set_error_handler` takes.
type mlx_error_handler_func = unsafe extern "C" fn(msg: *const c_char, data: *mut c_void);

unsafe extern "C" {
    fn mlx_array_new() -> mlx_array;
    fn mlx_array_new_data(
        data: *const c_void,
        shape: *const c_int,
        dim: c_int,
        dtype: mlx_dtype,
    ) -> mlx_array;
    fn mlx_array_free(arr: mlx_array) -> c_int;
    fn mlx_array_dtype(arr: mlx_array) -> mlx_dtype;
    fn mlx_array_size(arr: mlx_array) -> usize;
    fn mlx_array_data_float32(arr: mlx_array) -> *const f32;

    fn mlx_eval(outputs: mlx_vector_array) -> c_int;
    fn mlx_async_eval(outputs: mlx_vector_array) -> c_int;
    fn mlx_vector_array_new() -> mlx_vector_array;
    fn mlx_vector_array_new_data(data: *const mlx_array, size: usize) -> mlx_vector_array;
    fn mlx_vector_array_get(res: *mut mlx_array, vec: mlx_vector_array, idx: usize) -> c_int;
    fn mlx_vector_array_free(vec: mlx_vector_array) -> c_int;
    fn mlx_vector_string_new_data(data: *const *const c_char, size: usize) -> mlx_vector_string;
    fn mlx_vector_string_free(vec: mlx_vector_string) -> c_int;

    fn mlx_fast_metal_kernel_new(
        name: *const c_char,
        input_names: mlx_vector_string,
        output_names: mlx_vector_string,
        source: *const c_char,
        header: *const c_char,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> mlx_fast_metal_kernel;
    fn mlx_fast_metal_kernel_free(cls: mlx_fast_metal_kernel);
    fn mlx_fast_metal_kernel_apply(
        outputs: *mut mlx_vector_array,
        cls: mlx_fast_metal_kernel,
        inputs: mlx_vector_array,
        config: mlx_fast_metal_kernel_config,
        stream: mlx_stream,
    ) -> c_int;
    fn mlx_fast_metal_kernel_config_new() -> mlx_fast_metal_kernel_config;
    fn mlx_fast_metal_kernel_config_free(cls: mlx_fast_metal_kernel_config);
    fn mlx_fast_metal_kernel_config_add_output_arg(
        cls: mlx_fast_metal_kernel_config,
        shape: *const c_int,
        size: usize,
        dtype: mlx_dtype,
    ) -> c_int;
    fn mlx_fast_metal_kernel_config_set_grid(
        cls: mlx_fast_metal_kernel_config,
        grid1: c_int,
        grid2: c_int,
        grid3: c_int,
    ) -> c_int;
    fn mlx_fast_metal_kernel_config_set_thread_group(
        cls: mlx_fast_metal_kernel_config,
        thread1: c_int,
        thread2: c_int,
        thread3: c_int,
    ) -> c_int;
    fn mlx_fast_metal_kernel_config_add_template_arg_int(
        cls: mlx_fast_metal_kernel_config,
        name: *const c_char,
        value: c_int,
    ) -> c_int;

    fn mlx_default_gpu_stream_new() -> mlx_stream;
    fn mlx_stream_free(stream: mlx_stream) -> c_int;

    fn mlx_clear_cache() -> c_int;

    fn mlx_set_error_handler(
        handler: mlx_error_handler_func,
        data: *mut c_void,
        dtor: Option<unsafe extern "C" fn(*mut c_void)>,
    );
}

///////////////////
// Error handler //
///////////////////

/// Replaces mlx-c's default handler, which prints and calls `exit(-1)`. This
/// one only prints; the failing call's non-zero return then becomes a
/// [`ManifoldsError::MlxError`].
///
/// ### Params
///
/// * `msg` - NUL-terminated message from mlx-c
/// * `_data` - Unused user data
unsafe extern "C" fn print_error(msg: *const c_char, _data: *mut c_void) {
    if !msg.is_null() {
        // SAFETY: mlx-c hands over a NUL-terminated buffer valid for the call.
        let msg = unsafe { CStr::from_ptr(msg) };
        eprintln!("[MLX] {}", msg.to_string_lossy());
    }
}

/// Install [`print_error`] once per process.
pub fn install_error_handler() {
    static ONCE: Once = Once::new();
    // SAFETY: a plain function pointer with no user data and no destructor.
    ONCE.call_once(|| unsafe { mlx_set_error_handler(print_error, std::ptr::null_mut(), None) });
}

/// Map an mlx-c return code onto the crate error.
///
/// ### Params
///
/// * `code` - Return code, zero on success
/// * `op` - Name of the mlx-c call, for the error message
///
/// ### Returns
///
/// `Ok(())` on zero, `MlxError` otherwise
fn check(code: c_int, op: &'static str) -> Result<(), ManifoldsError> {
    if code == 0 {
        Ok(())
    } else {
        Err(ManifoldsError::MlxError { op, code })
    }
}

/// Hand MLX's cached free buffers back to the system.
///
/// MLX keeps freed device buffers for reuse, and on unified memory they count
/// against the process. Every epoch allocates fresh outputs, so without this
/// the last epochs' buffers stay resident after the optimiser returns.
///
/// ### Returns
///
/// `Ok(())`, or `MlxError` if mlx-c reports a failure
pub fn clear_cache() -> Result<(), ManifoldsError> {
    // SAFETY: no arguments, no preconditions.
    check(unsafe { mlx_clear_cache() }, "mlx_clear_cache")
}

/// Evaluate several arrays in one graph run.
///
/// ### Params
///
/// * `arrays` - Arrays to materialise
/// * `async_` - Submit and return at once (`mlx_async_eval`) instead of
///   waiting. A later blocking eval of the same arrays waits for them.
///
/// ### Returns
///
/// `Ok(())` once submitted (async) or materialised
pub fn eval_all(arrays: &[&Array], async_: bool) -> Result<(), ManifoldsError> {
    let raw: Vec<mlx_array> = arrays.iter().map(|a| a.raw).collect();
    // SAFETY: the vector holds copies of live handles and is freed below.
    unsafe {
        let v = mlx_vector_array_new_data(raw.as_ptr(), raw.len());
        let code = if async_ {
            mlx_async_eval(v)
        } else {
            mlx_eval(v)
        };
        mlx_vector_array_free(v);
        check(code, if async_ { "mlx_async_eval" } else { "mlx_eval" })
    }
}

////////////
// Stream //
////////////

/// Owned MLX stream on the default GPU device.
///
/// MLX streams are thread affine (the GPU command encoder is registered per
/// thread), so the raw pointer keeps this `!Send` and `!Sync` on purpose.
pub struct Stream {
    /// The raw handle
    raw: mlx_stream,
}

impl Stream {
    /// New stream on the default GPU device.
    ///
    /// ### Returns
    ///
    /// The owned stream
    pub fn default_gpu() -> Self {
        // SAFETY: no preconditions; the handle is freed in `Drop`.
        Self {
            raw: unsafe { mlx_default_gpu_stream_new() },
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: owned handle, freed exactly once.
        unsafe { mlx_stream_free(self.raw) };
    }
}

///////////
// Array //
///////////

/// Owned MLX array, freed on drop. Ops are lazy: nothing runs on the device
/// until [`eval_all`].
pub struct Array {
    /// The raw handle
    raw: mlx_array,
}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: owned handle, freed exactly once.
        unsafe { mlx_array_free(self.raw) };
    }
}

impl Array {
    /// Copy a row-major buffer into a new array.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major values, `shape.iter().product()` of them
    /// * `shape` - Array shape
    /// * `dtype` - mlx dtype matching the element type of `data`
    ///
    /// ### Returns
    ///
    /// The owned array
    fn from_raw<E>(data: &[E], shape: &[i32], dtype: mlx_dtype) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().map(|&s| s as usize).product::<usize>()
        );
        // SAFETY: mlx-c copies `data`, so the borrow only needs to outlive
        // the call.
        Self {
            raw: unsafe {
                mlx_array_new_data(
                    data.as_ptr().cast(),
                    shape.as_ptr(),
                    shape.len() as c_int,
                    dtype,
                )
            },
        }
    }

    /// Copy a row-major f32 buffer into a new array.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major values, `shape.iter().product()` of them
    /// * `shape` - Array shape
    ///
    /// ### Returns
    ///
    /// The owned array
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Self {
        Self::from_raw(data, shape, MLX_FLOAT32)
    }

    /// Copy a row-major u32 buffer into a new array.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major values, `shape.iter().product()` of them
    /// * `shape` - Array shape
    ///
    /// ### Returns
    ///
    /// The owned array
    pub fn from_u32(data: &[u32], shape: &[i32]) -> Self {
        Self::from_raw(data, shape, MLX_UINT32)
    }

    /// Borrow the f32 buffer of an evaluated, contiguous array.
    ///
    /// ### Returns
    ///
    /// The values, row-major
    pub fn as_f32(&self) -> Result<&[f32], ManifoldsError> {
        // SAFETY: the dtype check makes the reinterpretation sound; the slice
        // borrows `self`, which keeps the buffer alive.
        unsafe {
            if mlx_array_dtype(self.raw) != MLX_FLOAT32 {
                return Err(ManifoldsError::MlxError {
                    op: "mlx_array_data_float32 (dtype)",
                    code: mlx_array_dtype(self.raw),
                });
            }
            let ptr = mlx_array_data_float32(self.raw);
            if ptr.is_null() {
                return Err(ManifoldsError::MlxError {
                    op: "mlx_array_data_float32",
                    code: -1,
                });
            }
            Ok(std::slice::from_raw_parts(ptr, mlx_array_size(self.raw)))
        }
    }
}

//////////////////
// Metal kernel //
//////////////////

/// Specification of one kernel output.
pub struct OutputSpec<'a> {
    /// Output shape
    pub shape: &'a [i32],
    /// Output element type
    pub dtype: mlx_dtype,
}

/// A custom Metal kernel compiled through MLX's `metal_kernel`. MLX generates
/// the signature from the input and output names and JIT-compiles once per
/// distinct set of template arguments, so only values fixed for a run belong
/// in the template.
pub struct MetalKernel {
    /// The raw handle
    raw: mlx_fast_metal_kernel,
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        // SAFETY: owned handle, freed exactly once.
        unsafe { mlx_fast_metal_kernel_free(self.raw) };
    }
}

impl MetalKernel {
    /// Register a kernel body.
    ///
    /// ### Params
    ///
    /// * `name` - Kernel name
    /// * `inputs` - Input buffer names, in `apply` order
    /// * `outputs` - Output buffer names, in `apply` order
    /// * `header` - Metal source placed before the generated kernel, for
    ///   helper functions; may be empty
    /// * `source` - Metal body; the signature is generated by MLX
    ///
    /// ### Returns
    ///
    /// The kernel handle
    pub fn new(name: &str, inputs: &[&str], outputs: &[&str], header: &str, source: &str) -> Self {
        let cstr = |s: &str| CString::new(s).expect("kernel strings carry no NUL");
        let in_c: Vec<CString> = inputs.iter().map(|s| cstr(s)).collect();
        let out_c: Vec<CString> = outputs.iter().map(|s| cstr(s)).collect();
        let in_p: Vec<*const c_char> = in_c.iter().map(|s| s.as_ptr()).collect();
        let out_p: Vec<*const c_char> = out_c.iter().map(|s| s.as_ptr()).collect();
        let (name, source, header) = (cstr(name), cstr(source), cstr(header));
        // SAFETY: mlx-c copies every string; the vectors are freed after.
        unsafe {
            let in_v = mlx_vector_string_new_data(in_p.as_ptr(), in_p.len());
            let out_v = mlx_vector_string_new_data(out_p.as_ptr(), out_p.len());
            let raw = mlx_fast_metal_kernel_new(
                name.as_ptr(),
                in_v,
                out_v,
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            );
            mlx_vector_string_free(in_v);
            mlx_vector_string_free(out_v);
            Self { raw }
        }
    }

    /// Queue the kernel on a stream (lazy, like any MLX op).
    ///
    /// ### Params
    ///
    /// * `inputs` - Input arrays, in the registered order
    /// * `outputs` - Output specs, in the registered order
    /// * `grid` - Total threads per axis (`dispatchThreads` semantics)
    /// * `threadgroup` - Threads per threadgroup per axis
    /// * `template_ints` - Integer template arguments
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The output arrays, in the registered order
    pub fn apply(
        &self,
        inputs: &[&Array],
        outputs: &[OutputSpec],
        grid: [i32; 3],
        threadgroup: [i32; 3],
        template_ints: &[(&str, i32)],
        s: &Stream,
    ) -> Result<Vec<Array>, ManifoldsError> {
        let raw_in: Vec<mlx_array> = inputs.iter().map(|a| a.raw).collect();
        let names: Vec<CString> = template_ints
            .iter()
            .map(|(n, _)| CString::new(*n).expect("template names carry no NUL"))
            .collect();
        // SAFETY: every handle created here is freed before returning; the
        // outputs move into owned `Array`s.
        unsafe {
            let config = mlx_fast_metal_kernel_config_new();
            let in_v = mlx_vector_array_new_data(raw_in.as_ptr(), raw_in.len());
            let mut out_v = mlx_vector_array_new();
            let mut code = 0;
            for o in outputs {
                code |= mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    o.shape.as_ptr(),
                    o.shape.len(),
                    o.dtype,
                );
            }
            code |= mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]);
            code |= mlx_fast_metal_kernel_config_set_thread_group(
                config,
                threadgroup[0],
                threadgroup[1],
                threadgroup[2],
            );
            for (name, (_, v)) in names.iter().zip(template_ints) {
                code |=
                    mlx_fast_metal_kernel_config_add_template_arg_int(config, name.as_ptr(), *v);
            }
            if code == 0 {
                code = mlx_fast_metal_kernel_apply(&mut out_v, self.raw, in_v, config, s.raw);
            }
            let mut res = Vec::with_capacity(outputs.len());
            if code == 0 {
                for i in 0..outputs.len() {
                    let mut a = Array {
                        raw: mlx_array_new(),
                    };
                    code |= mlx_vector_array_get(&mut a.raw, out_v, i);
                    res.push(a);
                }
            }
            mlx_vector_array_free(in_v);
            mlx_vector_array_free(out_v);
            mlx_fast_metal_kernel_config_free(config);
            check(code, "mlx_fast_metal_kernel_apply")?;
            Ok(res)
        }
    }
}

//! Picks up mlx-c for the experimental `mlx` feature.
//!
//! ann-search-rs owns the `links = "mlxc"` key and does the linking. When it
//! found the library it exports `DEP_MLXC_AVAILABLE=1`, and only then does
//! this emit `cfg(mlx_available)`. Without the library ann-search-rs already
//! warns, so the MLX code here is silently left out.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_MLXC_AVAILABLE");
    println!("cargo::rustc-check-cfg=cfg(mlx_available)");
    if std::env::var_os("CARGO_FEATURE_MLX").is_some()
        && std::env::var("DEP_MLXC_AVAILABLE").as_deref() == Ok("1")
    {
        println!("cargo:rustc-cfg=mlx_available");
    }
}

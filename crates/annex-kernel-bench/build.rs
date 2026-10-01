fn main() {
    if std::env::var_os("CARGO_FEATURE_BLAS").is_some() {
        println!("cargo:rustc-link-lib=openblas");
    }
}

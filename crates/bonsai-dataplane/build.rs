fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        if let Some(root) = std::env::var_os("CUDA_HOME") {
            for dir in ["lib", "lib64"] {
                println!(
                    "cargo:rustc-link-search=native={}",
                    std::path::Path::new(&root).join(dir).display()
                );
            }
        }
    }
}

use std::{env, path::PathBuf, process::Command};
fn run(command: &mut Command) {
    let status = command.status().expect("could not launch CUDA toolchain");
    assert!(
        status.success(),
        "CUDA toolchain command failed: {command:?}"
    );
}
fn main() {
    println!("cargo:rerun-if-changed=kernels/bonsai_codec.cu");
    println!("cargo:rerun-if-changed=kernels/bonsai_codec.cuh");
    for v in ["CUDA_HOME", "CUDA_PATH", "NVCC", "BONSAI_CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = PathBuf::from(
        env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".into()),
    );
    let nvcc = env::var_os("NVCC").map(PathBuf::from).unwrap_or_else(|| {
        let p = cuda.join("bin/nvcc");
        if p.exists() { p } else { PathBuf::from("nvcc") }
    });
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let obj = out.join("bonsai_codec.o");
    let mut cc = Command::new(nvcc);
    cc.args([
        "-std=c++17",
        "-O3",
        "--fmad=false",
        "-Xcompiler",
        "-fPIC",
        "-c",
        "kernels/bonsai_codec.cu",
        "-o",
    ])
    .arg(&obj);
    if let Ok(arch) = env::var("BONSAI_CUDA_ARCH") {
        cc.arg(format!("-arch={arch}"));
    }
    run(&mut cc);
    let lib = out.join("libbonsai_codec_cuda.a");
    run(Command::new("ar").arg("crus").arg(&lib).arg(&obj));
    println!("cargo:rustc-link-search=native={}", out.display());
    for path in [
        cuda.join("lib64"),
        cuda.join("lib"),
        cuda.join("targets/x86_64-linux/lib"),
    ] {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    println!("cargo:rustc-link-lib=static=bonsai_codec_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

//! Builds the CUDA kernel library with nvcc and embeds it in the binary.
//! CUDA toolkit: $CUDA_HOME (or $CUDA_PATH), default /usr/local/cuda; nvcc may be overridden with $NVCC.
use std::{env, path::PathBuf, process::Command};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda = PathBuf::from(env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")).unwrap_or("/usr/local/cuda".into()));
    let nvcc = env::var("NVCC").map(PathBuf::from).unwrap_or(cuda.join("bin/nvcc"));
    let pearl = root.join("third_party/pearl/miner/pearl-gemm");
    let cutlass = pearl.join("third_party/cutlass/include");
    for var in ["CUDA_HOME", "CUDA_PATH", "NVCC"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=kernels");
    assert!(
        cutlass.join("cutlass/cutlass.h").is_file(),
        "third-party sources missing: run `git submodule update --init --recursive`"
    );
    assert!(nvcc.is_file(), "nvcc not found at {} (set CUDA_HOME or NVCC)", nvcc.display());

    let mut includes = vec![format!("-I{}", pearl.join("csrc").display()), format!("-I{}", cutlass.display())];
    let cccl = cuda.join("include/cccl");
    if cccl.is_dir() {
        includes.push(format!("-I{}", cccl.display()));
    }
    let common = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a", "-Xcompiler", "-fPIC"];
    let units: [(&str, &[&str]); 3] = [
        ("kernel780", &["-Xptxas", "-O1"]),
        ("kernel448", &["--ptxas-options=-O3"]),
        ("support", &[]),
    ];
    let jobs: Vec<_> = units
        .iter()
        .map(|(name, extra)| {
            let mut cmd = Command::new(&nvcc);
            cmd.args(common).args(*extra).args(&includes).arg("-c").arg(root.join(format!("kernels/{name}.cu"))).arg("-o").arg(out.join(format!("{name}.o")));
            (name.to_string(), cmd.spawn().unwrap_or_else(|e| panic!("cannot run {}: {e}", nvcc.display())))
        })
        .collect();
    for (name, mut child) in jobs {
        assert!(child.wait().unwrap().success(), "nvcc failed for kernels/{name}.cu");
    }
    let status = Command::new(&nvcc)
        .args(["-shared", "-gencode=arch=compute_120a,code=sm_120a"])
        .args(units.iter().map(|(n, _)| out.join(format!("{n}.o"))))
        .arg(format!("-L{}", cuda.join("lib64/stubs").display()))
        .args(["-lcuda", "-o"])
        .arg(out.join("libpearlkernels.so"))
        .status()
        .expect("nvcc link");
    assert!(status.success(), "linking the kernel library failed");
    // The host binary uses the CUDA driver API (libcuda.so.1 from the NVIDIA driver).
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64/stubs").display());
    println!("cargo:rustc-link-search=native={}", cuda.join("targets/x86_64-linux/lib/stubs").display());
    println!("cargo:rustc-link-lib=dylib=cuda");
}

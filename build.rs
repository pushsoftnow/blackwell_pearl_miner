//! Builds the CUDA kernel library with nvcc and embeds it in the binary.
//! CUDA toolkit: $CUDA_HOME (or $CUDA_PATH), default /usr/local/cuda; nvcc may be overridden with $NVCC.
//! Kernel 780 is compiled once per line of kernels/variants.txt (launch symbol pm_k780_launch_<i>).
use std::{env, fmt::Write as _, fs, path::PathBuf, process::Command};

struct Variant {
    name: String,
    ptxas: Vec<String>,
    knobs: Vec<(String, i32)>,
}

const KNOBS: [&str; 10] = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "PL", "MID"];

fn parse_variants(text: &str) -> Vec<Variant> {
    let mut out: Vec<Variant> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let name = it.next().unwrap().to_string();
        let bad = |what: &str| -> ! { panic!("kernels/variants.txt line {}: {what}", n + 1) };
        if name.len() > 32 || !name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
            bad("invalid variant name");
        }
        if out.iter().any(|v| v.name == name) {
            bad("duplicate variant name");
        }
        let ptxas: Vec<String> = it.next().unwrap_or_else(|| bad("missing ptxas options")).split(',').map(str::to_string).collect();
        if ptxas.iter().any(|o| !o.starts_with('-') || o.contains(char::is_whitespace)) {
            bad("ptxas options must look like -O1 or -O3,-maxrregcount=255");
        }
        let knobs = it
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap_or_else(|| bad("knob must be KNOB=VALUE"));
                if !KNOBS.contains(&k) {
                    bad("unknown knob (P0..P7, PL, MID)");
                }
                (k.to_string(), v.parse().unwrap_or_else(|_| bad("knob value must be an integer")))
            })
            .collect();
        out.push(Variant { name, ptxas, knobs });
    }
    assert!(!out.is_empty(), "kernels/variants.txt lists no variant");
    out
}

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

    let variants = parse_variants(&fs::read_to_string(root.join("kernels/variants.txt")).expect("kernels/variants.txt"));
    let mut includes = vec![format!("-I{}", pearl.join("csrc").display()), format!("-I{}", cutlass.display())];
    let cccl = cuda.join("include/cccl");
    if cccl.is_dir() {
        includes.push(format!("-I{}", cccl.display()));
    }
    let common = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-gencode=arch=compute_120a,code=sm_120a", "-Xcompiler", "-fPIC", "-diag-suppress", "549"];
    // (object name, source, extra arguments)
    let mut units: Vec<(String, &str, Vec<String>)> = vec![
        ("kernel448".into(), "kernel448", vec!["--ptxas-options=-O3".into()]),
        ("support".into(), "support", vec![]),
    ];
    for (i, v) in variants.iter().enumerate() {
        let mut args: Vec<String> = v.ptxas.iter().flat_map(|o| ["-Xptxas".to_string(), o.clone()]).collect();
        args.push(format!("-DK780_NS=k780_v{i}"));
        args.push(format!("-DK780_LAUNCH=pm_k780_launch_{i}"));
        args.extend(v.knobs.iter().map(|(k, val)| format!("-DK780_{k}={val}")));
        units.push((format!("kernel780_v{i}"), "kernel780", args));
    }
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    for chunk in units.chunks(jobs) {
        let children: Vec<_> = chunk
            .iter()
            .map(|(obj, src, extra)| {
                let mut cmd = Command::new(&nvcc);
                cmd.args(common).args(extra).args(&includes).arg("-c").arg(root.join(format!("kernels/{src}.cu"))).arg("-o").arg(out.join(format!("{obj}.o")));
                (obj.clone(), cmd.spawn().unwrap_or_else(|e| panic!("cannot run {}: {e}", nvcc.display())))
            })
            .collect();
        for (obj, mut child) in children {
            assert!(child.wait().unwrap().success(), "nvcc failed for {obj}");
        }
    }
    let status = Command::new(&nvcc)
        .args(["-shared", "-gencode=arch=compute_120a,code=sm_120a"])
        .args(units.iter().map(|(o, _, _)| out.join(format!("{o}.o"))))
        .arg(format!("-L{}", cuda.join("lib64/stubs").display()))
        .args(["-lcuda", "-o"])
        .arg(out.join("libpearlkernels.so"))
        .status()
        .expect("nvcc link");
    assert!(status.success(), "linking the kernel library failed");

    let mut table = String::from("/// Kernel 780 variants: (name, ptxas options and knobs), launch symbol pm_k780_launch_<index>.\npub const VARIANTS: &[(&str, &str)] = &[\n");
    for v in &variants {
        let mut desc = v.ptxas.join(",");
        for (k, val) in &v.knobs {
            write!(desc, " {k}={val}").unwrap();
        }
        writeln!(table, "    ({:?}, {:?}),", v.name, desc).unwrap();
    }
    table.push_str("];\n");
    fs::write(out.join("variants.rs"), table).unwrap();

    // The host binary uses the CUDA driver API (libcuda.so.1 from the NVIDIA driver).
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64/stubs").display());
    println!("cargo:rustc-link-search=native={}", cuda.join("targets/x86_64-linux/lib/stubs").display());
    println!("cargo:rustc-link-lib=dylib=cuda");
}

use std::env;
use std::fs;
use std::process::Command;
use std::path::Path;

fn library_exists(dir: &str, name: &str) -> bool {
    let static_name = format!("lib{name}.a");
    let shared_name = format!("lib{name}.so");
    let path = Path::new(dir);
    path.join(&static_name).exists() || path.join(&shared_name).exists()
}

fn main() {
    let out_dir = env::var("OUT_DIR").expect("Cargo did not set OUT_DIR");
    let batch_size = tuning_value("CUDA_BATCH_SIZE", 1 << 20, &[1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24]);
    let threads_per_block = tuning_value("CUDA_THREADS_PER_BLOCK", 128, &[64, 128, 256, 512]);
    let max_workers = tuning_value("CUDA_MAX_WORKERS", 65_536, &[16_384, 32_768, 65_536, 131_072]);
    let warp_batch = env::var_os("CUDA_WARP_BATCH").is_some_and(|value| value == "1");
    fs::write(
        Path::new(&out_dir).join("cuda_tuning.rs"),
        format!(
            "pub const BATCH_SIZE: u64 = {batch_size};\npub const THREADS_PER_BLOCK: u64 = {threads_per_block};\npub const MAX_WORKERS: usize = {max_workers};\npub const WARP_BATCH_INVERSION: bool = {warp_batch};\n"
        ),
    ).expect("Failed to write CUDA tuning configuration");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let cubin = Path::new(&out_dir).join("secp256k1_search.cubin");
    let mut nvcc_command = Command::new(&nvcc);
    nvcc_command.args(["-std=c++14", "-O3", "-arch=sm_86"]);
    if warp_batch {
        nvcc_command.arg("-DWARP_BATCH_INVERSION=1");
    }
    let status = nvcc_command
        .args(["-cubin", "cuda/secp256k1_search.cu", "-o"])
        .arg(&cubin)
        .status()
        .unwrap_or_else(|error| panic!("Failed to start CUDA compiler `{nvcc}`: {error}"));
    if !status.success() {
        panic!("nvcc compilation of cuda/secp256k1_search.cu failed with status {status}");
    }

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_owned());
    let mut candidates = vec![
        "/tmp/ufsecp-build/src/cuda".to_owned(),
        format!("{cuda_home}/targets/x86_64-linux/lib"),
        format!("{cuda_home}/targets/x86_64-linux/lib/stubs"),
        "/usr/local/cuda-13.3/targets/x86_64-linux/lib".to_owned(),
        "/usr/local/cuda-13.3/targets/x86_64-linux/lib/stubs".to_owned(),
        "/usr/local/cuda/targets/x86_64-linux/lib".to_owned(),
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs".to_owned(),
    ];
    candidates.sort();
    candidates.dedup();

    for dir in &candidates {
        if Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    let custom_cuda_lib = "/tmp/ufsecp-build/src/cuda";
    if Path::new(custom_cuda_lib)
        .join("libsecp256k1_cuda_lib.a")
        .exists()
        || Path::new(custom_cuda_lib)
            .join("libsecp256k1_cuda_lib.so")
            .exists()
    {
        println!("cargo:rustc-link-lib=static=secp256k1_cuda_lib");
    }

    for lib_name in ["cuda", "cudart"] {
        if candidates.iter().any(|dir| library_exists(dir, lib_name)) {
            println!("cargo:rustc-link-lib={lib_name}");
        }
    }

    if library_exists("/usr/lib/x86_64-linux-gnu", "stdc++")
        || library_exists("/usr/lib64", "stdc++")
        || library_exists("/usr/local/lib", "stdc++")
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/secp256k1_search.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_WARP_BATCH");
    println!("cargo:rerun-if-env-changed=CUDA_BATCH_SIZE");
    println!("cargo:rerun-if-env-changed=CUDA_THREADS_PER_BLOCK");
    println!("cargo:rerun-if-env-changed=CUDA_MAX_WORKERS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}

fn tuning_value(name: &str, default: usize, allowed: &[usize]) -> usize {
    let value = env::var(name)
        .ok()
        .map(|value| value.parse::<usize>().unwrap_or_else(|_| panic!("{name} must be an integer")))
        .unwrap_or(default);
    if !allowed.contains(&value) {
        panic!("{name}={value} is not supported; allowed values: {allowed:?}");
    }
    value
}

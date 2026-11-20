//! The CUDA kernel library (built by build.rs), embedded in the binary and loaded from
//! memory (Linux: memfd + dlopen; Windows: a DLL file in %TEMP%\\bpm). Kernel 780 exists in
//! several variants (kernels/variants.txt); the tuning (tune.rs) selects one before any CUDA
//! graph is captured.
use anyhow::{bail, Result};
use crate::platform::Lib;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};

include!(concat!(env!("OUT_DIR"), "/variants.rs"));

type P = *mut c_void;
type C = *const c_void;

static LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pearlkernels.bin"));

pub type K780Fn = unsafe extern "C" fn(C, C, C, C, c_int, c_int, c_int, C, C, P, P, c_int, P) -> c_int;

/// FNV-1a hash of the embedded library: identifies the build in the tuning cache.
pub fn library_hash() -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in LIBRARY {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

static SELECTED: AtomicUsize = AtomicUsize::new(0);

/// Use kernel 780 variant `i` (index into VARIANTS) for all graphs captured from now on.
pub fn select(i: usize) {
    assert!(i < VARIANTS.len());
    SELECTED.store(i, Ordering::Relaxed);
}
pub fn selected() -> usize {
    SELECTED.load(Ordering::Relaxed)
}

pub struct Kernels {
    pub k780_variants: Vec<K780Fn>,
    pub k448: unsafe extern "C" fn(C, C, C, C, P, P, c_int, P) -> c_int,
    pub merkle_init: unsafe extern "C" fn(C, P, c_int, P) -> c_int,
    pub merkle_root: unsafe extern "C" fn(C, C, C, P, c_int, P) -> c_int,
    pub advance_nonce: unsafe extern "C" fn(P, P, c_int, P) -> c_int,
    pub build_indices: unsafe extern "C" fn(C, P, P, c_int, P) -> c_int,
    pub pack_basis: unsafe extern "C" fn(C, P, P) -> c_int,
    pub prepare_packed: unsafe extern "C" fn(C, C, C, P, P, c_int, c_int, P) -> c_int,
    pub commit: unsafe extern "C" fn(C, C, C, P, P, c_int, c_int, c_int, P) -> c_int,
    pub noise: unsafe extern "C" fn(P, P, P, P, C, C, c_int, c_int, c_int, c_int, P) -> c_int,
    /// FP8 kernel f780 (--test-fp8) (pa, pb, m, n, k, key, bound, count, hits, cap, digests, grid, stream, cyc)
    pub f780: unsafe extern "C" fn(C, C, c_int, c_int, c_int, C, C, P, P, c_int, P, c_int, P, P) -> c_int,
}
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

fn sym<T: Copy>(lib: &Lib, name: &str) -> Result<T> {
    lib.sym(name).map_err(|_| anyhow::anyhow!("missing kernel symbol {name}"))
}

impl Kernels {
    pub fn load() -> Result<Self> {
        let lib = Lib::open_bytes(LIBRARY, &format!("pearlkernels-{}", library_hash()))?;
        let h = &lib;
        let k = Kernels {
            k780_variants: (0..VARIANTS.len()).map(|i| sym(h, &format!("pm_k780_launch_{i}"))).collect::<Result<_>>()?,
            k448: sym(h, "pm_k448_launch")?,
            merkle_init: sym(h, "pm_merkle_init")?,
            merkle_root: sym(h, "pm_merkle_root")?,
            advance_nonce: sym(h, "pm_advance_nonce")?,
            build_indices: sym(h, "pm_build_indices")?,
            pack_basis: sym(h, "pm_pack_basis")?,
            prepare_packed: sym(h, "pm_prepare_packed")?,
            commit: sym(h, "pm_commit")?,
            noise: sym(h, "pm_noise")?,
            f780: sym(h, "pm_f780_launch")?,
        };
        lib.leak();
        Ok(k)
    }

    /// The selected kernel 780 variant.
    pub fn k780(&self) -> K780Fn {
        self.k780_variants[selected()]
    }
}

/// Loads `pm_k780_launch` from an external candidate library (tools/tune-wizard.py).
pub fn load_candidate(path: &std::path::Path) -> Result<K780Fn> {
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    let tag = format!("candidate-{:016x}", bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3)));
    let lib = Lib::open_bytes(&bytes, &tag)?;
    let f = sym(&lib, "pm_k780_launch")?;
    lib.leak();
    Ok(f)
}

static CELL: std::sync::OnceLock<std::result::Result<Kernels, String>> = std::sync::OnceLock::new();

/// The process-wide kernel library (loaded once).
pub fn get() -> Result<&'static Kernels> {
    match CELL.get_or_init(|| Kernels::load().map_err(|e| format!("{e:#}"))) {
        Ok(k) => Ok(k),
        Err(e) => bail!("{e}"),
    }
}

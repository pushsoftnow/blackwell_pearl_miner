//! The CUDA kernel library (built by build.rs), embedded in the binary and loaded from
//! memory with memfd_create + dlopen. Kernel 780 exists in several variants (kernels/variants.txt);
//! the tuning (tune.rs) selects one before any CUDA graph is captured.
use anyhow::{bail, Result};
use std::ffi::{c_void, CString};
use std::os::raw::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};

include!(concat!(env!("OUT_DIR"), "/variants.rs"));

type P = *mut c_void;
type C = *const c_void;

static LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libpearlkernels.so"));

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
    _handle: *mut c_void,
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
}
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

fn open() -> Result<*mut c_void> {
    open_bytes(LIBRARY)
}

fn open_bytes(bytes: &[u8]) -> Result<*mut c_void> {
    unsafe {
        let name = CString::new("pearl-kernels")?;
        let fd = libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC);
        if fd < 0 {
            bail!("memfd_create failed");
        }
        let mut written = 0;
        while written < bytes.len() {
            let n = libc::write(fd, bytes[written..].as_ptr().cast(), bytes.len() - written);
            if n <= 0 {
                libc::close(fd);
                bail!("cannot write the embedded kernel library");
            }
            written += n as usize;
        }
        // The memfd stays open: the loader identifies libraries by path.
        let path = CString::new(format!("/proc/self/fd/{fd}"))?;
        let handle = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            libc::close(fd);
            let err = std::ffi::CStr::from_ptr(libc::dlerror()).to_string_lossy().into_owned();
            bail!("cannot load the kernel library: {err}");
        }
        Ok(handle)
    }
}

fn sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T> {
    let c = CString::new(name)?;
    let p = unsafe { libc::dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        bail!("missing kernel symbol {name}");
    }
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

impl Kernels {
    pub fn load() -> Result<Self> {
        let h = open()?;
        Ok(Kernels {
            _handle: h,
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
        })
    }

    /// The selected kernel 780 variant.
    pub fn k780(&self) -> K780Fn {
        self.k780_variants[selected()]
    }
}

/// Loads `pm_k780_launch` from an external candidate library (tools/tune-wizard.py).
pub fn load_candidate(path: &std::path::Path) -> Result<K780Fn> {
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    let h = open_bytes(&bytes)?;
    sym(h, "pm_k780_launch")
}

static CELL: std::sync::OnceLock<std::result::Result<Kernels, String>> = std::sync::OnceLock::new();

/// The process-wide kernel library (loaded once).
pub fn get() -> Result<&'static Kernels> {
    match CELL.get_or_init(|| Kernels::load().map_err(|e| format!("{e:#}"))) {
        Ok(k) => Ok(k),
        Err(e) => bail!("{e}"),
    }
}

//! Operating-system specifics: shared libraries, local time, cache directory.
//! Linux and Windows; everything else in the miner is portable.
use anyhow::{bail, Result};
use std::ffi::c_void;
use std::path::PathBuf;

/// A loaded shared library (kept for the lifetime of the process unless dropped).
pub struct Lib(*mut c_void);
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

#[cfg(unix)]
mod imp {
    use super::*;
    use std::ffi::{CStr, CString};

    pub fn open(name: &str) -> Result<*mut c_void> {
        let c = CString::new(name)?;
        let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if h.is_null() {
            let err = unsafe { CStr::from_ptr(libc::dlerror()) }.to_string_lossy().into_owned();
            bail!("cannot load {name}: {err}");
        }
        Ok(h)
    }

    /// Loads a library image from memory (memfd: nothing is written to disk).
    pub fn open_bytes(bytes: &[u8], _tag: &str) -> Result<*mut c_void> {
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
            open(&format!("/proc/self/fd/{fd}")).inspect_err(|_| {
                libc::close(fd);
            })
        }
    }

    pub fn sym(h: *mut c_void, name: &str) -> *mut c_void {
        let Ok(c) = CString::new(name) else { return std::ptr::null_mut() };
        unsafe { libc::dlsym(h, c.as_ptr()) }
    }

    pub fn close(h: *mut c_void) {
        unsafe { libc::dlclose(h) };
    }

    pub fn local_time(t: i64) -> [i32; 6] {
        let t = t as libc::time_t;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe { libc::localtime_r(&t, &mut tm) };
        [tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec]
    }

    pub fn cpu_seconds() -> f64 {
        let mut u: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
        let t = |tv: libc::timeval| tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6;
        t(u.ru_utime) + t(u.ru_stime)
    }

    pub fn cache_dir() -> Option<PathBuf> {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::os::windows::ffi::OsStrExt;

    #[repr(C)]
    struct SystemTime {
        year: u16,
        month: u16,
        weekday: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        ms: u16,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(h: *mut c_void) -> i32;
        fn GetLastError() -> u32;
        fn GetLocalTime(t: *mut SystemTime);
        fn FileTimeToSystemTime(ft: *const u64, st: *mut SystemTime) -> i32;
        fn SystemTimeToTzSpecificLocalTime(tz: *const c_void, utc: *const SystemTime, local: *mut SystemTime) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(p: *mut c_void, creation: *mut u64, exit: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
    }

    pub fn cpu_seconds() -> f64 {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        if unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) } == 0 {
            return 0.0;
        }
        (k + u) as f64 / 1e7
    }

    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    pub fn open(name: &str) -> Result<*mut c_void> {
        let h = unsafe { LoadLibraryW(wide(std::ffi::OsStr::new(name)).as_ptr()) };
        if h.is_null() {
            bail!("cannot load {name} (Windows error {})", unsafe { GetLastError() });
        }
        Ok(h)
    }

    /// Windows loads DLLs only from files: the image goes to %TEMP%\bpm\<tag>.dll
    /// (written once per build, name keyed by the image hash).
    pub fn open_bytes(bytes: &[u8], tag: &str) -> Result<*mut c_void> {
        let dir = std::env::temp_dir().join("bpm");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{tag}.dll"));
        if std::fs::read(&path).map(|b| b != bytes).unwrap_or(true) {
            let tmp = dir.join(format!("{tag}.{}.tmp", std::process::id()));
            std::fs::write(&tmp, bytes)?;
            if std::fs::rename(&tmp, &path).is_err() {
                // In use by another miner instance: identical content, keep it.
                let _ = std::fs::remove_file(&tmp);
            }
        }
        open(&path.to_string_lossy())
    }

    pub fn sym(h: *mut c_void, name: &str) -> *mut c_void {
        let mut c = name.as_bytes().to_vec();
        c.push(0);
        unsafe { GetProcAddress(h, c.as_ptr()) }
    }

    pub fn close(h: *mut c_void) {
        unsafe { FreeLibrary(h) };
    }

    pub fn local_time(t: i64) -> [i32; 6] {
        // FILETIME: 100 ns since 1601-01-01 UTC.
        let ft = (t.max(0) as u64 + 11_644_473_600) * 10_000_000;
        let mut utc: SystemTime = unsafe { std::mem::zeroed() };
        let mut st: SystemTime = unsafe { std::mem::zeroed() };
        unsafe {
            if FileTimeToSystemTime(&ft, &mut utc) == 0 || SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut st) == 0 {
                GetLocalTime(&mut st);
            }
        }
        [st.year as i32, st.month as i32, st.day as i32, st.hour as i32, st.minute as i32, st.second as i32]
    }

    pub fn cache_dir() -> Option<PathBuf> {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from).filter(|p| p.is_absolute())
    }
}

impl Lib {
    /// Opens a library by name (system search path) or path.
    pub fn open(name: &str) -> Result<Lib> {
        Ok(Lib(imp::open(name)?))
    }
    /// Loads a library image held in memory; `tag` names it where a file is required.
    pub fn open_bytes(bytes: &[u8], tag: &str) -> Result<Lib> {
        Ok(Lib(imp::open_bytes(bytes, tag)?))
    }
    /// Function pointer `name`; T must be an `extern "C" fn` type.
    pub fn sym<T: Copy>(&self, name: &str) -> Result<T> {
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        let p = imp::sym(self.0, name);
        if p.is_null() {
            bail!("missing symbol {name}");
        }
        Ok(unsafe { std::mem::transmute_copy(&p) })
    }
    /// Keep the library loaded until the process exits.
    pub fn leak(self) {
        std::mem::forget(self);
    }
}

impl Drop for Lib {
    fn drop(&mut self) {
        imp::close(self.0);
    }
}

/// Local time of a Unix timestamp: [year, month, day, hour, minute, second].
pub fn local_time(t: i64) -> [i32; 6] {
    imp::local_time(t)
}

/// CPU time (user + system) of this process in seconds.
pub fn cpu_seconds() -> f64 {
    imp::cpu_seconds()
}

pub fn now_unix() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Per-user cache directory: $XDG_CACHE_HOME or ~/.cache (Linux), %LOCALAPPDATA% (Windows).
pub fn cache_dir() -> Option<PathBuf> {
    imp::cache_dir()
}

/// File name of the NVML library shipped with the driver.
pub const NVML: &str = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };

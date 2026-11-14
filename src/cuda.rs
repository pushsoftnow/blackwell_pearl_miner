//! Minimal CUDA driver API (libcuda.so.1, shipped with the driver). All host-side
//! memory, streams, graphs and events are managed here; kernels run from the embedded
//! kernel libraries on the same primary context.
#![allow(non_camel_case_types)]
use anyhow::{bail, Result};
use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int, c_uint};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUdeviceptr = u64;

const CU_STREAM_NON_BLOCKING: c_uint = 1;
const CU_EVENT_BLOCKING_SYNC: c_uint = 1;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(dev: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetPCIBusId(id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuGetErrorString(err: CUresult, s: *mut *const c_char) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuMemHostAlloc(p: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    fn cuMemsetD8_v2(p: CUdeviceptr, v: u8, n: usize) -> CUresult;
    fn cuMemsetD8Async(p: CUdeviceptr, v: u8, n: usize, s: CUstream) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: CUstream) -> CUresult;
    fn cuStreamCreate(s: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamDestroy_v2(s: CUstream) -> CUresult;
    fn cuStreamSynchronize(s: CUstream) -> CUresult;
    fn cuStreamBeginCapture_v2(s: CUstream, mode: c_int) -> CUresult;
    fn cuStreamEndCapture(s: CUstream, g: *mut CUgraph) -> CUresult;
    fn cuGraphInstantiateWithFlags(e: *mut CUgraphExec, g: CUgraph, flags: u64) -> CUresult;
    fn cuGraphLaunch(e: CUgraphExec, s: CUstream) -> CUresult;
    fn cuGraphExecDestroy(e: CUgraphExec) -> CUresult;
    fn cuGraphDestroy(g: CUgraph) -> CUresult;
    fn cuEventCreate(e: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventRecord(e: CUevent, s: CUstream) -> CUresult;
    fn cuEventSynchronize(e: CUevent) -> CUresult;
    fn cuEventDestroy_v2(e: CUevent) -> CUresult;
    fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;
    fn cuDriverGetVersion(v: *mut c_int) -> CUresult;
}

pub fn check(rc: CUresult, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    unsafe { cuGetErrorString(rc, &mut s) };
    let text = if s.is_null() { "unknown".into() } else { unsafe { CStr::from_ptr(s) }.to_string_lossy() };
    bail!("CUDA {what} failed: {text} ({rc})")
}

/// Kernel launcher return code: cudaError_t from the CUDA runtime inside a kernel library.
pub fn check_kernel(rc: c_int, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("kernel launcher {what} failed with CUDA runtime error {rc}");
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct Device {
    pub ordinal: i32,
    ctx: CUcontext,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct DeviceInfo {
    pub name: String,
    pub pci_bus_id: String,
    pub sms: i32,
    pub capability: (i32, i32),
    pub total: usize,
}

impl Device {
    pub fn open(ordinal: i32) -> Result<Self> {
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev = 0;
            check(cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
            let mut ctx = std::ptr::null_mut();
            // The primary context is shared with the CUDA runtime inside the kernel libraries.
            check(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            let d = Device { ordinal: dev, ctx };
            d.bind()?;
            Ok(d)
        }
    }
    /// Make the primary context current on the calling thread.
    pub fn bind(&self) -> Result<()> {
        check(unsafe { cuCtxSetCurrent(self.ctx) }, "cuCtxSetCurrent")
    }
    pub fn info(&self) -> Result<DeviceInfo> {
        let mut name = [0 as c_char; 128];
        let attr = |a| -> Result<i32> {
            let mut v = 0;
            check(unsafe { cuDeviceGetAttribute(&mut v, a, self.ordinal) }, "cuDeviceGetAttribute")?;
            Ok(v)
        };
        let (mut _free, mut total) = (0, 0);
        let mut bus = [0 as c_char; 32];
        unsafe {
            check(cuDeviceGetName(name.as_mut_ptr(), 128, self.ordinal), "cuDeviceGetName")?;
            check(cuDeviceGetPCIBusId(bus.as_mut_ptr(), 32, self.ordinal), "cuDeviceGetPCIBusId")?;
            check(cuMemGetInfo_v2(&mut _free, &mut total), "cuMemGetInfo")?;
        }
        Ok(DeviceInfo {
            name: unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned(),
            pci_bus_id: unsafe { CStr::from_ptr(bus.as_ptr()) }.to_string_lossy().into_owned(),
            sms: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            capability: (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?, attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?),
            total,
        })
    }
}

/// Owned device allocation.
pub struct DevBuf {
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}
unsafe impl Send for DevBuf {}
impl DevBuf {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = 0;
        check(unsafe { cuMemAlloc_v2(&mut ptr, bytes.max(1)) }, "cuMemAlloc")?;
        Ok(DevBuf { ptr, bytes })
    }
    pub fn filled(bytes: usize, value: u8) -> Result<Self> {
        let b = Self::new(bytes)?;
        check(unsafe { cuMemsetD8_v2(b.ptr, value, bytes) }, "cuMemsetD8")?;
        Ok(b)
    }
    pub fn p(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }
    pub fn upload(&self, data: &[u8]) -> Result<()> {
        assert!(data.len() <= self.bytes);
        check(unsafe { cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), data.len()) }, "cuMemcpyHtoD")
    }
    pub fn download(&self, out: &mut [u8]) -> Result<()> {
        assert!(out.len() <= self.bytes);
        check(unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, out.len()) }, "cuMemcpyDtoH")
    }
}
impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.ptr) };
    }
}

/// Pinned host buffer for asynchronous device-to-host copies.
pub struct HostBuf {
    pub ptr: *mut u8,
    pub bytes: usize,
}
unsafe impl Send for HostBuf {}
impl HostBuf {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { cuMemHostAlloc(&mut p, bytes, 0) }, "cuMemHostAlloc")?;
        Ok(HostBuf { ptr: p.cast(), bytes })
    }
    pub fn slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }
}
impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe { cuMemFreeHost(self.ptr.cast()) };
    }
}

pub struct Stream(pub CUstream);
unsafe impl Send for Stream {}
impl Stream {
    pub fn new() -> Result<Self> {
        let mut s = std::ptr::null_mut();
        check(unsafe { cuStreamCreate(&mut s, CU_STREAM_NON_BLOCKING) }, "cuStreamCreate")?;
        Ok(Stream(s))
    }
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
    pub fn sync(&self) -> Result<()> {
        check(unsafe { cuStreamSynchronize(self.0) }, "cuStreamSynchronize")
    }
    pub fn memset(&self, p: CUdeviceptr, v: u8, n: usize) -> Result<()> {
        check(unsafe { cuMemsetD8Async(p, v, n, self.0) }, "cuMemsetD8Async")
    }
    pub fn copy_d2d(&self, dst: CUdeviceptr, src: CUdeviceptr, n: usize) -> Result<()> {
        check(unsafe { cuMemcpyDtoDAsync_v2(dst, src, n, self.0) }, "cuMemcpyDtoDAsync")
    }
    pub fn copy_d2h(&self, dst: *mut u8, src: CUdeviceptr, n: usize) -> Result<()> {
        check(unsafe { cuMemcpyDtoHAsync_v2(dst.cast(), src, n, self.0) }, "cuMemcpyDtoHAsync")
    }
    pub fn begin_capture(&self) -> Result<()> {
        check(unsafe { cuStreamBeginCapture_v2(self.0, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) }, "cuStreamBeginCapture")
    }
    pub fn end_capture(&self) -> Result<Graph> {
        let mut g = std::ptr::null_mut();
        check(unsafe { cuStreamEndCapture(self.0, &mut g) }, "cuStreamEndCapture")?;
        let mut e = std::ptr::null_mut();
        let rc = unsafe { cuGraphInstantiateWithFlags(&mut e, g, 0) };
        if rc != 0 {
            unsafe { cuGraphDestroy(g) };
            check(rc, "cuGraphInstantiate")?;
        }
        Ok(Graph { graph: g, exec: e })
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            cuStreamSynchronize(self.0);
            cuStreamDestroy_v2(self.0);
        }
    }
}

pub struct Graph {
    graph: CUgraph,
    exec: CUgraphExec,
}
unsafe impl Send for Graph {}
impl Graph {
    pub fn launch(&self, s: &Stream) -> Result<()> {
        check(unsafe { cuGraphLaunch(self.exec, s.0) }, "cuGraphLaunch")
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            cuGraphExecDestroy(self.exec);
            cuGraphDestroy(self.graph);
        }
    }
}

pub struct Event(CUevent);
unsafe impl Send for Event {}
impl Event {
    /// Blocking-sync event: the waiting CPU thread sleeps instead of spinning.
    pub fn new() -> Result<Self> {
        let mut e = std::ptr::null_mut();
        check(unsafe { cuEventCreate(&mut e, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING) }, "cuEventCreate")?;
        Ok(Event(e))
    }
    /// Event with timing, for kernel timing (blocking sync as well).
    pub fn timed() -> Result<Self> {
        let mut e = std::ptr::null_mut();
        check(unsafe { cuEventCreate(&mut e, CU_EVENT_BLOCKING_SYNC) }, "cuEventCreate")?;
        Ok(Event(e))
    }
    /// Milliseconds from `start` to `self` (both recorded and completed, both timed).
    pub fn since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0f32;
        check(unsafe { cuEventElapsedTime(&mut ms, start.0, self.0) }, "cuEventElapsedTime")?;
        Ok(ms)
    }
    pub fn record(&self, s: &Stream) -> Result<()> {
        check(unsafe { cuEventRecord(self.0, s.0) }, "cuEventRecord")
    }
    pub fn sync(&self) -> Result<()> {
        check(unsafe { cuEventSynchronize(self.0) }, "cuEventSynchronize")
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cuEventDestroy_v2(self.0) };
    }
}

/// CUDA driver API version, e.g. 13020 for 13.2.
pub fn driver_version() -> Result<i32> {
    let mut v = 0;
    check(unsafe { cuDriverGetVersion(&mut v) }, "cuDriverGetVersion")?;
    Ok(v)
}

pub fn memory_info() -> Result<(usize, usize)> {
    let (mut f, mut t) = (0, 0);
    check(unsafe { cuMemGetInfo_v2(&mut f, &mut t) }, "cuMemGetInfo")?;
    Ok((f, t))
}

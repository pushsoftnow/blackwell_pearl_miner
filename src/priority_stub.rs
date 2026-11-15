//! --background on systems without /proc (Windows): pauses while another process shows SM
//! utilisation >= threshold in NVML's per-process samples, resumes after `resume_delay` quiet
//! seconds. --priority PATH needs the process tree and is Linux-only for now.
use crate::nvml::Nvml;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub trait GpuProcesses: Send {
    fn active(&mut self) -> Result<HashMap<u32, u32>>;
}

impl GpuProcesses for Nvml {
    fn active(&mut self) -> Result<HashMap<u32, u32>> {
        Nvml::active(self)
    }
}

pub struct Options {
    pub resume_delay: f64,
    pub threshold: u32,
    pub interval: Duration,
    pub background: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { resume_delay: 10.0, threshold: 5, interval: Duration::from_millis(100), background: false }
    }
}

pub struct PriorityWatch {
    holder: Arc<Mutex<Option<Value>>>,
    pub busy: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<()>>,
}

fn wall() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl PriorityWatch {
    pub fn start(paths: &[PathBuf], opts: Options, mut source: Box<dyn GpuProcesses>) -> Result<Self> {
        if !paths.is_empty() {
            bail!("--priority PATH is supported on Linux only; use --background on this system");
        }
        if !opts.background {
            bail!("no priority program given");
        }
        let (busy, stop) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (holder, error) = (Arc::new(Mutex::new(None)), Arc::new(Mutex::new(None)));
        let (b, s, h, e) = (busy.clone(), stop.clone(), holder.clone(), error.clone());
        let me = std::process::id();
        let thread = std::thread::spawn(move || {
            let mut last_seen: Option<Instant> = None;
            while !s.load(Ordering::SeqCst) {
                std::thread::sleep(opts.interval);
                let now = Instant::now();
                let active = match source.active() {
                    Ok(a) => a,
                    Err(err) => {
                        *e.lock().unwrap() = Some(format!("{err:#}"));
                        return;
                    }
                };
                let mut found: Vec<(u32, u32)> = active.into_iter().filter(|&(pid, util)| pid != me && util >= opts.threshold).collect();
                found.sort();
                if let Some(&(pid, util)) = found.first() {
                    let mut g = h.lock().unwrap();
                    let detected = if b.load(Ordering::SeqCst) { g.as_ref().and_then(|v: &Value| v["detected_at"].as_f64()).unwrap_or_else(wall) } else { wall() };
                    *g = Some(json!({"pid":pid,"path":format!("pid {pid}"),"sm_util":util,"signal":"gpu","detected_at":detected}));
                    last_seen = Some(now);
                    b.store(true, Ordering::SeqCst);
                } else if b.load(Ordering::SeqCst) && last_seen.map(|t| now.duration_since(t).as_secs_f64() >= opts.resume_delay).unwrap_or(true) {
                    b.store(false, Ordering::SeqCst);
                }
            }
        });
        Ok(PriorityWatch { holder, busy, stop, error, thread: Some(thread) })
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    pub fn holder(&self) -> Value {
        self.holder.lock().unwrap().clone().unwrap_or(json!({}))
    }

    /// Slowed rounds alone never pause mining here (no per-process CPU signal).
    pub fn confirm(&self, _slowdown: f64) -> bool {
        false
    }

    pub fn check(&self) -> Result<()> {
        if let Some(e) = &*self.error.lock().unwrap() {
            bail!("priority watch failed: {e}");
        }
        Ok(())
    }
}

impl Drop for PriorityWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

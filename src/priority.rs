//! --priority: yield the GPU while a named program (script or binary) is computing on it.
//! --background: the same for every other process with a GPU context, no names needed.
//!
//! Activity, not presence: a priority program may keep its context and memory on the GPU
//! permanently (e.g. an LLM server); the miner yields only while it is working and resumes
//! after `resume_delay` quiet seconds. CPU load alone never pauses mining. Signals:
//! - fast (~0.2 s): the miner's own rounds become slower (another context time-slices the
//!   GPU) WHILE a priority process holding a GPU context (or a priority-program ancestor,
//!   e.g. a server process above its GPU worker) used CPU within the last `cpu_window` s; the
//!   miner reports the slowdown via `confirm()`;
//! - fallback and hold: SM utilisation >= threshold in NVML per-process samples (lags
//!   ~0.5-1.5 s); it also keeps the pause alive, since paused rounds cannot slow down.
//!
//! A GPU process belongs to a priority program when it or an ancestor runs that file: its
//! executable or a script argument (`bash run.sh`, `python tool.py`). Scripts that end with
//! `exec binary` keep their PID: every process seen running a priority script is remembered
//! (PID + start time); a script that execs at once is caught via inotify IN_OPEN (processes
//! started within the second before the open are remembered).
use crate::nvml::Nvml;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub trait GpuProcesses: Send {
    fn pids(&mut self) -> Result<HashSet<u32>>;
    fn active(&mut self) -> Result<HashMap<u32, u32>>;
}

impl GpuProcesses for Nvml {
    fn pids(&mut self) -> Result<HashSet<u32>> {
        Nvml::pids(self)
    }
    fn active(&mut self) -> Result<HashMap<u32, u32>> {
        Nvml::active(self)
    }
}

fn clk_tck() -> f64 {
    unsafe { libc::sysconf(libc::_SC_CLK_TCK) as f64 }
}

fn read(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

/// Fields after the parenthesised command name of /proc/<pid>/stat.
fn stat_fields(pid: u32) -> Option<Vec<String>> {
    let raw = read(&format!("/proc/{pid}/stat"))?;
    let close = raw.iter().rposition(|&b| b == b')')?;
    Some(String::from_utf8_lossy(&raw[close + 2..]).split_whitespace().map(str::to_owned).collect())
}

pub fn parent(pid: u32) -> Option<u32> {
    stat_fields(pid)?.get(1)?.parse().ok()
}

pub fn start_time(pid: u32) -> Option<u64> {
    stat_fields(pid)?.get(19)?.parse().ok()
}

/// (start time, utime + stime over all threads).
fn cpu_ticks(pid: u32) -> Option<(u64, u64)> {
    let f = stat_fields(pid)?;
    Some((f.get(19)?.parse().ok()?, f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?))
}

fn uptime_ticks() -> Option<u64> {
    let raw = read("/proc/uptime")?;
    let s = String::from_utf8_lossy(&raw);
    Some((s.split_whitespace().next()?.parse::<f64>().ok()? * clk_tck()) as u64)
}

/// Files a process runs: its executable and its first few path-like arguments.
pub fn process_files(pid: u32) -> HashSet<PathBuf> {
    let mut files = HashSet::new();
    if let Ok(exe) = std::fs::read_link(format!("/proc/{pid}/exe")) {
        if let Ok(p) = std::fs::canonicalize(&exe) {
            files.insert(p);
        } else {
            files.insert(exe);
        }
    }
    if let Some(raw) = read(&format!("/proc/{pid}/cmdline")) {
        let cwd = std::fs::read_link(format!("/proc/{pid}/cwd")).ok();
        for arg in raw.split(|&b| b == 0).take(4) {
            let arg = String::from_utf8_lossy(arg);
            if arg.is_empty() || arg.starts_with('-') {
                continue;
            }
            let path = Path::new(arg.as_ref());
            let full = if path.is_absolute() { path.to_path_buf() } else if let Some(c) = &cwd { c.join(path) } else { continue };
            if let Ok(p) = std::fs::canonicalize(&full) {
                files.insert(p);
            }
        }
    }
    files
}

type Remembered = HashMap<u32, (u64, PathBuf)>;

/// Matching priority file for pid or an ancestor, else None.
pub fn owner(mut pid: u32, targets: &HashSet<PathBuf>, remembered: &Remembered) -> Option<PathBuf> {
    for _ in 0..32 {
        if pid <= 1 {
            return None;
        }
        if let Some(hit) = process_files(pid).intersection(targets).min() {
            return Some(hit.clone());
        }
        if let Some((started, path)) = remembered.get(&pid) {
            if Some(*started) == start_time(pid) {
                return Some(path.clone());
            }
        }
        pid = parent(pid)?;
    }
    None
}

fn wall() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

struct Inner {
    /// Background mode: every other GPU process has priority.
    all: bool,
    targets: HashSet<PathBuf>,
    names: Vec<Vec<u8>>,
    source: Box<dyn GpuProcesses>,
    resume_delay: f64,
    threshold: u32,
    cpu_threshold: f64,
    cpu_window: f64,
    remembered: Remembered,
    watched: HashMap<u32, PathBuf>,
    cpu_prev: HashMap<u32, (u64, u64, Instant)>,
    cpu_seen: Option<Instant>,
    cpu_holder: Option<Value>,
    last_scan: Option<Instant>,
    last_gpu: Option<Instant>,
    last_seen: Option<Instant>,
    holder: Option<Value>,
}

pub struct PriorityWatch {
    inner: Arc<Mutex<Inner>>,
    pub busy: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    threads: Vec<JoinHandle<()>>,
    inotify_fd: i32,
}

pub struct Options {
    pub resume_delay: f64,
    pub threshold: u32,
    pub cpu_threshold: f64,
    pub cpu_window: f64,
    pub interval: Duration,
    pub background: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { resume_delay: 10.0, threshold: 5, cpu_threshold: 0.3, cpu_window: 0.5, interval: Duration::from_millis(50), background: false }
    }
}

impl Inner {
    fn me() -> u32 {
        std::process::id()
    }

    /// Program a GPU process belongs to: a priority file, or in background mode its executable.
    fn owner_of(&self, pid: u32) -> Option<PathBuf> {
        if self.all {
            return Some(std::fs::read_link(format!("/proc/{pid}/exe")).ok().unwrap_or_else(|| {
                let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
                PathBuf::from(if comm.trim().is_empty() { format!("pid {pid}") } else { comm.trim().to_owned() })
            }));
        }
        owner(pid, &self.targets, &self.remembered)
    }

    fn remember(&mut self) {
        if self.targets.is_empty() {
            return;
        }
        let mut alive: Remembered = HashMap::new();
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for e in entries.flatten() {
                let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
                let Some(raw) = read(&format!("/proc/{pid}/cmdline")) else { continue };
                if !self.names.iter().any(|n| raw.windows(n.len()).any(|w| w == n.as_slice())) {
                    continue;
                }
                if let Some(hit) = process_files(pid).intersection(&self.targets).min() {
                    if let Some(st) = start_time(pid) {
                        alive.insert(pid, (st, hit.clone()));
                    }
                }
            }
        }
        for (pid, (st, path)) in &self.remembered {
            if start_time(*pid) == Some(*st) {
                alive.entry(*pid).or_insert((*st, path.clone()));
            }
        }
        self.remembered = alive;
    }

    /// A priority file was opened: remember processes started just before (fork -> exec).
    fn opened(&mut self, target: &Path) {
        let Some(now) = uptime_ticks() else { return };
        let low = now.saturating_sub(clk_tck() as u64);
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for e in entries.flatten() {
                let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
                if pid == Self::me() {
                    continue;
                }
                if let Some(st) = start_time(pid) {
                    if st >= low && st <= now + 1 {
                        self.remembered.entry(pid).or_insert((st, target.to_path_buf()));
                    }
                }
            }
        }
    }

    /// Priority processes with a GPU context, plus their priority-program ancestors.
    fn resident(&mut self) -> Result<HashMap<u32, PathBuf>> {
        let mut watched = HashMap::new();
        let mut pids: Vec<u32> = self.source.pids()?.into_iter().collect();
        pids.sort();
        for pid in pids {
            if pid == Self::me() {
                continue;
            }
            let Some(m) = self.owner_of(pid) else { continue };
            watched.insert(pid, m.clone());
            if self.all {
                continue;
            }
            let mut up = parent(pid);
            while let Some(u) = up.filter(|&u| u > 1) {
                let known = self.remembered.get(&u).map(|(st, _)| Some(*st) == start_time(u)).unwrap_or(false);
                if process_files(u).intersection(&self.targets).next().is_none() && !known {
                    break;
                }
                watched.insert(u, m.clone());
                up = parent(u);
            }
        }
        Ok(watched)
    }

    fn cpu_active(&mut self, now: Instant) -> Option<Value> {
        let mut holder: Option<(f64, Value)> = None;
        let mut current = HashMap::new();
        for (pid, path) in &self.watched {
            let Some((st, ticks)) = cpu_ticks(*pid) else { continue };
            current.insert(*pid, (st, ticks, now));
            if let Some((pst, pticks, pt)) = self.cpu_prev.get(pid) {
                let dt = now.duration_since(*pt).as_secs_f64();
                if *pst == st && dt > 0.0 {
                    let cores = ticks.saturating_sub(*pticks) as f64 / clk_tck() / dt;
                    if cores >= self.cpu_threshold && holder.as_ref().map(|(c, _)| cores > *c).unwrap_or(true) {
                        holder = Some((cores, json!({"pid":pid,"path":path.display().to_string(),"cpu_cores":(cores*100.0).round()/100.0,"signal":"cpu"})));
                    }
                }
            }
        }
        self.cpu_prev = current;
        holder.map(|(_, v)| v)
    }

    fn trigger(&mut self, mut holder: Value, now: Instant, busy: &AtomicBool) {
        let detected = if busy.load(Ordering::SeqCst) {
            self.holder.as_ref().and_then(|h| h["detected_at"].as_f64()).unwrap_or_else(wall)
        } else {
            wall()
        };
        holder["detected_at"] = json!(detected);
        self.holder = Some(holder);
        self.last_seen = Some(now);
        busy.store(true, Ordering::SeqCst);
    }

    fn poll(&mut self, now: Instant, busy: &AtomicBool) -> Result<bool> {
        if self.last_scan.map(|t| now.duration_since(t).as_secs_f64() >= 0.5).unwrap_or(true) {
            self.remember();
            self.last_scan = Some(now);
            self.watched = self.resident()?;
        }
        if let Some(cpu) = self.cpu_active(now) {
            self.cpu_seen = Some(now);
            self.cpu_holder = Some(cpu);
        }
        let mut holder = None;
        if self.last_gpu.map(|t| now.duration_since(t).as_secs_f64() >= 0.1).unwrap_or(true) {
            self.last_gpu = Some(now);
            let mut active: Vec<(u32, u32)> = self.source.active()?.into_iter().collect();
            active.sort();
            for (pid, util) in active {
                if pid == Self::me() || util < self.threshold {
                    continue;
                }
                if let Some(m) = self.owner_of(pid) {
                    holder = Some(json!({"pid":pid,"path":m.display().to_string(),"sm_util":util,"signal":"gpu"}));
                    break;
                }
            }
        }
        if let Some(h) = holder {
            self.trigger(h, now, busy);
        } else if busy.load(Ordering::SeqCst) && self.last_seen.map(|t| now.duration_since(t).as_secs_f64() >= self.resume_delay).unwrap_or(true) {
            busy.store(false, Ordering::SeqCst);
        }
        Ok(busy.load(Ordering::SeqCst))
    }
}

impl PriorityWatch {
    pub fn start(paths: &[PathBuf], opts: Options, source: Box<dyn GpuProcesses>) -> Result<Self> {
        let mut targets = HashSet::new();
        for p in paths {
            let r = std::fs::canonicalize(p).map_err(|_| anyhow::anyhow!("--priority: no such file: {}", p.display()))?;
            if !r.is_file() {
                bail!("--priority: not a file: {}", p.display());
            }
            targets.insert(r);
        }
        if targets.is_empty() && !opts.background {
            bail!("no priority program given");
        }
        let names = targets.iter().map(|t| t.file_name().unwrap().as_encoded_bytes().to_vec()).collect();
        let inner = Arc::new(Mutex::new(Inner {
            all: opts.background,
            targets: targets.clone(),
            names,
            source,
            resume_delay: opts.resume_delay,
            threshold: opts.threshold,
            cpu_threshold: opts.cpu_threshold,
            cpu_window: opts.cpu_window,
            remembered: HashMap::new(),
            watched: HashMap::new(),
            cpu_prev: HashMap::new(),
            cpu_seen: None,
            cpu_holder: None,
            last_scan: None,
            last_gpu: None,
            last_seen: None,
            holder: None,
        }));
        let busy = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        inner.lock().unwrap().poll(Instant::now(), &busy)?;
        let mut threads = Vec::new();
        // inotify IN_OPEN on the directories of the priority files.
        let fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if fd < 0 {
            bail!("inotify_init1 failed");
        }
        let mut watches: HashMap<i32, HashMap<Vec<u8>, PathBuf>> = HashMap::new();
        for t in &targets {
            let dir = std::ffi::CString::new(t.parent().unwrap().as_os_str().as_encoded_bytes())?;
            let wd = unsafe { libc::inotify_add_watch(fd, dir.as_ptr(), libc::IN_OPEN) };
            if wd < 0 {
                bail!("inotify watch failed: {}", t.parent().unwrap().display());
            }
            watches.entry(wd).or_default().insert(t.file_name().unwrap().as_encoded_bytes().to_vec(), t.clone());
        }
        {
            let (inner, stop) = (inner.clone(), stop.clone());
            threads.push(std::thread::Builder::new().name("pearl-priority-open".into()).spawn(move || {
                let mut buf = vec![0u8; 65536];
                while !stop.load(Ordering::SeqCst) {
                    let mut pfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
                    if unsafe { libc::poll(&mut pfd, 1, 200) } <= 0 {
                        continue;
                    }
                    let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
                    if n <= 0 {
                        continue;
                    }
                    let mut off = 0usize;
                    while off + 16 <= n as usize {
                        let wd = i32::from_ne_bytes(buf[off..off + 4].try_into().unwrap());
                        let len = u32::from_ne_bytes(buf[off + 12..off + 16].try_into().unwrap()) as usize;
                        let name: Vec<u8> = buf[off + 16..off + 16 + len].iter().copied().take_while(|&b| b != 0).collect();
                        off += 16 + len;
                        if let Some(t) = watches.get(&wd).and_then(|m| m.get(&name)) {
                            inner.lock().unwrap().opened(t);
                        }
                    }
                }
            })?);
        }
        {
            let (inner, stop, busy, error) = (inner.clone(), stop.clone(), busy.clone(), error.clone());
            threads.push(std::thread::Builder::new().name("pearl-priority".into()).spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(opts.interval);
                    if let Err(e) = inner.lock().unwrap().poll(Instant::now(), &busy) {
                        *error.lock().unwrap() = Some(format!("{e:#}"));
                        busy.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            })?);
        }
        Ok(PriorityWatch { inner, busy, stop, error, threads, inotify_fd: fd })
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    pub fn holder(&self) -> Value {
        self.inner.lock().unwrap().holder.clone().unwrap_or(json!({}))
    }

    /// Miner saw a slowed round. Pause only if a priority process was CPU-active.
    pub fn confirm(&self, slowdown: f64) -> bool {
        let now = Instant::now();
        let mut g = self.inner.lock().unwrap();
        let (Some(seen), Some(holder)) = (g.cpu_seen, g.cpu_holder.clone()) else { return false };
        if now.duration_since(seen).as_secs_f64() > g.cpu_window {
            return false;
        }
        let mut h = holder;
        h["signal"] = json!("cpu+slowdown");
        h["slowdown"] = json!((slowdown * 1000.0).round() / 1000.0);
        g.trigger(h, now, &self.busy);
        true
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
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        unsafe { libc::close(self.inotify_fd) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command};

    struct Fake {
        on_gpu: Arc<Mutex<(HashSet<u32>, u32)>>,
    }
    impl GpuProcesses for Fake {
        fn pids(&mut self) -> Result<HashSet<u32>> {
            Ok(self.on_gpu.lock().unwrap().0.clone())
        }
        fn active(&mut self) -> Result<HashMap<u32, u32>> {
            let g = self.on_gpu.lock().unwrap();
            Ok(g.0.iter().map(|p| (*p, g.1)).collect())
        }
    }

    fn script(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join("prio.sh");
        std::fs::write(&p, format!("#!/bin/bash\n{body}\n")).unwrap();
        Command::new("chmod").arg("+x").arg(&p).status().unwrap();
        p
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pearl-prio-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn watch(path: &Path, util: u32, resume: f64) -> (PriorityWatch, Arc<Mutex<(HashSet<u32>, u32)>>) {
        let on = Arc::new(Mutex::new((HashSet::new(), util)));
        let w = PriorityWatch::start(&[path.to_path_buf()], Options { resume_delay: resume, ..Default::default() }, Box::new(Fake { on_gpu: on.clone() })).unwrap();
        (w, on)
    }

    fn kill(mut c: Child) {
        let _ = c.kill();
        let _ = c.wait();
    }

    fn wait_for(f: impl Fn() -> bool, secs: f64) -> bool {
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < secs {
            if f() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn gpu_use_of_child_pauses_and_resumes() {
        let d = tmp("child");
        let p = script(&d, "sleep 30");
        let child = Command::new(&p).spawn().unwrap();
        let pid = child.id();
        assert!(wait_for(|| std::fs::read_to_string("/proc/self/stat").is_ok() && Command::new("pgrep").args(["-P", &pid.to_string()]).output().map(|o| !o.stdout.is_empty()).unwrap_or(false), 3.0));
        let sleeper: u32 = String::from_utf8(Command::new("pgrep").args(["-P", &pid.to_string()]).output().unwrap().stdout).unwrap().trim().parse().unwrap();
        let (w, on) = watch(&p, 50, 0.3);
        assert!(!w.is_busy());
        on.lock().unwrap().0.insert(sleeper);
        assert!(wait_for(|| w.is_busy(), 2.0));
        assert_eq!(w.holder()["pid"], sleeper);
        on.lock().unwrap().0.clear();
        assert!(wait_for(|| !w.is_busy(), 2.0));
        let _ = Command::new("pkill").args(["-P", &pid.to_string()]).status();
        kill(child);
    }

    #[test]
    fn immediate_exec_caught_by_inotify() {
        let d = tmp("exec");
        let p = script(&d, "exec sleep 30");
        let (w, on) = watch(&p, 50, 0.3);
        let child = Command::new(&p).spawn().unwrap();
        let pid = child.id();
        assert!(wait_for(|| !std::fs::read(format!("/proc/{pid}/cmdline")).map(|c| c.windows(7).any(|x| x == b"prio.sh")).unwrap_or(true), 3.0));
        std::thread::sleep(Duration::from_millis(300));
        on.lock().unwrap().0.insert(pid);
        assert!(wait_for(|| w.is_busy(), 2.0));
        kill(child);
    }

    #[test]
    fn cpu_alone_never_pauses_but_confirms_slowdown() {
        let d = tmp("cpu");
        let p = script(&d, "while true; do :; done");
        let child = Command::new("bash").arg(&p).spawn().unwrap();
        let (w, on) = watch(&p, 0, 10.0);
        on.lock().unwrap().0.insert(child.id());
        std::thread::sleep(Duration::from_millis(900));
        assert!(!w.is_busy(), "CPU load alone must keep mining");
        assert!(w.confirm(1.4) && w.is_busy());
        assert_eq!(w.holder()["signal"], "cpu+slowdown");
        kill(child);
    }

    #[test]
    fn background_yields_to_any_gpu_process() {
        let child = Command::new("bash").args(["-c", "while true; do :; done"]).spawn().unwrap();
        let on = Arc::new(Mutex::new((HashSet::new(), 0)));
        let w = PriorityWatch::start(&[], Options { resume_delay: 0.3, background: true, ..Default::default() }, Box::new(Fake { on_gpu: on.clone() })).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(!w.confirm(1.4), "process without GPU context must not count");
        on.lock().unwrap().0.insert(child.id());
        std::thread::sleep(Duration::from_millis(900));
        assert!(!w.is_busy(), "CPU load alone must keep mining");
        assert!(w.confirm(1.4) && w.is_busy());
        assert!(w.holder()["path"].as_str().unwrap().ends_with("bash"));
        assert!(wait_for(|| !w.is_busy(), 2.0));
        on.lock().unwrap().1 = 50;
        assert!(wait_for(|| w.is_busy(), 2.0));
        assert_eq!(w.holder()["signal"], "gpu");
        on.lock().unwrap().0.clear();
        assert!(wait_for(|| !w.is_busy(), 2.0));
        kill(child);
    }

    #[test]
    fn slowdown_without_priority_cpu_is_ignored() {
        let sleep = PathBuf::from(String::from_utf8(Command::new("which").arg("sleep").output().unwrap().stdout).unwrap().trim());
        let child = Command::new(&sleep).arg("30").spawn().unwrap();
        let (w, on) = watch(&sleep, 0, 10.0);
        on.lock().unwrap().0.insert(child.id());
        std::thread::sleep(Duration::from_millis(600));
        assert!(!w.confirm(1.4) && !w.is_busy());
        kill(child);
    }
}

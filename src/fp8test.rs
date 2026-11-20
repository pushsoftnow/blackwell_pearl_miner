//! `--test-fp8`. Compares today's kernel 780 (INT7, certificate v3)
//! with the experimental FP8 kernel f780 (PIP-3, certificate v4) on this GPU and prints preliminary hashrates.
//!
//! 1. f780 is checked bit for bit against an embedded reference job (noised FP8 operands, jackpot key and the
//!    expected 80 lottery tickets, produced with pearl `zk-pow`, branch fp8-scheme @ 1dd75a52).
//! 2. Both kernels run alternately in blocks (A B B A ...), so clock, temperature and power drift hit both alike.
//!    Kernel 780 uses the production profile (K = 65536) on the self-test job; f780 runs real noised FP8 codes
//!    (tiled from the reference job) at M = 1600, N = 10496, K = 16384 with 16x32 lottery tiles.
//! 3. Hashrate = MAC/s for both certificates (the share target scales with the work of a ticket).
use crate::cuda::{self, DevBuf, Event, Stream};
use crate::engine::Engine;
use crate::kernels;
use crate::nvml::Nvml;
use crate::protocol::{Kernel, Profile};
use anyhow::{bail, ensure, Result};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const REF_A: &[u8] = include_bytes!("../testdata/fp8/a.bin");
const REF_B: &[u8] = include_bytes!("../testdata/fp8/b.bin");
const REF_KEY: &[u8] = include_bytes!("../testdata/fp8/jkey.bin");
const REF_TICKETS: &[u8] = include_bytes!("../testdata/fp8/tickets.bin");
const REF_M: usize = 160;
const REF_N: usize = 256;
const REF_K: usize = 1024;
const TILE_H: usize = 16;
const TILE_W: usize = 32;
const SPEC: &str = "pearl fp8-scheme @ 1dd75a52 (PR #311, 2026-11-18)";
const RULE: &str = "==========================================================================================";
const THIN: &str = "------------------------------------------------------------------------------------------";

// f780 CTA tile (kernels/f780.cu).
const BM: usize = 160;
const BN: usize = 256;

/// Row-major (rows x k) e4m3 -> f780 stage layout (same as f780::pack on the GPU).
fn pack(src: &[u8], rows: usize, k: usize, is_b: bool) -> Vec<u8> {
    let rpb = if is_b { BN } else { BM };
    let (s_steps, frags) = (k / 32, rpb / 16);
    let mut dst = vec![0u8; rows * k];
    let ld = |r: usize, c: usize| -> [u8; 4] { src[r * k + c..r * k + c + 4].try_into().unwrap() };
    for blk in 0..rows / rpb {
        for s in 0..s_steps {
            for f in 0..frags {
                for lane in 0..32 {
                    let (g, t) = (lane >> 2, lane & 3);
                    let (r0, r1, c0) = (blk * rpb + f * 16 + g, blk * rpb + f * 16 + g + 8, s * 32 + 4 * t);
                    let w = if is_b { [ld(r0, c0), ld(r0, c0 + 16), ld(r1, c0), ld(r1, c0 + 16)] } else { [ld(r0, c0), ld(r1, c0), ld(r0, c0 + 16), ld(r1, c0 + 16)] };
                    let o = ((blk * s_steps + s) * frags + f) * 512 + lane * 16;
                    for (i, x) in w.iter().enumerate() {
                        dst[o + 4 * i..o + 4 * i + 4].copy_from_slice(x);
                    }
                }
            }
        }
    }
    dst
}

struct F780 {
    pa: DevBuf,
    pb: DevBuf,
    key: DevBuf,
    bound: DevBuf,
    result: DevBuf,
    digests: Option<DevBuf>,
    m: usize,
    n: usize,
    k: usize,
}

impl F780 {
    fn new(a: &[u8], b: &[u8], key: &[u8], m: usize, n: usize, k: usize, digests: bool) -> Result<Self> {
        let pa = DevBuf::new(m * k)?;
        pa.upload(&pack(a, m, k, false))?;
        let pb = DevBuf::new(n * k)?;
        pb.upload(&pack(b, n, k, true))?;
        let kb = DevBuf::new(32)?;
        kb.upload(key)?;
        // Bound with the top word 0x07ffffff: about 1/32 of the tickets hit (checks the compare path too).
        let mut bound = [0xffu8; 32];
        bound[28..32].copy_from_slice(&0x07ff_ffffu32.to_le_bytes());
        let bb = DevBuf::new(32)?;
        bb.upload(&bound)?;
        let result = DevBuf::filled(4 * 4097, 0)?;
        let digests = if digests { Some(DevBuf::filled((m / TILE_H) * (n / TILE_W) * 32, 0)?) } else { None };
        Ok(F780 { pa, pb, key: kb, bound: bb, result, digests, m, n, k })
    }
    /// One launch; returns kernel milliseconds.
    fn run(&self, f: &kernels::Kernels, s: &Stream) -> Result<f32> {
        let (start, end) = (Event::timed()?, Event::timed()?);
        s.memset(self.result.ptr, 0, 4)?;
        start.record(s)?;
        let rc = unsafe {
            (f.f780)(self.pa.p(), self.pb.p(), self.m as i32, self.n as i32, self.k as i32, self.key.p(), self.bound.p(), self.result.p(),
                (self.result.ptr + 4) as *mut std::ffi::c_void, 4096, self.digests.as_ref().map_or(std::ptr::null_mut(), |d| d.p()), 0, s.raw(), std::ptr::null_mut())
        };
        cuda::check_kernel(rc, "kernel f780")?;
        end.record(s)?;
        end.sync()?;
        end.since(&start)
    }
    /// `n` launches queued back to back (no host sync in between, like the miner would run them); one sync at
    /// the end. Returns wall-clock milliseconds of the whole batch.
    fn run_batch(&self, f: &kernels::Kernels, s: &Stream, n: usize) -> Result<f64> {
        let t0 = Instant::now();
        for _ in 0..n {
            s.memset(self.result.ptr, 0, 4)?;
            let rc = unsafe {
                (f.f780)(self.pa.p(), self.pb.p(), self.m as i32, self.n as i32, self.k as i32, self.key.p(), self.bound.p(), self.result.p(),
                    (self.result.ptr + 4) as *mut std::ffi::c_void, 4096, std::ptr::null_mut(), 0, s.raw(), std::ptr::null_mut())
            };
            cuda::check_kernel(rc, "kernel f780")?;
        }
        s.sync()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    }
    fn macs(&self) -> f64 {
        self.m as f64 * self.n as f64 * self.k as f64
    }
}

/// Bit-exact check of f780 on this GPU against the embedded reference tickets.
fn verify(f: &kernels::Kernels, s: &Stream) -> Result<usize> {
    let job = F780::new(REF_A, REF_B, REF_KEY, REF_M, REF_N, REF_K, true)?;
    job.run(f, s)?;
    let tiles = (REF_M / TILE_H) * (REF_N / TILE_W);
    let mut got = vec![0u8; tiles * 32];
    job.digests.as_ref().unwrap().download(&mut got)?;
    let bad = (0..tiles).filter(|&t| got[t * 32..t * 32 + 32] != REF_TICKETS[t * 32..t * 32 + 32]).count();
    let mut cnt = [0u8; 4];
    job.result.download(&mut cnt)?;
    let expected = (0..tiles).filter(|&t| u32::from_le_bytes(REF_TICKETS[t * 32 + 28..t * 32 + 32].try_into().unwrap()) <= 0x07ff_ffff).count();
    ensure!(bad == 0, "f780 differs from the reference on {bad} of {tiles} lottery tiles: FP8 results on this GPU would be invalid");
    ensure!(u32::from_le_bytes(cnt) as usize == expected, "f780 reports {} hits, the reference {expected}", u32::from_le_bytes(cnt));
    Ok(tiles)
}

#[derive(Default, Clone)]
struct Tele {
    w: Vec<f64>,
    mhz: Vec<f64>,
    temp: Vec<f64>,
}

/// Diagnostic trace (env BPM_FP8_TRACE=1): unix time, event, values on stderr.
fn trace(ev: &str) {
    if std::env::var_os("BPM_FP8_TRACE").is_some() {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
        eprintln!("TRACE {t:.3} {ev}");
    }
}

/// Live progress line while the test runs (terminal only): spinner, bar, phase and remaining time,
/// redrawn in place every 125 ms; `clear` removes it before a normal line is printed.
struct Ticker {
    st: Arc<Mutex<TickState>>,
    stop: Arc<AtomicBool>,
    h: Option<std::thread::JoinHandle<()>>,
}

struct TickState {
    on: bool,
    head: String,
    label: String,
    t0: Instant,
    dur: f64,
}

const TICK_W: usize = 90;

impl Ticker {
    fn new() -> Self {
        use std::io::{IsTerminal, Write};
        let now = Instant::now();
        let st = Arc::new(Mutex::new(TickState { on: false, head: String::new(), label: String::new(), t0: now, dur: 1.0 }));
        let stop = Arc::new(AtomicBool::new(false));
        let h = std::io::stdout().is_terminal().then(|| {
            let (st, stop) = (st.clone(), stop.clone());
            std::thread::spawn(move || {
                const SPIN: [char; 4] = ['|', '/', '-', '\\'];
                let mut i = 0usize;
                while !stop.load(Ordering::SeqCst) {
                    {
                        let g = st.lock().unwrap();
                        if g.on {
                            let f = (g.t0.elapsed().as_secs_f64() / g.dur).clamp(0.0, 1.0);
                            let n = (f * 20.0).round() as usize;
                            let line = format!("         {:<11} {} [{}{}] {:>3} %  {}",
                                g.head, SPIN[i % 4], "#".repeat(n), "-".repeat(20 - n), (f * 100.0) as u32, g.label);
                            let mut o = std::io::stdout().lock();
                            let _ = write!(o, "\r{line:<TICK_W$}");
                            let _ = o.flush();
                        }
                    }
                    i += 1;
                    std::thread::sleep(Duration::from_millis(125));
                }
            })
        });
        Ticker { st, stop, h }
    }
    /// Start a progress phase: `head` (e.g. "round 1/6"), bar over `dur` seconds.
    fn phase(&self, head: &str, dur: f64) {
        let mut g = self.st.lock().unwrap();
        (g.on, g.head, g.t0, g.dur) = (true, head.to_string(), Instant::now(), dur);
    }
    fn label(&self, label: &str) {
        self.st.lock().unwrap().label = label.to_string();
    }
    /// Remove the progress line (the next println starts at column 0).
    fn clear(&self) {
        use std::io::Write;
        let mut g = self.st.lock().unwrap();
        if g.on && self.h.is_some() {
            let mut o = std::io::stdout().lock();
            let _ = write!(o, "\r{:TICK_W$}\r", "");
            let _ = o.flush();
        }
        g.on = false;
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        self.clear();
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

pub fn run(device: i32, seconds: f64) -> Result<()> {
    let dev = cuda::Device::open(device)?;
    let info = dev.info()?;
    ensure!(info.capability == (12, 0), "the kernels are built for compute capability 12.0 (found {} with {}.{})", info.name, info.capability.0, info.capability.1);
    let nvml = Nvml::open(&info.pci_bus_id).ok();
    let limit = nvml.as_ref().and_then(|n| n.power_limit_w().ok());
    println!("{RULE}");
    println!(" bpm --test-fp8   Pearl FP8 hard fork (PIP-3): hashrate impact on this GPU   [PRELIMINARY]");
    println!("{RULE}");
    println!(" GPU          {} ({} SMs), power limit {}", info.name, info.sms, limit.map_or("unknown".into(), |w| format!("{w:.0} W")));
    println!("{THIN}");
    let support = kernels::get()?;
    let stream = Stream::new()?;

    trace(&format!("spec {SPEC}"));
    trace("verify-start");
    let tiles = verify(support, &stream)?;
    trace("verify-end");
    println!(" [1/3] FP8 correctness   OK - {tiles}/{tiles} lottery tiles bit-exact against the reference job");

    // f780 benchmark operands: real noised FP8 codes. Default: tiled from the small embedded reference job
    // (160x256x1024, repeats). BPM_FP8_OPS=<dir> with a.bin/b.bin (row-major e4m3, rows of 8192 codes, e.g. the
    // unique 1600x10496x8192 dump) uses non-repeating operands instead.
    let (m, n, k) = (1600usize, 41 * BN, 16384usize);
    let tile = |src: &[u8], src_k: usize, rows: usize| -> Vec<u8> {
        let sr = src.len() / src_k;
        let mut v = vec![0u8; rows * k];
        for r in 0..rows {
            for c in 0..k {
                v[r * k + c] = src[(r % sr) * src_k + c % src_k];
            }
        }
        v
    };
    let ops_dir = std::env::var_os("BPM_FP8_OPS").map(std::path::PathBuf::from);
    let (src_a, src_b, src_k, data) = match &ops_dir {
        Some(d) => {
            let (a, b) = (std::fs::read(d.join("a.bin"))?, std::fs::read(d.join("b.bin"))?);
            const SK: usize = 8192;
            ensure!(a.len() % SK == 0 && b.len() % SK == 0 && !a.is_empty() && !b.is_empty(), "BPM_FP8_OPS: a.bin/b.bin must hold rows of {SK} codes");
            let desc = format!("{} ({}x{}x{SK}, unique rows)", d.display(), a.len() / SK, b.len() / SK);
            (a, b, SK, desc)
        }
        None => (REF_A.to_vec(), REF_B.to_vec(), REF_K, format!("embedded reference job tiled ({REF_M}x{REF_N}x{REF_K} repeats)")),
    };
    trace(&format!("operands {data}"));
    let (free, _) = cuda::memory_info()?;
    let need = m * k + n * k + (480 << 20);
    if free < need {
        bail!("not enough free GPU memory: {} MiB free, about {} MiB needed (stop other GPU programs, e.g. a running miner)", free >> 20, need >> 20);
    }
    trace("f780-setup-start");
    let fp8 = F780::new(&tile(&src_a, src_k, m), &tile(&src_b, src_k, n), REF_KEY, m, n, k, false)?;
    drop((src_a, src_b));
    // Batch size: about 400 ms of f780 per sync (one 780 round takes ~400 ms too).
    fp8.run(support, &stream)?;
    let one_ms = fp8.run(support, &stream)? as f64;
    let batch = ((400.0 / one_ms).ceil() as usize).max(1);
    println!(" [2/3] Setup             OK - both kernels loaded, FP8 operands: real noised codes");
    trace("f780-setup-end");
    // Kernel 780 in the production profile (default variant u6).
    let profile = Profile::new(Kernel::K780, 65536)?;
    let job = crate::test_job(profile)?;
    trace("k780-setup-start");
    let mut eng = Engine::new(profile, 2, 4096)?;
    eng.set_job(&job, 0)?;
    eng.launch(0)?;
    eng.collect(0)?;
    trace("k780-setup-end");
    let k780 = support.k780_variants[0];
    let macs780 = profile.macs_per_round();

    // Telemetry sampler: tags every sample with the kernel that is running.
    let phase = Arc::new(AtomicU8::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let tele = Arc::new(Mutex::new([Tele::default(), Tele::default(), Tele::default()]));
    let sampler = nvml.map(|nv| {
        let (p, s, t) = (phase.clone(), stop.clone(), tele.clone());
        std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(200));
                let ph = p.load(Ordering::SeqCst) as usize;
                if ph == 0 {
                    continue;
                }
                let mut g = t.lock().unwrap();
                if let Some(w) = nv.power_w() { g[ph].w.push(w); }
                if let Some(c) = nv.sm_clock_mhz() { g[ph].mhz.push(c); }
                if let Ok(x) = nv.temperature_c() { g[ph].temp.push(x as f64); }
            }
        })
    });

    // Blocks A B B A ...: each block runs both kernels, each half block first runs SETTLE seconds unmeasured (the power limiter and clock
    // settle on the new kernel; without it the kernel that follows the more efficient one starts at its
    // higher clock and is overrated), then measures for `half` seconds.
    const SETTLE: f64 = 6.0;
    // `seconds` = total time of all blocks (settle included); even block count so both orders occur equally.
    let blocks = ((seconds / (2.0 * (SETTLE + 15.0))).floor() as usize).max(2) / 2 * 2;
    let half = (seconds / (2.0 * blocks as f64) - SETTLE).max(15.0);
    println!(" [3/3] Measuring         {blocks} rounds, TODAY and AFTER FORK alternate");
    println!("                         keep other GPU load off during the test (Ctrl+C aborts)");
    trace(&format!("f780 batch {batch} x {one_ms:.2} ms, settle {SETTLE} s, measure {half:.1} s"));
    let ticker = Ticker::new();
    let run_block = |which: usize, settle: f64, secs: f64, eng: &Engine| -> Result<(f64, f64)> {
        let one = || -> Result<(f64, f64)> {
            let w0 = Instant::now();
            let r = if which == 1 {
                let kms = eng.time_k780(k780)?.0;
                (macs780, kms as f64, w0.elapsed().as_secs_f64() * 1e3)
            } else {
                let wall = fp8.run_batch(support, &stream, batch)?;
                (fp8.macs() * batch as f64, f64::NAN, wall)
            };
            trace(&format!("launch k={which} kernel_ms={:.1} wall_ms={:.1}", r.1, r.2));
            Ok((r.0, r.2))
        };
        let name = if which == 1 { "TODAY (INT7)" } else { "AFTER FORK (FP8)" };
        ticker.label(&format!("{} {name}", if secs > 0.0 { "running" } else { "warming up" }));
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < settle {
            one()?;
        }
        trace(&format!("measure-start k={which}"));
        phase.store(which as u8, Ordering::SeqCst);
        let t1 = Instant::now();
        let (mut ms, mut macs) = (0.0f64, 0.0f64);
        while t1.elapsed().as_secs_f64() < secs {
            let (a, b) = one()?;
            macs += a;
            ms += b;
        }
        phase.store(0, Ordering::SeqCst);
        Ok((macs, ms))
    };
    ticker.phase("warm-up", 10.0);
    run_block(1, 10.0, 0.0, &eng)?; // warm-up
    let mut rate = [Vec::new(), Vec::new(), Vec::new()];
    for b in 0..blocks {
        // A B B A: kernel 780 first in even blocks, f780 first in odd blocks.
        let order = if b % 2 == 0 { [1, 2] } else { [2, 1] };
        ticker.phase(&format!("round {}/{}", b + 1, blocks), 2.0 * (SETTLE + half));
        for &w in &order {
            let (macs, ms) = run_block(w, SETTLE, half, &eng)?;
            rate[w].push(macs / (ms / 1000.0) / 1e12);
        }
        ticker.clear();
        println!("         round {:>2}/{:<2}      TODAY {:>6.1} TH/s   AFTER FORK {:>6.1} TH/s   ({:+.1} %)",
            b + 1, blocks, rate[1][b], rate[2][b], 100.0 * (rate[2][b] / rate[1][b] - 1.0));
    }
    drop(ticker);
    stop.store(true, Ordering::SeqCst);
    if let Some(h) = sampler {
        let _ = h.join();
    }
    let t = tele.lock().unwrap().clone();
    let (r780, r8) = (mean(&rate[1]), mean(&rate[2]));
    // Paired ratios per block (adjacent in time): drift cancels.
    let ratios: Vec<f64> = rate[2].iter().zip(&rate[1]).map(|(a, b)| a / b).collect();
    let rm = mean(&ratios);
    let se = if ratios.len() > 1 { (ratios.iter().map(|r| (r - rm).powi(2)).sum::<f64>() / (ratios.len() - 1) as f64 / ratios.len() as f64).sqrt() } else { f64::NAN };

    let (w1, w2) = (mean(&t[1].w), mean(&t[2].w));
    let (e1, e2) = (r780 / w1, r8 / w2);
    let fmt = |v: f64, unit: &str, prec: usize| if v.is_finite() { format!("{v:.prec$} {unit}") } else { "n/a".into() };
    let pct = |a: f64, b: f64| if a.is_finite() && b.is_finite() { format!("{:+.1} %", 100.0 * (b / a - 1.0)) } else { String::new() };
    println!();
    println!("{RULE}");
    println!(" RESULT  {}  @ {} power limit", info.name, limit.map_or("unknown".into(), |w| format!("{w:.0} W")));
    println!("{THIN}");
    println!(" {:<18}{:>22}{:>24}{:>16}", "", "TODAY (INT7, v3)", "AFTER FORK (FP8, v4)", "CHANGE");
    println!(" {:<18}{:>22}{:>24}{:>16}", "Hashrate", fmt(r780, "TH/s", 1), fmt(r8, "TH/s", 1), format!("{:+.1} %", 100.0 * (rm - 1.0)));
    println!(" {:<18}{:>22}{:>24}{:>16}", "Efficiency", fmt(e1, "TH/J", 3), fmt(e2, "TH/J", 3), pct(e1, e2));
    println!(" {:<18}{:>22}{:>24}{:>16}", "Power draw", fmt(w1, "W", 1), fmt(w2, "W", 1), "");
    println!(" {:<18}{:>22}{:>24}{:>16}", "SM clock", fmt(mean(&t[1].mhz), "MHz", 0), fmt(mean(&t[2].mhz), "MHz", 0), pct(mean(&t[1].mhz), mean(&t[2].mhz)));
    println!(" {:<18}{:>22}{:>24}{:>16}", "GPU temperature", fmt(mean(&t[1].temp), "°C", 0), fmt(mean(&t[2].temp), "°C", 0), "");
    println!("{THIN}");
    println!(" IMPACT: after the FP8 fork this GPU does {:.0} % of today's hashrate ({:+.1} % ± {:.1} %, {} rounds).",
        100.0 * rm, 100.0 * (rm - 1.0), 100.0 * se, ratios.len());
    println!(" Why: FP8 operands are dense random data that must be streamed from VRAM for every attempt.");
    println!(" Moving them costs power, so at the same power limit the GPU runs at a lower clock.");
    println!(" A higher power limit narrows the gap (tested on RTX PRO 4500: -17 % at 150 W, -13 % at 200 W).");
    println!(" Earnings depend on your share of the network hashrate. Every GPU has to switch to FP8, so");
    println!(" network hashrate and difficulty will move too; a lower TH/s does not mean lower earnings 1:1.");
    println!("{THIN}");
    println!(" Notes:");
    println!("  * PRELIMINARY. PIP-3 is not final and has no mainnet fork height. Protocol, tile shapes,");
    println!("    limits and pool rules can still change; results may change with them.");
    println!("  * The FP8 kernel is experimental: its arithmetic and lottery hash match the reference bit for");
    println!("    bit, but no pool accepts FP8 shares yet and nothing was mined end to end.");
    println!("  * Hashrate = multiply-adds per second (1 TH/s = 10^12 MAC/s) of the mining GEMM, measured");
    println!("    by wall clock. Per-job/per-attempt operand preparation is not included (FP8: est. < 1 %).");
    println!("  * Results depend on power limit, cooling, memory clock, driver and other GPU load.");
    println!("    No guarantee of future hashrate or earnings.");
    println!("{RULE}");
    // The test is part of the normal miner binary: show how to mine with exactly this binary right away.
    let exe = std::env::args().next().unwrap_or_else(|| "bpm".into());
    let dev = if device != 0 { format!(" --device {device}") } else { String::new() };
    println!(" START MINING (today's algorithm) with this same binary, no rebuild needed:");
    println!();
    println!("   {exe} --pool prl-eu.kryptex.network:7048 --wallet YOUR_KRYPTEX_ID --worker rig1{dev}");
    println!();
    println!(" Replace YOUR_KRYPTEX_ID (starts with krx) and rig1. Add --background if the GPU is shared,");
    println!(" --tune once to pick the fastest kernel variant. Any PearlHash pool works: see README.");
    println!("{RULE}");
    Ok(())
}

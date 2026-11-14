//! Kernel tuning. Kernel 780 is compiled in several variants (kernels/variants.txt): placements
//! of the operand stores and loads and ptxas optimisation levels. How fast each one runs depends
//! on the GPU, the CUDA toolkit that built it and the power limit. Started with --tune, every variant
//! is checked against the default (identical hits on a fixed job) and timed on this GPU,
//! interleaved round by round so clock and temperature drift hit all variants alike. The fastest
//! valid variant is used and remembered per GPU, driver, build, K and power limit.
use crate::engine::Engine;
use crate::kernels::{self, K780Fn, VARIANTS};
use crate::protocol::{Kernel, Profile};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Smallest relative gain that replaces the default variant.
pub const MIN_GAIN: f64 = 0.0005;

pub struct Options {
    /// Measure now (--tune), even if a stored result exists.
    pub tune: bool,
    /// Measure if no stored result exists (--tune-only).
    pub if_missing: bool,
    /// Use the default variant without measuring.
    pub disabled: bool,
    /// Use this variant (name) without measuring.
    pub force: Option<String>,
    /// Measuring time budget in seconds.
    pub seconds: f64,
    /// Console progress bar (human-readable mode on a terminal).
    pub progress: bool,
}

/// Identifies the conditions a measurement holds for.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Key {
    pub gpu: String,
    pub driver: String,
    pub library: String,
    pub k: usize,
    pub power_limit_w: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Score {
    pub name: String,
    /// Median kernel time in milliseconds (0 if invalid).
    pub ms: f64,
    /// Time relative to the default variant (< 1 is faster; 0 if invalid).
    pub relative: f64,
    /// Standard error of `relative` (paired rounds).
    #[serde(default)]
    pub error: f64,
    pub valid: bool,
    pub samples: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Entry {
    #[serde(flatten)]
    key: Key,
    variant: String,
    measured_unix: u64,
    results: Vec<Score>,
}

#[derive(Serialize, Deserialize, Default)]
struct Cache {
    version: u32,
    entries: Vec<Entry>,
}

fn cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).filter(|p| p.is_absolute()).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("bpm/tune.json"))
}

fn load_cache() -> Cache {
    cache_path().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice::<Cache>(&b).ok()).filter(|c| c.version == 1).unwrap_or_default()
}

fn store(entry: Entry) -> Result<PathBuf> {
    let Some(path) = cache_path() else { bail!("neither XDG_CACHE_HOME nor HOME is set") };
    let mut cache = load_cache();
    cache.version = 1;
    cache.entries.retain(|e| e.key != entry.key);
    cache.entries.push(entry);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&cache)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// NVIDIA driver version (e.g. "620.70"), else the CUDA driver API version.
pub fn driver_version() -> String {
    if let Ok(t) = std::fs::read_to_string("/proc/driver/nvidia/version") {
        if let Some(v) = t.split_whitespace().find(|w| w.contains('.') && w.chars().all(|c| c.is_ascii_digit() || c == '.')) {
            return v.to_string();
        }
    }
    crate::cuda::driver_version().map(|v| format!("cuda-{}.{}", v / 1000, v % 1000 / 10)).unwrap_or_else(|_| "unknown".into())
}

pub fn key(gpu: &str, k: usize, power_limit_w: Option<f64>) -> Key {
    Key { gpu: gpu.to_string(), driver: driver_version(), library: kernels::library_hash(), k, power_limit_w: power_limit_w.map(|w| w.round()) }
}

fn median(v: &[f32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// Result of an interleaved measurement; index 0 is the reference.
pub struct Measurement {
    pub times: Vec<Vec<f32>>,
    pub valid: Vec<bool>,
    /// More than 3 % slower than the default in the first round: no longer timed.
    pub dropped: Vec<bool>,
    pub rounds: usize,
    pub hits: u64,
}

impl Measurement {
    pub fn scores(&self, names: &[String]) -> Vec<Score> {
        names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                // Invalid variants get 0 (JSON has no NaN).
                let ms = if self.valid[i] { median(&self.times[i]) } else { 0.0 };
                // Paired per round (same operands, adjacent in time): drift cancels.
                let ratios: Vec<f32> = self.times[i].iter().zip(&self.times[0]).map(|(t, t0)| t / t0).collect();
                let relative = if self.valid[i] { median(&ratios) } else { 0.0 };
                let error = if self.valid[i] && ratios.len() > 1 {
                    let mean = ratios.iter().map(|&r| r as f64).sum::<f64>() / ratios.len() as f64;
                    let var = ratios.iter().map(|&r| (r as f64 - mean).powi(2)).sum::<f64>() / (ratios.len() - 1) as f64;
                    // Standard error of the median, about 1.25 times that of the mean.
                    1.25 * (var / ratios.len() as f64).sqrt()
                } else {
                    0.0
                };
                Score { name: n.clone(), ms, relative, error, valid: self.valid[i], samples: self.times[i].len() }
            })
            .collect()
    }
}

/// Times every function on the self-test job, round by round in shuffled order. Each round
/// advances the nonce (new operands); every function must report the hits of `fns[0]`.
/// Stops after `min_rounds` once `seconds` have passed, or after `max_rounds`.
pub fn measure(
    profile: Profile,
    fns: &[K780Fn],
    seconds: f64,
    min_rounds: usize,
    max_rounds: usize,
    stop: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(f64, &Measurement),
) -> Result<Measurement> {
    let job = crate::test_job(profile)?;
    let mut e = Engine::new(profile, 2, 4096)?;
    e.set_job(&job, 0)?;
    // Warm-up: clocks and caches settle, every variant loads its module. Variants more than twice
    // as slow as the default are timed once and not again (see below).
    let mut warm = Vec::with_capacity(fns.len());
    for f in fns {
        warm.push(e.time_k780(*f)?);
    }
    e.time_k780(fns[0])?;
    let mut m = Measurement { times: vec![Vec::new(); fns.len()], valid: vec![true; fns.len()], dropped: vec![false; fns.len()], rounds: 0, hits: 0 };
    let mut rng: u64 = 0x9e3779b97f4a7c15;
    for i in 1..fns.len() {
        if warm[i].0 > 2.0 * warm[0].0 {
            // Checked and timed on the warm-up operands only.
            m.dropped[i] = true;
            m.valid[i] = warm[i].1 == warm[0].1 && warm[i].2 == warm[0].2;
            m.times[i].push(warm[i].0);
        }
    }
    let start = Instant::now();
    while m.rounds < max_rounds && !(m.rounds >= min_rounds && start.elapsed().as_secs_f64() >= seconds) {
        if stop() {
            bail!("interrupted");
        }
        e.launch(0)?;
        e.collect(0)?;
        // A kernel runs at the clock its predecessor left behind (power and temperature): a fixed
        // order would favour whatever follows a slow variant, so the order is shuffled every round.
        let mut order: Vec<usize> = (0..fns.len()).collect();
        for j in (1..order.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            order.swap(j, (rng % (j as u64 + 1)) as usize);
        }
        let mut out: Vec<Option<(f32, u32, Vec<u32>)>> = vec![None; fns.len()];
        for &i in &order {
            if i == 0 || (m.valid[i] && !m.dropped[i]) {
                out[i] = Some(e.time_k780(fns[i])?);
            }
        }
        let (_, ref_count, ref_ids) = out[0].clone().unwrap();
        m.hits += ref_count as u64;
        for (i, o) in out.into_iter().enumerate() {
            if let Some((ms, count, ids)) = o {
                if count != ref_count || ids != ref_ids {
                    m.valid[i] = false;
                } else {
                    m.times[i].push(ms);
                }
            }
        }
        m.rounds += 1;
        // Variants clearly slower than the default are not timed any further: while a slow kernel
        // runs, the power limiter recovers, and the next kernel runs at a higher clock for a while.
        if m.rounds == 1 {
            let base = median(&m.times[0]);
            for i in 1..fns.len() {
                if median(&m.times[i]) > base * 1.03 {
                    m.dropped[i] = true;
                }
            }
        }
        let frac = if max_rounds <= min_rounds { m.rounds as f64 / max_rounds as f64 } else { (start.elapsed().as_secs_f64() / seconds).max(m.rounds as f64 / max_rounds as f64).min(1.0) };
        progress(frac, &m);
    }
    if m.hits == 0 {
        bail!("the tuning job produced no hits, variants cannot be compared");
    }
    Ok(m)
}

fn bar(frac: f64, text: &str) {
    let width = 30;
    let filled = ((frac.clamp(0.0, 1.0) * width as f64).round()) as usize;
    let mut err = std::io::stderr();
    let _ = write!(err, "\r\x1b[2KKernel tuning [{}{}] {:3.0} % {text}", "#".repeat(filled), ".".repeat(width - filled), frac * 100.0);
    let _ = err.flush();
}

fn names() -> Vec<String> {
    VARIANTS.iter().map(|(n, _)| n.to_string()).collect()
}

/// Selects the kernel 780 variant for this run (logs through `log`). `busy`: another process
/// computes on the GPU, so a measurement now would be distorted.
pub fn choose(gpu: &str, profile: Profile, power_limit_w: Option<f64>, opts: &Options, busy: bool, stop: &dyn Fn() -> bool, log: &dyn Fn(&str, Value)) -> Result<()> {
    kernels::select(0);
    if profile.kernel != Kernel::K780 {
        return Ok(());
    }
    if let Some(name) = &opts.force {
        let Some(i) = VARIANTS.iter().position(|(n, _)| n == name) else {
            bail!("unknown --kernel-variant {name} (available: {})", names().join(", "));
        };
        kernels::select(i);
        log("tune_fixed", json!({"kernel_variant": name, "reason": "--kernel-variant"}));
        return Ok(());
    }
    if opts.disabled || VARIANTS.len() == 1 {
        log("tune_fixed", json!({"kernel_variant": VARIANTS[0].0, "reason": if opts.disabled { "--no-tune" } else { "single variant" }}));
        return Ok(());
    }
    let key = key(gpu, profile.k, power_limit_w);
    let cached = load_cache().entries.into_iter().find(|e| e.key == key);
    if let (Some(c), false) = (&cached, opts.tune) {
        if let Some(i) = VARIANTS.iter().position(|(n, _)| *n == c.variant) {
            kernels::select(i);
            let gain = c.results.iter().find(|s| s.name == c.variant).map(|s| (1.0 / s.relative - 1.0) * 100.0).unwrap_or(0.0);
            log("tune_cached", json!({"kernel_variant": c.variant, "gain_percent": gain, "measured_unix": c.measured_unix, "k": profile.k}));
            return Ok(());
        }
    }
    if !opts.tune && !opts.if_missing {
        log("tune_fixed", json!({"kernel_variant": VARIANTS[0].0, "reason": "default; --tune measures all variants on this GPU"}));
        return Ok(());
    }
    if busy {
        log("tune_skipped", json!({"kernel_variant": VARIANTS[0].0, "reason": "another process computes on the GPU; start again with --tune when the GPU is free"}));
        return Ok(());
    }
    log("tune_start", json!({"variants": VARIANTS.len(), "seconds": opts.seconds, "k": profile.k}));
    let support = kernels::get()?;
    let n = names();
    let mut draw = |frac: f64, m: &Measurement| {
        if opts.progress {
            let scores = m.scores(&n);
            let best = scores.iter().filter(|s| s.valid).min_by(|a, b| a.relative.partial_cmp(&b.relative).unwrap());
            let text = best.map(|b| format!("| best so far {} ({:+.2} %)", b.name, (1.0 / b.relative - 1.0) * 100.0)).unwrap_or_default();
            bar(frac, &text);
        }
    };
    let m = measure(profile, &support.k780_variants, opts.seconds, 5, 10_000, stop, &mut draw);
    if opts.progress {
        eprintln!();
    }
    let m = m?;
    let scores = m.scores(&n);
    let fastest = (0..scores.len()).filter(|&i| scores[i].valid).min_by(|&a, &b| scores[a].relative.partial_cmp(&scores[b].relative).unwrap()).unwrap_or(0);
    // The fastest variant replaces the default only if it is faster beyond twice its measuring error
    // and by at least 0.05 % (identical machine code from another library differs by up to 0.02 %).
    let significant = 1.0 - scores[fastest].relative > (2.0 * scores[fastest].error).max(MIN_GAIN);
    let best = if significant { fastest } else { 0 };
    kernels::select(best);
    let invalid: Vec<&str> = scores.iter().filter(|s| !s.valid).map(|s| s.name.as_str()).collect();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let saved = store(Entry { key, variant: scores[best].name.clone(), measured_unix: now, results: scores.clone() });
    log(
        "tune_result",
        json!({"kernel_variant": scores[best].name, "gain_percent": (1.0 / scores[best].relative - 1.0) * 100.0, "rounds": m.rounds,
               "fastest": scores[fastest].name, "fastest_gain_percent": (1.0 / scores[fastest].relative - 1.0) * 100.0, "significant": significant,
               "invalid": invalid, "results": scores, "cache": saved.as_ref().map(|p| p.display().to_string()).map_err(|e| format!("{e:#}"))}),
    );
    Ok(())
}

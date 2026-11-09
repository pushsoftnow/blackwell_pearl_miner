//! Human-readable console events; the JSONL log stays independent and complete.
use serde_json::Value;

fn clean(v: &str, limit: usize) -> String {
    v.chars().map(|c| if c.is_control() { ' ' } else { c }).take(limit).collect()
}

fn s(row: &Value, key: &str) -> String {
    match &row[key] {
        Value::String(x) => clean(x, 180),
        Value::Null => String::new(),
        other => clean(&other.to_string(), 180),
    }
}

pub fn number(v: &Value, digits: usize) -> String {
    match v.as_f64() {
        Some(x) if x.is_finite() => format!("{:.digits$}", if x == 0.0 { 0.0 } else { x }),
        _ => "-".into(),
    }
}

pub fn duration(v: &Value) -> String {
    let total = v.as_f64().unwrap_or(0.0).max(0.0) as u64;
    let (days, rest) = (total / 86400, total % 86400);
    let (h, m, sec) = (rest / 3600, rest % 3600 / 60, rest % 60);
    format!("{}{h:02}:{m:02}:{sec:02}", if days > 0 { format!("{days}d ") } else { String::new() })
}

fn stamp(row: &Value) -> String {
    let t = row["time"].as_f64().unwrap_or(0.0) as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    format!("[{:02}:{:02}:{:02}] ", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

pub fn format_event(row: &Value) -> String {
    let p = stamp(row);
    let event = row["event"].as_str().unwrap_or("info");
    match event {
        "startup" => format!(
            "\nbpm {} | PearlHash\nKernel {} | K={} | run time {}\nPool:   {}\nWallet: {}\nWorker: {}\nLog:    {}\nStop with Ctrl+C.\n",
            env!("CARGO_PKG_VERSION"), s(row, "variant"), s(row, "k"),
            if row["seconds"].as_f64().unwrap_or(0.0) > 0.0 { duration(&row["seconds"]) } else { "unlimited".into() },
            s(row, "endpoint"), s(row, "wallet"), s(row, "worker"), clean(row["log_path"].as_str().unwrap_or(""), 1024)
        ),
        "authorized" => format!(
            "{p}Pool connected | worker {} | {}",
            s(row, "worker"),
            if row["proof_compression"] == "gzip" { "gzip proofs" } else { "uncompressed proofs" }
        ),
        "job" => format!("{p}New job: {}", clean(row["job_id"].as_str().unwrap_or(""), 36)),
        "mining" => format!("{p}Mining | kernel {} | K={} | sampling {} | VRAM {} MiB", s(row, "variant"), s(row, "k"), s(row, "sampling"), number(&row["gpu_memory_mib"], 0)),
        "share" => {
            let label = if row["accepted"] == true { "SHARE ACCEPTED" } else { "SHARE REJECTED" };
            let mut out = format!(
                "{p}{label} | {} accepted / {} rejected | {} ms",
                s(row, "accepted_total"), s(row, "rejected_total"), number(&row["roundtrip_ms"], 0)
            );
            if row["accepted"] != true {
                if let Some(e) = row["pool_response"].get("error").filter(|e| !e.is_null()) {
                    out += &format!(" | {}", clean(&e.to_string(), 180));
                }
            }
            out
        }
        "stats" | "complete" => {
            let mut title = if event == "complete" { "Finished".to_string() } else { "Status".to_string() };
            if row["interrupted"] == true {
                title += " (stopped)";
            }
            let mut out = format!(
                "{p}{title} | run time {} | compute {} TMAC/s\n           submitted {} TH/s | window {} (max 30 min) | {} shares in window\n           shares: {} accepted / {} rejected / {} stale",
                duration(&row["seconds"]), number(&row["TMAC_wall_s"], 1), number(&row["submitted_TH_s_30m"], 1),
                duration(&row["submitted_window_s"]), row["submitted_shares_30m"].as_u64().unwrap_or(0),
                row["accepted"].as_u64().unwrap_or(0), row["rejected"].as_u64().unwrap_or(0), row["stale"].as_u64().unwrap_or(0)
            );
            if row["pauses"].as_u64().unwrap_or(0) > 0 {
                out += &format!("\n           paused for priority programs: {}x, {}", row["pauses"], duration(&row["paused_seconds"]));
            }
            out
        }
        "share_discarded" | "round_discarded" => format!("{p}Stale work discarded (job changed)."),
        "priority_pause" => {
            let who = row["path"].as_str().map(|path| format!(" | {} (PID {})", clean(path.rsplit('/').next().unwrap_or(path), 80), row["pid"])).unwrap_or_default();
            format!("{p}PAUSED: GPU yielded to a priority program{who}")
        }
        "priority_resume" => format!("{p}RESUMED after {}", duration(&row["paused_seconds"])),
        "disconnected" => format!("{p}Pool connection lost: {} | retrying in {} s", s(row, "error"), s(row, "retry_seconds")),
        "profile_fallback" => format!("{p}Pool without gzip proofs: using kernel {} with K={}", s(row, "variant"), s(row, "k")),
        "error" => format!("{p}ERROR: {}", s(row, "error")),
        _ => {
            let mut m = row.as_object().cloned().unwrap_or_default();
            m.remove("time");
            m.remove("event");
            format!("{p}{}: {}", clean(event, 60), clean(&Value::Object(m).to_string(), 240))
        }
    }
}

//! CPU proof worker and network worker. The socket and its request IDs have exactly
//! one owner (the network thread); no CUDA call happens outside the GPU thread.
use crate::pool::{Pool, SubmitStats};
use crate::proof::{verify, ProofCache};
use crate::protocol::{hex, share_work, MiningJob};
use crate::state::{merge, Generation, Log, State};
use anyhow::Result;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use crossbeam_channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Immutable per-job proof context (job data + Merkle heap); shared by in-flight tasks.
pub struct ProofContext {
    pub job: MiningJob,
    pub cache: ProofCache,
}

pub struct ProofTask {
    pub generation: Generation,
    pub ctx: Arc<ProofContext>,
    pub group: u32,
    pub signal: [u8; 64],
    pub root_a: [u8; 32],
}

pub struct VerifiedShare {
    generation: Generation,
    group: u32,
    bits: u32,
    proof: Vec<u8>,
    b64_len: usize,
    sha256: String,
    proof_ms: f64,
    verification_ms: f64,
    serialization_ms: f64,
}

/// Blocking send that keeps checking for worker failure (`put_checked`).
pub fn put<T>(tx: &Sender<T>, mut item: T, state: &State) -> Result<()> {
    loop {
        state.check()?;
        match tx.send_timeout(item, Duration::from_millis(50)) {
            Ok(()) => return Ok(()),
            Err(SendTimeoutError::Timeout(back)) => item = back,
            Err(SendTimeoutError::Disconnected(_)) => anyhow::bail!("worker channel closed"),
        }
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

pub fn proof_worker(state: Arc<State>, rx: Receiver<Option<ProofTask>>, tx: Sender<Option<VerifiedShare>>, log: Log) {
    let run = || -> Result<()> {
        while !state.abort.load(Ordering::SeqCst) {
            let task = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(t) => t,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            };
            let Some(task) = task else {
                return put(&tx, None, &state);
            };
            let discard = |reason: &str| {
                state.add_stale(1);
                log.event("share_discarded", json!({"job_id":task.generation.job.job_id,"group":task.group,"reason":reason}));
            };
            if !state.submittable(&task.generation) {
                discard("job_changed_before_proof");
                continue;
            }
            let t = Instant::now();
            let proof = task.ctx.cache.proof(task.group, &task.signal, &task.root_a)?;
            let proof_ms = ms(t);
            let t = Instant::now();
            let bytes = verify(&task.ctx.job, &proof)?;
            let verification_ms = ms(t);
            let t = Instant::now();
            let b64 = STANDARD.encode(&bytes);
            let sha256 = hex(&Sha256::digest(b64.as_bytes()));
            let serialization_ms = ms(t);
            if !state.submittable(&task.generation) {
                discard("job_changed_after_verification");
                continue;
            }
            let share = VerifiedShare {
                generation: task.generation.clone(),
                group: task.group,
                bits: task.ctx.job.bits,
                proof: bytes,
                b64_len: b64.len(),
                sha256,
                proof_ms,
                verification_ms,
                serialization_ms,
            };
            put(&tx, Some(share), &state)?;
        }
        Ok(())
    };
    if let Err(e) = run() {
        state.fail(format!("proof worker: {e:#}"));
    }
}

pub struct NetworkConfig {
    pub endpoint: String,
    pub wallet: String,
    pub worker: String,
    pub stop_after: u64,
    pub max_inflight: usize,
    pub ack_timeout: Duration,
    pub retry: Duration,
}

pub struct NetworkStats {
    pub max_pending_seen: usize,
}

fn ack(state: &State, log: &Log, share: &VerifiedShare, response: &Value, roundtrip_ms: f64, wire: &SubmitStats) {
    let good = response["result"] == json!(true) && response["error"].is_null();
    let (accepted, rejected) = {
        let mut c = state.counters.lock().unwrap();
        if good { c.accepted += 1 } else { c.rejected += 1 }
        (c.accepted, c.rejected)
    };
    let job = &share.generation.job;
    let rate = state.share_rate.lock().unwrap().snapshot();
    let row = merge(&[
        rate,
        json!({"proof_compression":wire.proof_compression,"wire_proof_bytes":wire.wire_proof_bytes,
               "wire_frame_bytes":wire.wire_frame_bytes,"compression_ms":wire.compression_ms,
               "wire_reduction_percent":wire.wire_reduction_percent,
               "accepted":good,"accepted_total":accepted,"rejected_total":rejected,"job_id":job.job_id,
               "group":share.group,"proof_bytes":share.b64_len,"proof_sha256":share.sha256,
               "local_verification":"official verifier accepted","roundtrip_ms":roundtrip_ms,"pool_response":response,
               "proof_ms":share.proof_ms,"verification_ms":share.verification_ms,"serialization_ms":share.serialization_ms}),
    ]);
    let artifact = good.then(|| {
        json!({"header":hex(&job.header),"cert_version":job.cert_version,"job_id":job.job_id,"share_nbits":share.bits,
               "plain_proof":STANDARD.encode(&share.proof),"pool_response":response})
    });
    log.emit("share", row, artifact);
}

/// Owns the pool connection; reconnects after failures. Returns when the proof stream
/// has finished and every submit was acknowledged, or on abort.
pub fn network_worker(state: Arc<State>, rx: Receiver<Option<VerifiedShare>>, cfg: NetworkConfig, log: Log) -> NetworkStats {
    let mut stats = NetworkStats { max_pending_seen: 0 };
    let mut finished = false;
    while !state.abort.load(Ordering::SeqCst) {
        let mut pending: BTreeMap<u64, (VerifiedShare, Instant, SubmitStats)> = BTreeMap::new();
        let session = (|| -> Result<bool> {
            let mut pool = Pool::connect(&cfg.endpoint, &cfg.wallet, &cfg.worker, &state.abort)?;
            *state.compression.lock().unwrap() = Some(pool.compression().into());
            log.event("authorized", json!({"endpoint":cfg.endpoint,"worker":cfg.worker,"proof_compression":pool.compression()}));
            for j in pool.take_announcements() {
                log.event("job", json!({"job_id":j.job_id,"cert_version":j.cert_version}));
            }
            state.update_job(pool.latest.as_ref().unwrap());
            state.ready.store(true, Ordering::SeqCst);
            loop {
                if state.abort.load(Ordering::SeqCst) {
                    return Ok(true);
                }
                pool.read(Duration::from_millis(5))?;
                for j in pool.take_announcements() {
                    log.event("job", json!({"job_id":j.job_id,"cert_version":j.cert_version}));
                    state.update_job(&j);
                }
                let now = Instant::now();
                let ids: Vec<u64> = pending.keys().copied().collect();
                for id in ids {
                    if let Some(resp) = pool.responses.remove(&id) {
                        let (share, sent, wire) = pending.remove(&id).unwrap();
                        ack(&state, &log, &share, &resp, now.duration_since(sent).as_secs_f64() * 1000.0, &wire);
                    } else if now.duration_since(pending[&id].1) > cfg.ack_timeout {
                        anyhow::bail!("pool did not acknowledge submit id {id}");
                    }
                }
                let accepted = state.counters().accepted;
                if cfg.stop_after > 0 && accepted >= cfg.stop_after {
                    state.stop_mining.store(true, Ordering::SeqCst);
                }
                if finished && pending.is_empty() {
                    return Ok(true);
                }
                while !finished && pending.len() < cfg.max_inflight {
                    let accepted = state.counters().accepted;
                    // Reserve remaining stop-after slots so in-flight requests cannot overshoot.
                    if cfg.stop_after > 0 && accepted < cfg.stop_after && accepted + pending.len() as u64 >= cfg.stop_after {
                        break;
                    }
                    let share = match rx.try_recv() {
                        Ok(Some(s)) => s,
                        Ok(None) | Err(TryRecvError::Disconnected) => {
                            finished = true;
                            break;
                        }
                        Err(TryRecvError::Empty) => break,
                    };
                    let stale = !state.submittable(&share.generation);
                    if stale || (cfg.stop_after > 0 && accepted >= cfg.stop_after) {
                        if stale {
                            state.add_stale(1);
                        }
                        log.event("share_discarded", json!({"job_id":share.generation.job.job_id,"group":share.group,
                            "reason": if stale { "job_changed_before_submit" } else { "stop_after_shares" }}));
                        continue;
                    }
                    let (id, wire) = pool.submit(&share.generation.job.job_id, &share.proof, share.b64_len)?;
                    let sent = Instant::now();
                    state.share_rate.lock().unwrap().record(share_work(share.bits), sent);
                    pending.insert(id, (share, sent, wire));
                    stats.max_pending_seen = stats.max_pending_seen.max(pending.len());
                }
                if finished && pending.is_empty() {
                    return Ok(true);
                }
            }
        })();
        match session {
            Ok(_) => return stats,
            Err(e) => {
                if !pending.is_empty() {
                    log.event("submit_unknown", json!({"count":pending.len(),"message":"connection ended before acknowledgement; shares are not resubmitted"}));
                }
                log.event("disconnected", json!({"endpoint":cfg.endpoint,"error":format!("{e:#}"),"retry_seconds":cfg.retry.as_secs()}));
                // While disconnected, shares cannot be submitted: drop them, and stop once the
                // miner has finished (end-of-stream sentinel) instead of reconnecting forever.
                let until = Instant::now() + cfg.retry;
                while !finished && Instant::now() < until && !state.abort.load(Ordering::SeqCst) {
                    match rx.try_recv() {
                        Ok(Some(share)) => log.event("share_discarded", json!({"job_id":share.generation.job.job_id,"group":share.group,"reason":"pool_disconnected"})),
                        Ok(None) | Err(TryRecvError::Disconnected) => finished = true,
                        Err(TryRecvError::Empty) => std::thread::sleep(Duration::from_millis(50)),
                    }
                }
                if finished {
                    return stats;
                }
            }
        }
    }
    stats
}

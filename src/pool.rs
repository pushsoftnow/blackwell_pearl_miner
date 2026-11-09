//! Pearl Stratum client (named-parameter dialect).
//! Protocol: https://gist.github.com/MattF42/7fe0ebf245e472c56e8822e7f9262a7b
//! Proof gzip-v2: https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe
use crate::protocol::PoolJob;
use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use flate2::{write::GzEncoder, Compression};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const MAX_LINE: usize = 256 * 1024;

#[derive(Clone, Debug, Default)]
pub struct SubmitStats {
    pub proof_compression: String,
    pub wire_proof_bytes: usize,
    pub wire_frame_bytes: usize,
    pub compression_ms: f64,
    pub wire_reduction_percent: f64,
}

pub struct Pool {
    socket: TcpStream,
    buffer: Vec<u8>,
    pub latest: Option<PoolJob>,
    pub responses: HashMap<u64, Value>,
    next_id: u64,
    pub gzip: bool,
    /// Jobs announced since the last call to `take_announcements` (for logging).
    announced: Vec<PoolJob>,
}

impl Pool {
    pub fn connect(endpoint: &str, wallet: &str, worker: &str, cancel: &AtomicBool) -> Result<Self> {
        let addr = endpoint.to_socket_addrs().with_context(|| format!("resolve {endpoint}"))?.next().context("no pool address")?;
        let socket = TcpStream::connect_timeout(&addr, Duration::from_secs(10)).with_context(|| format!("connect {endpoint}"))?;
        socket.set_nodelay(true)?;
        socket.set_write_timeout(Some(Duration::from_secs(10)))?;
        let mut pool = Pool { socket, buffer: Vec::new(), latest: None, responses: HashMap::new(), next_id: 1, gzip: false, announced: Vec::new() };
        let req = pool.send(
            "mining.authorize",
            json!({"wallet":wallet,"worker":worker,"pass":"x","password":"x","agent":concat!("bpm/", env!("CARGO_PKG_VERSION")),"type":"v2"}),
        )?;
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut authorized = false;
        while Instant::now() < deadline {
            ensure!(!cancel.load(Ordering::Relaxed), "pool authorization cancelled");
            pool.read(Duration::from_secs(1))?;
            if let Some(r) = pool.responses.remove(&req) {
                if !r["error"].is_null() || r["result"] != json!(true) {
                    bail!("pool authorization rejected: {r}");
                }
                pool.gzip = r["type"] == "v2";
                authorized = true;
            }
            if authorized && pool.latest.is_some() {
                return Ok(pool);
            }
        }
        bail!("pool did not authorize and supply a documented job")
    }

    pub fn compression(&self) -> &'static str {
        if self.gzip { "gzip" } else { "none" }
    }

    fn frame(id: u64, method: &str, params: &Value) -> Result<Vec<u8>> {
        let mut f = serde_json::to_vec(&json!({"id":id,"method":method,"params":params}))?;
        f.push(b'\n');
        Ok(f)
    }

    pub fn send(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let f = Self::frame(id, method, &params)?;
        ensure!(f.len() <= MAX_LINE, "request exceeds documented {MAX_LINE}-byte frame limit");
        self.socket.write_all(&f)?;
        Ok(id)
    }

    /// Submit a share. `proof_b64` is the base64 bincode PlainProof; with gzip-v2 the
    /// binary proof is gzip-compressed (fresh RFC1952 member) before base64.
    pub fn submit(&mut self, job_id: &str, proof: &[u8], proof_b64_len: usize) -> Result<(u64, SubmitStats)> {
        let started = Instant::now();
        let wire = if self.gzip {
            let mut g = GzEncoder::new(Vec::new(), Compression::fast());
            g.write_all(proof)?;
            STANDARD.encode(g.finish()?)
        } else {
            STANDARD.encode(proof)
        };
        let compression_ms = started.elapsed().as_secs_f64() * 1000.0;
        let id = self.next_id;
        self.next_id += 1;
        let f = Self::frame(id, "mining.submit", &json!({"job_id":job_id,"plain_proof":wire}))?;
        ensure!(f.len() <= MAX_LINE, "submit exceeds documented {MAX_LINE}-byte frame limit");
        self.socket.write_all(&f)?;
        let stats = SubmitStats {
            proof_compression: self.compression().into(),
            wire_proof_bytes: wire.len(),
            wire_frame_bytes: f.len(),
            compression_ms,
            wire_reduction_percent: if proof_b64_len > 0 { (1.0 - wire.len() as f64 / proof_b64_len as f64) * 100.0 } else { 0.0 },
        };
        Ok((id, stats))
    }

    /// Read for up to `timeout`, then process every complete line.
    pub fn read(&mut self, timeout: Duration) -> Result<()> {
        self.socket.set_read_timeout(Some(timeout.max(Duration::from_millis(1))))?;
        let mut chunk = [0u8; 65536];
        match self.socket.read(&mut chunk) {
            Ok(0) => bail!("pool closed connection"),
            Ok(n) => self.buffer.extend_from_slice(&chunk[..n]),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.into()),
        }
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=pos).collect();
            ensure!(line.len() <= MAX_LINE, "pool line exceeds frame limit");
            let line = &line[..line.len() - 1];
            if line.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }
            let msg: Value = serde_json::from_slice(line).context("pool sent invalid JSON")?;
            if msg["method"] == "mining.notify" {
                let job = PoolJob::parse(&msg["params"])?;
                self.announced.push(job.clone());
                self.latest = Some(job);
            } else if let Some(id) = msg["id"].as_u64() {
                self.responses.insert(id, msg);
            }
        }
        ensure!(self.buffer.len() <= MAX_LINE, "pool line exceeds frame limit");
        Ok(())
    }

    pub fn take_announcements(&mut self) -> Vec<PoolJob> {
        std::mem::take(&mut self.announced)
    }
}

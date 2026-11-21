//! Pearl Stratum client (named-parameter dialect).
//! Protocol: https://gist.github.com/MattF42/7fe0ebf245e472c56e8822e7f9262a7b
//! Proof gzip-v2: https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe
use crate::protocol::PoolJob;
use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use flate2::{write::GzEncoder, Compression};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub const MAX_LINE: usize = 256 * 1024;
/// Responses kept for ids nobody asked about (a well-behaved pool only answers our own requests).
const MAX_RESPONSES: usize = 256;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// A pool address: `host:port`, `stratum+tcp://host:port` or `stratum+ssl://host:port` (also `stratum+tls`, `ssl`, `tls`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub tls: bool,
}

impl Endpoint {
    pub fn parse(s: &str) -> Result<Self> {
        let (tls, rest) = match s.split_once("://") {
            None => (false, s),
            Some((scheme, rest)) => match scheme.to_ascii_lowercase().as_str() {
                "stratum+tcp" | "stratum" | "tcp" => (false, rest),
                "stratum+ssl" | "stratum+tls" | "ssl" | "tls" => (true, rest),
                other => bail!("unsupported pool scheme '{other}://' in '{s}' (use stratum+tcp:// or stratum+ssl://)"),
            },
        };
        let rest = rest.trim_end_matches('/');
        let (host, port) = rest.rsplit_once(':').with_context(|| format!("pool '{s}' needs HOST:PORT"))?;
        let host = host.trim_start_matches('[').trim_end_matches(']');
        ensure!(!host.is_empty(), "pool '{s}' has no host");
        let port = port.parse::<u16>().with_context(|| format!("pool '{s}': invalid port"))?;
        Ok(Endpoint { host: host.to_string(), port, tls })
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let host = if self.host.contains(':') { format!("[{}]", self.host) } else { self.host.clone() };
        write!(f, "{}{host}:{}", if self.tls { "stratum+ssl://" } else { "" }, self.port)
    }
}

enum Conn {
    Plain(TcpStream),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
}

impl Conn {
    fn tcp(&self) -> &TcpStream {
        match self {
            Conn::Plain(s) => s,
            Conn::Tls(s) => &s.sock,
        }
    }
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Plain(s) => s.read(buf),
            Conn::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Plain(s) => s.write(buf),
            Conn::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Plain(s) => s.flush(),
            Conn::Tls(s) => s.flush(),
        }
    }
}

fn tls_config() -> Arc<rustls::ClientConfig> {
    static CFG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    CFG.get_or_init(|| {
        let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        Arc::new(
            rustls::ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .expect("TLS protocol versions")
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    })
    .clone()
}

/// Try every resolved address in turn (round-robin DNS, dead IPv6 or edge nodes must not block the others).
fn connect_tcp(ep: &Endpoint) -> Result<TcpStream> {
    let addrs: Vec<SocketAddr> = (ep.host.as_str(), ep.port).to_socket_addrs().with_context(|| format!("resolve {}", ep.host))?.collect();
    ensure!(!addrs.is_empty(), "{} resolved to no address", ep.host);
    let mut errors = Vec::new();
    for addr in &addrs {
        match TcpStream::connect_timeout(addr, CONNECT_TIMEOUT) {
            Ok(s) => return Ok(s),
            Err(e) => errors.push(format!("{addr}: {e}")),
        }
    }
    bail!("connect {ep}: all {} addresses failed ({})", addrs.len(), errors.join("; "))
}

/// TCP keepalive: a silently dropped connection (NAT timeout, dead pool without FIN) errors out within ~1 min.
fn keepalive(s: &TcpStream) -> Result<()> {
    let ka = socket2::TcpKeepalive::new().with_time(Duration::from_secs(30)).with_interval(Duration::from_secs(10));
    // Windows has no per-socket retry count (system default 10 probes -> ~2 min).
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let ka = ka.with_retries(3);
    socket2::SockRef::from(s).set_tcp_keepalive(&ka)?;
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct SubmitStats {
    pub proof_compression: String,
    pub wire_proof_bytes: usize,
    pub wire_frame_bytes: usize,
    pub compression_ms: f64,
    pub wire_reduction_percent: f64,
}

pub struct Pool {
    conn: Conn,
    buffer: Vec<u8>,
    pub latest: Option<PoolJob>,
    pub responses: BTreeMap<u64, Value>,
    next_id: u64,
    pub gzip: bool,
    /// Jobs announced since the last call to `take_announcements` (for logging).
    announced: Vec<PoolJob>,
    /// Server pushes other than mining.notify since the last `take_notices` (method, params).
    notices: Vec<(String, Value)>,
    /// Time of the last byte received (dead-connection detection one layer up).
    pub last_rx: Instant,
    /// Set by `client.reconnect`: reconnect after this many seconds.
    pub reconnect_requested: Option<u64>,
}

impl Pool {
    pub fn connect(ep: &Endpoint, wallet: &str, worker: &str, cancel: &AtomicBool) -> Result<Self> {
        let tcp = connect_tcp(ep)?;
        tcp.set_nodelay(true)?;
        tcp.set_write_timeout(Some(Duration::from_secs(10)))?;
        keepalive(&tcp)?;
        let conn = if ep.tls {
            let name = rustls::pki_types::ServerName::try_from(ep.host.clone()).with_context(|| format!("TLS server name {}", ep.host))?;
            let tls = rustls::ClientConnection::new(tls_config(), name).context("TLS setup")?;
            tcp.set_read_timeout(Some(CONNECT_TIMEOUT))?;
            let mut s = rustls::StreamOwned::new(tls, tcp);
            while s.conn.is_handshaking() {
                s.conn.complete_io(&mut s.sock).with_context(|| format!("TLS handshake with {ep}"))?;
            }
            Conn::Tls(Box::new(s))
        } else {
            Conn::Plain(tcp)
        };
        let mut pool = Pool {
            conn,
            buffer: Vec::new(),
            latest: None,
            responses: BTreeMap::new(),
            next_id: 1,
            gzip: false,
            announced: Vec::new(),
            notices: Vec::new(),
            last_rx: Instant::now(),
            reconnect_requested: None,
        };
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

    fn write_frame(&mut self, f: &[u8]) -> Result<()> {
        self.conn.write_all(f)?;
        self.conn.flush()?;
        Ok(())
    }

    pub fn send(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let f = Self::frame(id, method, &params)?;
        ensure!(f.len() <= MAX_LINE, "request exceeds documented {MAX_LINE}-byte frame limit");
        self.write_frame(&f)?;
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
        self.write_frame(&f)?;
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
        self.conn.tcp().set_read_timeout(Some(timeout.max(Duration::from_millis(1))))?;
        let mut chunk = [0u8; 65536];
        match self.conn.read(&mut chunk) {
            Ok(0) => bail!("pool closed connection"),
            Ok(n) => {
                self.buffer.extend_from_slice(&chunk[..n]);
                self.last_rx = Instant::now();
            }
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(e.into()),
        }
        let mut start = 0;
        while let Some(len) = self.buffer[start..].iter().position(|&b| b == b'\n') {
            ensure!(len < MAX_LINE, "pool line exceeds frame limit");
            let msg = {
                let line = &self.buffer[start..start + len];
                start += len + 1;
                if line.iter().all(|b| b.is_ascii_whitespace()) {
                    continue;
                }
                serde_json::from_slice::<Value>(line).context("pool sent invalid JSON")?
            };
            self.handle(msg)?;
        }
        self.buffer.drain(..start);
        ensure!(self.buffer.len() <= MAX_LINE, "pool line exceeds frame limit");
        Ok(())
    }

    fn handle(&mut self, msg: Value) -> Result<()> {
        match msg["method"].as_str() {
            Some("mining.notify") => {
                let job = PoolJob::parse(&msg["params"])?;
                self.announced.push(job.clone());
                self.latest = Some(job);
            }
            Some("client.reconnect") => {
                // Only "reconnect to the same pool after N seconds" is honoured; redirects to other hosts are not
                // followed (a pool push must not be able to move the miner's hashrate elsewhere).
                let p = &msg["params"];
                let wait = p["wait"].as_u64().or_else(|| p.get(2).and_then(|w| w.as_u64())).unwrap_or(0).min(600);
                self.reconnect_requested = Some(wait);
                self.notices.push(("client.reconnect".into(), p.clone()));
            }
            Some(method) => self.notices.push((method.to_string(), msg["params"].clone())),
            None => {
                if let Some(id) = msg["id"].as_u64() {
                    self.responses.insert(id, msg);
                    while self.responses.len() > MAX_RESPONSES {
                        self.responses.pop_first();
                    }
                }
            }
        }
        Ok(())
    }

    pub fn take_announcements(&mut self) -> Vec<PoolJob> {
        std::mem::take(&mut self.announced)
    }

    pub fn take_notices(&mut self) -> Vec<(String, Value)> {
        std::mem::take(&mut self.notices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_forms() {
        let e = Endpoint::parse("prl-eu.kryptex.network:7048").unwrap();
        assert_eq!((e.host.as_str(), e.port, e.tls), ("prl-eu.kryptex.network", 7048, false));
        let e = Endpoint::parse("stratum+ssl://prl-eu.kryptex.network:8048").unwrap();
        assert!(e.tls && e.port == 8048);
        assert_eq!(e.to_string(), "stratum+ssl://prl-eu.kryptex.network:8048");
        assert!(!Endpoint::parse("stratum+tcp://h:1").unwrap().tls);
        let e = Endpoint::parse("[2001:db8::1]:3333").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("2001:db8::1", 3333));
        assert!(Endpoint::parse("http://h:1").is_err());
        assert!(Endpoint::parse("h").is_err());
        assert!(Endpoint::parse("h:99999").is_err());
    }
}

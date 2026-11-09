//! Consensus-facing profile data: kernel profiles, mining configuration, pool jobs.
//! Arithmetic and wire types come from the upstream zk-pow verifier crate.
use anyhow::{bail, ensure, Context, Result};
use primitive_types::U256;
use zk_pow::api::{
    proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern},
    proof_utils::nbits_to_difficulty,
    sanity_checks::penalized_target_bound,
};

pub const M: usize = 32768;
pub const N: usize = 32768;
pub const RANK: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Kernel 780: packed rank-128 factors, K 8192..65536.
    K780,
    /// Kernel 448: dense prepared operands, K = 4096 (pools without gzip proofs).
    K448,
}

#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub kernel: Kernel,
    pub k: usize,
}

impl Profile {
    pub fn new(kernel: Kernel, k: usize) -> Result<Self> {
        match kernel {
            Kernel::K780 => ensure!(matches!(k, 8192 | 16384 | 32768 | 65536), "kernel 780 supports K 8192/16384/32768/65536"),
            Kernel::K448 => ensure!(k == 4096, "kernel 448 supports K 4096 only"),
        }
        Ok(Profile { kernel, k })
    }
    pub fn variant(&self) -> i32 {
        match self.kernel {
            Kernel::K780 => 780,
            Kernel::K448 => 448,
        }
    }
    /// Rows sampled per 8-row stripe group.
    pub fn sampling_rows(&self) -> usize {
        match self.kernel {
            Kernel::K780 => 10,
            Kernel::K448 => 12,
        }
    }
    pub fn period(&self) -> usize {
        self.sampling_rows() * 8
    }
    /// Counted MACs per round: whole sampling periods only (780: whole 160-row CTAs).
    pub fn macs_per_round(&self) -> f64 {
        let rows = match self.kernel {
            Kernel::K780 => M / 160 * 160,
            Kernel::K448 => M / self.period() * self.period(),
        };
        rows as f64 * N as f64 * self.k as f64
    }
    pub fn groups(&self) -> u32 {
        ((M / self.period()) * (N / 64) * 32) as u32
    }
    pub fn config(&self) -> Result<MiningConfiguration> {
        Ok(MiningConfiguration {
            common_dim: self.k as u32,
            rank: RANK as u16,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&(0..self.sampling_rows() as u32).map(|i| i * 8).collect::<Vec<u32>>())?,
            cols_pattern: PeriodicPattern::from_list(&(0..8u32).flat_map(|j| [j * 8, j * 8 + 1]).collect::<Vec<u32>>())?,
            moe: None,
        })
    }
    /// Matrix rows of A and columns of B covered by a hit group.
    pub fn indices(&self, group: u32) -> (Vec<usize>, Vec<usize>) {
        let (tile, lane) = (group as usize / 32, group as usize % 32);
        let (mt, nt) = (tile / (N / 64), tile % (N / 64));
        let rows = (0..self.sampling_rows()).map(|i| mt * self.period() + i * 8 + lane / 4).collect();
        let cols = (0..8).flat_map(|j| [nt * 64 + j * 8 + (lane % 4) * 2, nt * 64 + j * 8 + (lane % 4) * 2 + 1]).collect();
        (rows, cols)
    }
}

pub fn unhex(s: &str) -> Result<Vec<u8>> {
    ensure!(s.len() % 2 == 0, "odd hexadecimal length");
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).context("invalid hexadecimal")).collect()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Largest compact target not above `value` (never eases the pool target).
pub fn compact_floor(value: U256) -> Result<u32> {
    ensure!(!value.is_zero(), "zero share target");
    let mut size = value.bits().div_ceil(8) as u32;
    let mut mant = if size <= 3 { value.low_u32() << (8 * (3 - size)) } else { (value >> (8 * (size - 3))).low_u32() };
    if mant & 0x800000 != 0 {
        mant >>= 8;
        size += 1;
    }
    Ok(size << 24 | mant)
}

pub fn compact_value(bits: u32) -> U256 {
    nbits_to_difficulty(bits)
}

/// Normalized hash work of one share: 2^256 / (target + 1).
pub fn share_work(bits: u32) -> f64 {
    let t = compact_value(bits);
    let f = (0..4).rev().fold(0.0f64, |acc, i| acc * 18446744073709551616.0 + t.0[i] as f64);
    2f64.powi(256) / (f + 1.0)
}

/// A pool job (`mining.notify` params).
#[derive(Clone, Debug, PartialEq)]
pub struct PoolJob {
    pub job_id: String,
    pub header: Vec<u8>,
    pub raw_target: U256,
    pub cert_version: u32,
}

impl PoolJob {
    pub fn parse(p: &serde_json::Value) -> Result<Self> {
        let header = unhex(p["header"].as_str().context("notify header missing")?)?;
        ensure!(header.len() == 76, "pool header must contain exactly 76 bytes");
        let target = p["target"].as_str().context("notify target missing")?;
        ensure!(target.len() == 64, "pool target must contain exactly 32 bytes");
        let raw_target = U256::from_big_endian(&unhex(target)?);
        ensure!(!raw_target.is_zero(), "invalid pool target");
        let cert_version = p["cert_version"].as_u64().context("cert_version missing")? as u32;
        if cert_version != 2 && cert_version != 3 {
            bail!("unsupported pool cert_version {cert_version}");
        }
        let job_id = p["job_id"].as_str().context("missing pool job id")?.to_owned();
        ensure!(!job_id.is_empty(), "missing pool job id");
        Ok(PoolJob { job_id, header, raw_target, cert_version })
    }
    /// Same previous block, target and certificate: a same-height refresh.
    pub fn same_work(&self, other: &PoolJob) -> bool {
        self.header[4..36] == other.header[4..36] && self.raw_target == other.raw_target && self.cert_version == other.cert_version
    }
}

/// Everything the GPU and proof workers need for one job and profile.
#[derive(Clone)]
pub struct MiningJob {
    pub pool: PoolJob,
    pub header: IncompleteBlockHeader,
    pub profile: Profile,
    pub bits: u32,
    /// Penalized target bound, little-endian uint256 (kernel input).
    pub bound: [u8; 32],
    /// blake3(header || config): the job key.
    pub key: [u8; 32],
}

impl MiningJob {
    pub fn new(pool: PoolJob, profile: Profile) -> Result<Self> {
        let header = IncompleteBlockHeader::from_bytes(&pool.header)?;
        let bits = compact_floor(pool.raw_target)?;
        let config = profile.config()?;
        let bound = penalized_target_bound(compact_value(bits), &config).context("scaled pool target overflows uint256")?;
        let mut le = [0u8; 32];
        bound.to_little_endian(&mut le);
        let mut h = blake3::Hasher::new();
        h.update(&header.to_bytes());
        h.update(&config.to_bytes());
        Ok(MiningJob { pool, header, profile, bits, bound: le, key: *h.finalize().as_bytes() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_never_eases_target() {
        for v in [U256::from(1), U256::from(0x80ffffu64), U256::MAX >> 32, U256::MAX >> 1] {
            let b = compact_floor(v).unwrap();
            assert!(compact_value(b) <= v);
            assert_eq!(compact_floor(compact_value(b)).unwrap(), b);
        }
    }
    #[test]
    fn group_indices_layout() {
        let p = Profile::new(Kernel::K780, 65536).unwrap();
        // group 33: tile 1 (mt 0, nt 1), lane 1
        let (r, c) = p.indices(33);
        assert_eq!(r, (0..10).map(|i| i * 8).collect::<Vec<_>>());
        assert_eq!(&c[..4], &[64 + 2, 64 + 3, 64 + 8 + 2, 64 + 8 + 3]);
        assert_eq!(p.groups(), 409 * 512 * 32);
    }
}

// SPDX-License-Identifier: ISC
// FP8 (PIP-3 / cert v4) mining kernel derived from kernel 780 u6.
// CTA tile 160x256 (8 warps 2x4, warp tile 80x64), mma.sync m16n8k32 e4m3 -> f32 (bit-exact B200 replay on sm_120),
// operands A' (M x K) and B' (N x K) pre-noised/quantized and pre-packed in fragment order, streamed through an
// NS-stage cp.async.bulk ring. Epilogue: lottery tiles 16x32 (h x w); each tile row (32 cells) is one jackpot lane:
// x = rotl13(x * 0x9E3779B1 + bits(cell)) in column order; 16 lanes -> 64-byte message -> keyed BLAKE3 (jackpot key)
// -> compare (LE U256) with bound = target * h * w * k.
#ifdef _WIN32
#define PM_API extern "C" __declspec(dllexport)
#else
#define PM_API extern "C"
#endif
#include <cuda_runtime.h>
#include <cstdint>

#ifndef F780_NS
#define F780_NS 6
#endif
#ifndef F780_DB
#define F780_DB 1
#endif
#ifndef F780_SWZ
#define F780_SWZ (1 << 20)  // default: column-major over all row blocks (A' should fit in L2, 64 MiB)
#endif
#ifndef F780_DIAG_NOLOAD
#define F780_DIAG_NOLOAD 0
#endif

namespace f780 {
constexpr int BM = 160, BN = 256, WM = 80, WN = 64, MI = WM / 16, NJ = WN / 8, NT = 256, NS = F780_NS;
constexpr int FA = BM / 16, FB = BN / 16;           // 16-row fragments per CTA tile
constexpr int STA = FA * 512, STB = FB * 512, STG = STA + STB;  // bytes per K32 stage
constexpr int OFF_STG = 0, OFF_BAR = NS * STG, SHARED = OFF_BAR + NS * 8;
static_assert(SHARED <= 101376 - 1024, "smem");
constexpr int TH = 16, TW = 32;                      // lottery tile

struct Params {
  const uint8_t *pa, *pb;   // packed: pa[(mb*S + s)*STA + f*512 + lane*16 + r*4 + byte], pb likewise with STB
  int M, N, K;
  const uint32_t *key;      // jackpot key, 8 LE words
  const uint32_t *bound;    // 8 LE words (word 7 most significant)
  uint32_t *count, *hits;   // hits: (tile_row << 16) | tile_col
  int cap;
  uint32_t *digests;        // optional: all digests [(tr * (N/TW) + tc) * 8]
  unsigned long long *cyc;  // optional: per-CTA [cycles, mac/4096] (diagnostic)
};

__device__ __forceinline__ unsigned sa(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
#define LDS4(a, r0, r1, r2, r3) asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory")

__device__ __forceinline__ void mma(float *c, const uint32_t *a, const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- BLAKE3 single-block keyed compression ----
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ void G(uint32_t *s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx; s[d] = rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];      s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my; s[d] = rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];      s[b] = rotr(s[b] ^ s[c], 7);
}
__device__ __forceinline__ void blake3_keyed64(const uint32_t *key, uint32_t *m, uint32_t *out) {
  const uint32_t IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
  uint32_t s[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i] = key[i];
  s[8] = IV[0]; s[9] = IV[1]; s[10] = IV[2]; s[11] = IV[3];
  s[12] = 0; s[13] = 0; s[14] = 64; s[15] = 1 | 2 | 8 | 16;  // CHUNK_START|CHUNK_END|ROOT|KEYED_HASH
  constexpr int P[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    G(s, 0, 4, 8, 12, m[0], m[1]); G(s, 1, 5, 9, 13, m[2], m[3]);
    G(s, 2, 6, 10, 14, m[4], m[5]); G(s, 3, 7, 11, 15, m[6], m[7]);
    G(s, 0, 5, 10, 15, m[8], m[9]); G(s, 1, 6, 11, 12, m[10], m[11]);
    G(s, 2, 7, 8, 13, m[12], m[13]); G(s, 3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {
      uint32_t t[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) t[i] = m[P[i]];
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = s[i] ^ s[i + 8];
}

__device__ __forceinline__ uint32_t fstep(uint32_t x, float v) { return __funnelshift_l(x * 0x9E3779B1u + __float_as_uint(v), x * 0x9E3779B1u + __float_as_uint(v), 13); }

__global__ __launch_bounds__(NT, 1) void kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wr = warp / (BN / WN), wc = warp % (BN / WN);
  const int S = p.K / 32, tilesN = p.N / BN, tiles = (p.M / BM) * tilesN;
  const unsigned bar0 = sa(sm + OFF_BAR), stg0 = sa(sm + OFF_STG);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar0 + 8 * i) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const long long c0 = clock64();
  unsigned long long nmma = 0;
  uint32_t phase = 0;  // bit i: parity of stage buffer i
  // per-warp fragment offsets inside a stage
  const unsigned aoff = (wr * MI) * 512 + lane * 16;               // + i*512
  const unsigned boff = STA + (wc * (NJ / 2)) * 512 + lane * 16;   // + jp*512 (2 n8 blocks per 16-row fragment)
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
#if F780_SWZ > 0
    // grouped order: F780_SWZ row blocks share their B tiles while they are hot in L2
    const int tilesM = p.M / BM, grpT = F780_SWZ * tilesN, first = (tile / grpT) * F780_SWZ, gsz = min(tilesM - first, F780_SWZ), ti = tile % grpT;
    const int bm = first + ti % gsz, bn = ti / gsz;
#else
    const int bm = tile / tilesN, bn = tile % tilesN;
#endif
    const uint8_t *ga = p.pa + size_t(bm) * S * STA, *gb = p.pb + size_t(bn) * S * STB;
    auto issue = [&](int s) {
      const int b = s % NS;
      const unsigned bar = bar0 + 8 * b, dst = stg0 + b * STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(STG) : "memory");
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(ga + size_t(s) * STA), "r"(STA), "r"(bar) : "memory");
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst + STA), "l"(gb + size_t(s) * STB), "r"(STB), "r"(bar) : "memory");
    };
    __syncthreads();  // previous tile's epilogue / reads done
    if (tid == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      for (int s = 0; s < NS - 1 && s < S; ++s) issue(s);
    }
    float acc[MI][NJ][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
#if F780_DB
    // Register double buffer: fragments of step s+1 load while step s's MMAs issue. One barrier per step.
    uint32_t af[2][MI][4], bf[2][NJ][2];
    auto waitst = [&](int s) {
      const int b = s % NS;
      asm volatile("{ .reg .pred q; L: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L; }" ::"r"(bar0 + 8 * b), "r"((phase >> b) & 1u) : "memory");
      phase ^= 1u << b;
    };
    if (tid == 0 && NS - 1 < S) issue(NS - 1);  // prologue issued NS-1 stages; fill the last buffer too
    waitst(0);
#pragma unroll
    for (int i = 0; i < MI; ++i) LDS4(stg0 + aoff + i * 512, af[0][i][0], af[0][i][1], af[0][i][2], af[0][i][3]);
#pragma unroll
    for (int jp = 0; jp < NJ / 2; ++jp) LDS4(stg0 + boff + jp * 512, bf[0][2 * jp][0], bf[0][2 * jp][1], bf[0][2 * jp + 1][0], bf[0][2 * jp + 1][1]);
    auto step = [&]<int C>(const int s) {
      constexpr int N_ = C ^ 1;
      const bool more = s + 1 < S;
      unsigned st = 0;
#if F780_DIAG_NOLOAD  // diagnostic: no L2->SM traffic after the prologue (wrong results, energy split only)
      if (more) { st = stg0 + ((s + 1) % NS) * STG; }
#else
      if (more) { waitst(s + 1); st = stg0 + ((s + 1) % NS) * STG; }
#endif
#pragma unroll
      for (int pp = 0; pp < NJ * MI; ++pp) {
        const int j = pp / MI, i = pp % MI;
        mma(acc[i][j], af[C][i], bf[C][j]);
        if (more) {
          if (pp % 4 == 1 && pp / 4 < MI) { const int x = pp / 4; LDS4(st + aoff + x * 512, af[N_][x][0], af[N_][x][1], af[N_][x][2], af[N_][x][3]); }
          if (pp % 4 == 3 && pp / 4 >= 2 && pp / 4 < 2 + NJ / 2) { const int jp = pp / 4 - 2;
            LDS4(st + boff + jp * 512, bf[N_][2 * jp][0], bf[N_][2 * jp][1], bf[N_][2 * jp + 1][0], bf[N_][2 * jp + 1][1]); }
        }
      }
      __syncthreads();  // all warps hold stage s+1 in registers -> buffer of stage s is free
      if (tid == 0 && s + NS < S && !F780_DIAG_NOLOAD) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue(s + NS);
      }
    };
#pragma unroll 1
    for (int s = 0; s < S; s += 2) {
      step.template operator()<0>(s);
      step.template operator()<1>(s + 1);
    }
#else
#pragma unroll 1
    for (int s = 0; s < S; ++s) {
      const int b = s % NS;
      asm volatile("{ .reg .pred q; L: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1; @!q bra L; }" ::"r"(bar0 + 8 * b), "r"((phase >> b) & 1u) : "memory");
      phase ^= 1u << b;
      const unsigned st = stg0 + b * STG;
      uint32_t af[MI][4], bf[NJ][2];
#pragma unroll
      for (int i = 0; i < MI; ++i) LDS4(st + aoff + i * 512, af[i][0], af[i][1], af[i][2], af[i][3]);
#pragma unroll
      for (int jp = 0; jp < NJ / 2; ++jp) LDS4(st + boff + jp * 512, bf[2 * jp][0], bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1]);
      __syncthreads();  // everyone has read stage s-1's buffer -> refill it with stage s+NS-1
      if (tid == 0 && s + NS - 1 < S) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue(s + NS - 1);
      }
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int i = 0; i < MI; ++i) mma(acc[i][j], af[i], bf[j]);
    }
#endif
    // ---- epilogue: 10 lottery tiles of 16x32 per warp ----
    __syncthreads();  // stage buffers free -> reuse as word scratch
    uint32_t *ws = (uint32_t *)(sm + OFF_STG) + warp * (MI * 2 * 16);
    const int q = lane & 3, qb = lane & ~3, g = lane >> 2;
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t xl = 0, xh = 0;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int j = h * 4 + jj;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            if (q == t) {
              xl = fstep(fstep(xl, acc[i][j][0]), acc[i][j][1]);
              xh = fstep(fstep(xh, acc[i][j][2]), acc[i][j][3]);
            }
            xl = __shfl_sync(0xffffffffu, xl, qb | t);
            xh = __shfl_sync(0xffffffffu, xh, qb | t);
          }
        }
        if (q == 0) { ws[(i * 2 + h) * 16 + g] = xl; ws[(i * 2 + h) * 16 + g + 8] = xh; }
      }
    __syncwarp();
    if (lane < MI * 2) {
      const int i = lane >> 1, h = lane & 1;
      uint32_t m[16], key[8], d[8];
#pragma unroll
      for (int w = 0; w < 16; ++w) m[w] = ws[lane * 16 + w];
#pragma unroll
      for (int w = 0; w < 8; ++w) key[w] = p.key[w];
      blake3_keyed64(key, m, d);
      const int tr = (bm * BM + wr * WM + i * 16) / TH, tc = (bn * BN + wc * WN + h * 32) / TW;
      if (p.digests) {
#pragma unroll
        for (int w = 0; w < 8; ++w) p.digests[(size_t(tr) * (p.N / TW) + tc) * 8 + w] = d[w];
      }
      bool good = true;
#pragma unroll
      for (int w = 7; w >= 0; --w) {
        const uint32_t bw = p.bound[w];
        if (d[w] > bw) { good = false; break; }
        if (d[w] < bw) break;
      }
      if (good) {
        const uint32_t at = atomicAdd(p.count, 1u);
        if (at < uint32_t(p.cap)) p.hits[at] = (uint32_t(tr) << 16) | uint32_t(tc);
      }
    }
    nmma += (unsigned long long)S * (BM / 16) * (BN / 8);
  }
  __syncthreads();
  if (p.cyc && tid == 0) { p.cyc[2 * blockIdx.x] = clock64() - c0; p.cyc[2 * blockIdx.x + 1] = nmma; }
}

// Row-major (rows x K) e4m3 -> packed stage layout. rows % 16 == 0, K % 32 == 0, rowsPerBlk = BM or BN.
__global__ void pack(const uint8_t *src, uint8_t *dst, int rows, int K, int rowsPerBlk) {
  const int S = K / 32, frags = rowsPerBlk / 16;
  const size_t total = size_t(rows) * K / 16;  // one 16-byte lane record per thread
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total; idx += size_t(gridDim.x) * blockDim.x) {
    const int lane = idx % 32;
    size_t r = idx / 32;
    const int f = r % frags; r /= frags;
    const int s = r % S; const size_t blk = r / S;
    const int g = lane >> 2, t = lane & 3;
    const size_t row0 = blk * rowsPerBlk + f * 16;
    const uint8_t *p0 = src + (row0 + g) * K + s * 32, *p1 = src + (row0 + g + 8) * K + s * 32;
    uint32_t w[4];
    // A layout (16 rows): a0 row g k4t, a1 row g+8 k4t, a2 row g k16+4t, a3 row g+8 k16+4t.
    // B layout (16 rows = two n8 blocks): lds4 -> b[2jp][0], b[2jp][1], b[2jp+1][0], b[2jp+1][1]
    //   n8 block 0 = rows 0..7 (row g), block 1 = rows 8..15 (row g+8); b0 = k4t, b1 = k16+4t.
    // Both A and B therefore use: w0 = row g k4t | A: w1 = row g+8 k4t, w2 = row g k16+4t, w3 = row g+8 k16+4t
    //                                             | B: w1 = row g k16+4t, w2 = row g+8 k4t, w3 = row g+8 k16+4t
    const bool isB = rowsPerBlk == BN;
    auto ld = [](const uint8_t *q) { return uint32_t(q[0]) | uint32_t(q[1]) << 8 | uint32_t(q[2]) << 16 | uint32_t(q[3]) << 24; };
    w[0] = ld(p0 + 4 * t);
    if (!isB) { w[1] = ld(p1 + 4 * t); w[2] = ld(p0 + 16 + 4 * t); w[3] = ld(p1 + 16 + 4 * t); }
    else { w[1] = ld(p0 + 16 + 4 * t); w[2] = ld(p1 + 4 * t); w[3] = ld(p1 + 16 + 4 * t); }
    uint32_t *o = (uint32_t *)(dst + ((blk * S + s) * frags + f) * 512 + lane * 16);
    o[0] = w[0]; o[1] = w[1]; o[2] = w[2]; o[3] = w[3];
  }
}
}  // namespace f780

PM_API int pm_f780_pack(const void *src, void *dst, int rows, int K, int isB, void *stream) {
  const int rpb = isB ? f780::BN : f780::BM;
  if (rows % rpb || K % 32) return cudaErrorInvalidValue;
  f780::pack<<<1024, 256, 0, (cudaStream_t)stream>>>((const uint8_t *)src, (uint8_t *)dst, rows, K, rpb);
  return cudaGetLastError();
}

PM_API int pm_f780_launch(const void *pa, const void *pb, int m, int n, int k, const void *key, const void *bound, void *count, void *hits, int cap,
                           void *digests, int grid, void *stream, void *cyc) {
  using namespace f780;
  if (m % BM || n % BN || k % 32 || k < 32 * NS) return cudaErrorInvalidValue;
  Params p{(const uint8_t *)pa, (const uint8_t *)pb, m, n, k, (const uint32_t *)key, (const uint32_t *)bound, (uint32_t *)count, (uint32_t *)hits, cap, (uint32_t *)digests, (unsigned long long *)cyc};
  auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED);
  if (e) return e;
  if (grid <= 0) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&grid, cudaDevAttrMultiProcessorCount, dev);
  }
  kernel<<<grid, NT, SHARED, (cudaStream_t)stream>>>(p);
  return cudaGetLastError();
}

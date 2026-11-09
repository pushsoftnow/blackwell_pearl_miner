// SPDX-License-Identifier: ISC
// Mining kernel on dense prepared operands, K = 4096.
// CTA tile 192x256 (8 warps, warp tile 96x64), TMA operand tiles, two pipeline stages.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include "gemm/pow_utils.hpp"

namespace k448 {
constexpr int COMMON_K = 4096, SWIZZLE = 32, BM = 192, BN = 256, WM = 96, WN = 64, STAGES = 2;
constexpr int BK = 64, NT = BM * BN / (WM * WN) * 32;
constexpr int MI = WM / 16, NJ = WN / 8, NWN = BN / WN;
constexpr int GROUPS = NT;
constexpr int SB = (BM + BN) * BK;
constexpr int TRACE = GROUPS * 16 * 4;
constexpr int SHARED = SB * STAGES + TRACE + 128;
static_assert(BN == 256 && WM % 16 == 0 && WN == 64 && BN % WN == 0 && BM % WM == 0);

struct Maps {
  CUtensorMap a{}, b{};
  const void *pa = nullptr, *pb = nullptr;
};
static thread_local Maps maps;

__device__ __forceinline__ unsigned address(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void copy_tile(int8_t *sa, int8_t *sb, const CUtensorMap *ma, const CUtensorMap *mb, int row, int col, int k, unsigned bar) {
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(SB) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];" ::"r"(address(sa)), "l"(ma), "r"(bar), "r"(k * 4), "r"(row / 16) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4}], [%2];" ::"r"(address(sb)), "l"(mb), "r"(bar), "r"(k * 4), "r"(col / 16) : "memory");
  }
}

__global__ __launch_bounds__(NT) void kernel(const uint32_t *__restrict__ key, const uint32_t *__restrict__ target, uint32_t *count, uint32_t *ids, int capacity,
                                             const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb) {
  extern __shared__ __align__(1024) int8_t smem[];
  for (int logical = blockIdx.x; logical < ((32768 + BM - 1) / BM) * (32768 / BN); logical += gridDim.x) {
    auto *trace = (uint32_t *)(smem + SB * STAGES);
    auto *barriers = (uint64_t *)(smem + SB * STAGES + TRACE);
    constexpr int MT = (32768 + BM - 1) / BM;
    int strip = logical / (MT * SWIZZLE), off = logical % (MT * SWIZZLE);
    int bm = off / SWIZZLE, bn = strip * SWIZZLE + off % SWIZZLE;
    int row = bm * BM, col = bn * BN;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int wr = warp / NWN, wc = warp % NWN;
    int32_t acc[MI][NJ][4] = {};
    if (logical == blockIdx.x && threadIdx.x == 0) {
#pragma unroll
      for (int s = 0; s < STAGES; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(barriers + s)) : "memory");
#pragma unroll
      for (int s = 0; s < STAGES; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(address(barriers + STAGES + s)), "r"(NT / 32) : "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if (logical == blockIdx.x) __syncthreads();
#pragma unroll
    for (int s = 0; s < (logical == blockIdx.x ? STAGES : 0); ++s) copy_tile(smem + s * SB, smem + s * SB + BM * 64, &ma, &mb, row, col, s * 64, address(barriers + s));
#pragma unroll(16)
    for (int tile = 0; tile < COMMON_K / 64; ++tile) {
      int stage = tile % STAGES;
      int8_t *sa = smem + stage * SB;
      int8_t *sb = sa + BM * 64;
      asm volatile("{ .reg .pred p; L: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra L; }" ::"r"(address(barriers + stage)), "r"((tile / STAGES) & 1) : "memory");
#pragma unroll
      for (int kk = 0; kk < 64; kk += 32) {
        uint32_t af[MI][4], bf[NJ][2];
#pragma unroll
        for (int i = 0; i < MI; ++i) {
          unsigned p = address(sa + ((wr * MI + i) * 2 + kk / 32) * 512 + lane * 16);
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3]) : "r"(p) : "memory");
        }
#pragma unroll
        for (int j = 0; j < NJ; j += 2) {
          unsigned p = address(sb + ((wc * (NJ / 2) + j / 2) * 2 + kk / 32) * 512 + lane * 16);
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bf[j][0]), "=r"(bf[j][1]), "=r"(bf[j + 1][0]), "=r"(bf[j + 1][1]) : "r"(p) : "memory");
        }
        // After the last operand load of a stage: release it and refill it during the MMAs.
        if (kk == 32) {
          __syncwarp();
          if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(address(barriers + STAGES + stage)) : "memory");
          if (threadIdx.x == 0) {
            asm volatile("{ .reg .pred p; L: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra L; }" ::"r"(address(barriers + STAGES + stage)), "r"((tile / STAGES) & 1) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          }
          if (tile + STAGES < COMMON_K / 64) copy_tile(sa, sb, &ma, &mb, row, col, (tile + STAGES) * 64, address(barriers + stage));
          else if (logical + gridDim.x < ((32768 + BM - 1) / BM) * (32768 / BN)) {
            int next = logical + gridDim.x;
            int next_strip = next / (MT * SWIZZLE), next_off = next % (MT * SWIZZLE);
            int next_row = (next_off / SWIZZLE) * BM;
            int next_col = (next_strip * SWIZZLE + next_off % SWIZZLE) * BN;
            copy_tile(sa, sb, &ma, &mb, next_row, next_col, (tile + STAGES - COMMON_K / 64) * 64, address(barriers + stage));
          }
        }
#pragma unroll
        for (int i = 0; i < MI; ++i) {
#pragma unroll
          for (int j = 0; j < NJ; ++j) {
            auto *c = acc[i][j];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                         : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                         : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
          }
        }
      }
      if (tile % 2 == 1) {
        uint32_t h = 0;
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
          for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) h ^= uint32_t(acc[i][j][c]);
        int step = tile / 2, slot = step % 16;
        trace[slot * GROUPS + threadIdx.x] = step < 16 ? h : pearl::rotl_xor<13>(trace[slot * GROUPS + threadIdx.x], h);
      }
    }
    for (int local = threadIdx.x; local < GROUPS; local += NT) {
      if (row + wr * WM + WM > 32768) continue;
      auto msg = cute::make_tensor<uint32_t>(cute::Int<16>{});
      auto digest = cute::make_tensor<uint32_t>(cute::Int<8>{});
#pragma unroll
      for (int j = 0; j < 16; ++j) msg(j) = trace[j * GROUPS + local];
#pragma unroll
      for (int j = 0; j < 8; ++j) digest(j) = key[j];
      blake3::compress_msg_block_u32(msg, digest, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
      uint32_t group = ((row / WM + wr) * 512 + bn * 4 + wc) * 32 + lane;
      bool good = true;
#pragma unroll
      for (int j = 7; j >= 0; --j) {
        if (digest(j) > target[j]) { good = false; break; }
        if (digest(j) < target[j]) break;
      }
      if (good) {
        uint32_t at = atomicAdd(count, 1u);
        if (at < uint32_t(capacity)) ids[at] = group;
      }
    }
  }
}

int encode_maps(const void *a, const void *b) {
  if (maps.pa == a && maps.pb == b) return 0;
  cuuint64_t dims[2] = {COMMON_K * 4, 32768 / 16}, stride[1] = {COMMON_K * 16};
  cuuint32_t box[2] = {256, BM / 16}, steps[2] = {1, 1};
  auto encode = [&](CUtensorMap *out, const void *p) {
    return cuTensorMapEncodeTiled(out, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, const_cast<void *>(p), dims, stride, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  int rc = encode(&maps.a, a);
  if (rc) return rc;
  box[1] = BN / 16;
  rc = encode(&maps.b, b);
  if (rc) return rc;
  maps.pa = a;
  maps.pb = b;
  return 0;
}
} // namespace k448

// Operands prepared by pm_prepare_packed (A: is_b=0, B: is_b=1), M = N = 32768, K = 4096.
extern "C" int pm_k448_launch(const void *a, const void *b, const void *key, const void *target, void *count, void *ids, int capacity, void *stream) {
  using namespace k448;
  int rc = encode_maps(a, b);
  if (rc) return rc;
  auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED);
  if (e) return e;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e) return e;
  kernel<<<sms, NT, SHARED, (cudaStream_t)stream>>>((const uint32_t *)key, (const uint32_t *)target, (uint32_t *)count, (uint32_t *)ids, capacity, maps.a, maps.b);
  return cudaGetLastError();
}

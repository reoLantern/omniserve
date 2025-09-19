// fake_launch_gemm_w4a8_per_group.cu
// nvcc -std=c++17 -arch=sm_80 fake_launch_gemm_w4a8_per_group.cu -o fake_gemm
// 运行： ./fake_gemm
//
// 说明：该文件会 #include 你提供的 gemm_w4a8_per_group.cu（仅包含 kernel 实现）
// 确保两者放在同一目录，或把包含路径改为实际路径。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cassert>

#include "gemm_w4a8_per_group.cuh"

// ---- CUDA error check ----
#define CK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

int main() {

  int dev = 0;
  CK(cudaGetDevice(&dev));

  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("=== Device %d: %s (cc %d.%d) ===\n",
         dev, prop.name, prop.major, prop.minor);

  // 设备级上限（per block / per SM）
  int maxShmemPerBlock = 0, maxShmemPerBlockOptin = 0, maxShmemPerSM = 0;
  int maxRegsPerBlock = 0, maxRegsPerSM = 0, warpSize = 0;
  CK(cudaDeviceGetAttribute(&maxShmemPerBlock,
      cudaDevAttrMaxSharedMemoryPerBlock, dev));
  CK(cudaDeviceGetAttribute(&maxShmemPerBlockOptin,
      cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  CK(cudaDeviceGetAttribute(&maxShmemPerSM,
      cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  CK(cudaDeviceGetAttribute(&maxRegsPerBlock,
      cudaDevAttrMaxRegistersPerBlock, dev));
  CK(cudaDeviceGetAttribute(&maxRegsPerSM,
      cudaDevAttrMaxRegistersPerMultiprocessor, dev));
  CK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, dev));

  printf("Device caps:\n");
  printf("  Shared mem per BLOCK (default/opt-in): %d / %d bytes\n",
         maxShmemPerBlock, maxShmemPerBlockOptin);
  printf("  Shared mem per SM: %d bytes\n", maxShmemPerSM);
  printf("  Registers per BLOCK/SM: %d / %d\n", maxRegsPerBlock, maxRegsPerSM);
  printf("  Warp size: %d, SMs: %d\n", warpSize, prop.multiProcessorCount);

  // ---- 根据 JSON 设定 ----
  const int M = 2048;    // num_out_feats
  const int N = 6144;    // num_out_channels
  const int K = 4096;    // num_in_channels
  const int G = 128;     // per-group size

  // kernel 模板参数
  constexpr int CTA_M = 128;
  constexpr int CTA_N = 64;
  constexpr int CTA_K = 64;
  constexpr int WARP_M = 64;
  constexpr int WARP_N = 32;
  constexpr int WARP_K = 64;
  constexpr int STAGES = 4;

  // ---- 计算 grid / block / smem（与 KERNEL_LAUNCH_CODE 完全一致）----

  // ceil(M/CTA_M)
  int num_blocks_m = (M + CTA_M - 1) / CTA_M;
  int num_blocks_n = N / CTA_N;

  const int log_tile = get_log_tile<8>(num_blocks_m);
  const int tile_shift = 1 << log_tile;
  dim3 grid(num_blocks_n * tile_shift,
            (num_blocks_m + tile_shift - 1) / tile_shift, 1);

  constexpr int NUM_WARPS =
      (CTA_M / WARP_M) * (CTA_N / WARP_N) * (CTA_K / WARP_K);
  dim3 block(WARP_SIZE, NUM_WARPS, 1);

  // 动态 shared memory 计算
  constexpr int SCALES_SMEM_SIZE =
      (G >= CTA_K) ? (CTA_N * STAGES * 2) : (CTA_N * (CTA_K / G) * STAGES * 2);
  constexpr int kSmemByteSize =
      ((CTA_M * (CTA_K + SMEM_PAD_A) + CTA_N * (CTA_K + SMEM_PAD_B) / 2) * STAGES
        + SCALES_SMEM_SIZE) * sizeof(int8_t);

  printf("grid=(%u,%u,%u) block=(%u,%u,%u) smem=%d\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z, kSmemByteSize);

  // ---- 分配设备内存（按 W4A8 per-group 布局）----
  // A: [M,K] int8
  size_t bytesA = size_t(M) * K * sizeof(int8_t);
  // B: [N, K/2] int8（4bit pack）
  size_t bytesB = size_t(N) * (K/2) * sizeof(int8_t);
  // zeros/scales_i8: [(K/G), N] int8
  const int groups = K / G;
  size_t bytesZS = size_t(groups) * N * sizeof(int8_t);
  // wscales: [N] half（内核按 half2* 读取，每两列共用一对）
  size_t bytesW = size_t(N) * sizeof(half);
  // ascales: [M] half
  size_t bytesAS = size_t(M) * sizeof(half);
  // C: [M,N] half
  size_t bytesC = size_t(M) * N * sizeof(half);

  printf("Alloc sizes(MB): A=%.2f B=%.2f Z/S=%.2f W=%.2f As=%.2f C=%.2f\n",
         bytesA/1048576.0, bytesB/1048576.0, (2*bytesZS)/1048576.0,
         bytesW/1048576.0, bytesAS/1048576.0, bytesC/1048576.0);

  int8_t *dA=nullptr, *dB=nullptr, *dZ=nullptr, *dS=nullptr;
  half *dW=nullptr, *dAS=nullptr, *dC=nullptr;
  CK(cudaMalloc(&dA,  bytesA));
  CK(cudaMalloc(&dB,  bytesB));
  CK(cudaMalloc(&dZ,  bytesZS));
  CK(cudaMalloc(&dS,  bytesZS));
  CK(cudaMalloc(&dW,  bytesW));
  CK(cudaMalloc(&dAS, bytesAS));
  CK(cudaMalloc(&dC,  bytesC));
  CK(cudaMemset(dC, 0, bytesC));

  // ---- 随机初始化 host 数据并拷贝到设备 ----
  std::mt19937 rng(25);
  std::uniform_int_distribution<int> distA(-127, 127);
  std::uniform_int_distribution<int> distNibble(0, 15);
  std::uniform_int_distribution<int> distSmall(1, 8);

  {
    std::vector<int8_t> hA(bytesA);
    for (size_t i=0;i<bytesA;i++) hA[i] = static_cast<int8_t>(distA(rng));
    CK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  }
  {
    // B: 每字节两组 4bit（0..15），(hi<<4)|lo
    std::vector<int8_t> hB(bytesB);
    for (size_t i=0;i<bytesB;i++) {
      uint8_t lo = distNibble(rng);
      uint8_t hi = distNibble(rng);
      hB[i] = static_cast<int8_t>((hi<<4) | lo);
    }
    CK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  }
  {
    // zeros/scales_i8：小整数
    std::vector<int8_t> hZ(bytesZS), hS(bytesZS);
    for (size_t i=0;i<bytesZS;i++) {
      hZ[i] = static_cast<int8_t>(distNibble(rng));
      hS[i] = static_cast<int8_t>(distSmall(rng));
    }
    CK(cudaMemcpy(dZ, hZ.data(), bytesZS, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, hS.data(), bytesZS, cudaMemcpyHostToDevice));
  }
  {
    // wscales / ascales
    std::vector<half> hW(N), hAS(M);
    for (int i=0;i<N;i++) hW[i]  = __float2half(1.0f/16.0f);
    for (int i=0;i<M;i++) hAS[i] = __float2half(1.0f/16.0f);
    CK(cudaMemcpy(dW,  hW.data(),  bytesW,  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAS, hAS.data(), bytesAS, cudaMemcpyHostToDevice));
  }

  // ---- 设置动态 shared memory 上限并启动 kernel ----
  // wscales 以 half2* 传入
  auto kernel = dense_kernel0<CTA_M, CTA_N, CTA_K, WARP_M, WARP_N, WARP_K, STAGES, G>;

  // kernel 静态属性（编译器给出的 numRegs 等）
  cudaFuncAttributes fattr{};
  CK(cudaFuncGetAttributes(&fattr, kernel));
  printf("Kernel attrs:\n");
  printf("  numRegs per thread: %d\n", fattr.numRegs);
  printf("  static sharedSizeBytes: %d\n", fattr.sharedSizeBytes);
  printf("  maxDynamicSharedSizeBytes (current): %d\n",
         fattr.maxDynamicSharedSizeBytes);

  CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemByteSize));

  CK(cudaFuncGetAttributes(&fattr, kernel));
  printf("  maxDynamicSharedSizeBytes (after set): %d\n",
          fattr.maxDynamicSharedSizeBytes);

  // occupancy estimate
  int maxBlocksPerSM = 0;
  int threadsPerBlock = block.x * block.y * block.z;
  cudaError_t occErr =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &maxBlocksPerSM, kernel, threadsPerBlock, kSmemByteSize);
  if (occErr == cudaSuccess) {
    printf("Occupancy estimate: up to %d blocks/SM at %d threads, dynSmem=%zu\n",
           maxBlocksPerSM, threadsPerBlock, kSmemByteSize);
  } else {
    printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor error: %s\n",
           cudaGetErrorString(occErr));
  }

  // // ---- warm-up + 计时 ----
  // const int warmup = 5;
  // const int iters  = 50;

  // printf("Warming up (%d iters)...\n", warmup);
  // for (int i = 0; i < warmup; ++i) {
  //   kernel<<<grid, block, kSmemByteSize>>>(
  //       dA, dB, dZ, dS,
  //       reinterpret_cast<half2*>(dW),
  //       dAS, dC,
  //       M, N, K);
  // }
  // CK(cudaGetLastError());
  // CK(cudaDeviceSynchronize());

  // cudaEvent_t start, stop;
  // CK(cudaEventCreate(&start));
  // CK(cudaEventCreate(&stop));

  // printf("Timing kernel (%d iters)...\n", iters);
  // CK(cudaEventRecord(start));
  // for (int i = 0; i < iters; ++i) {
  //   kernel<<<grid, block, kSmemByteSize>>>(
  //       dA, dB, dZ, dS,
  //       reinterpret_cast<half2*>(dW),
  //       dAS, dC,
  //       M, N, K);
  // }
  // CK(cudaEventRecord(stop));
  // CK(cudaEventSynchronize(stop));

  // float total_ms = 0.0f;
  // CK(cudaEventElapsedTime(&total_ms, start, stop));
  // printf("Kernel time: total %.3f ms  |  avg %.3f ms/iter  |  %.3f us/launch\n",
  //        total_ms, total_ms/iters, 1000.0f*total_ms/iters);

  // CK(cudaEventDestroy(start));
  // CK(cudaEventDestroy(stop));

  printf("Launching kernel...\n");
  kernel<<<grid, block, kSmemByteSize>>>(
      dA, dB, dZ, dS,
      reinterpret_cast<half2*>(dW),
      dAS, dC,
      M, N, K);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  printf("Done.\n");

  // ---- 取回一小块 C 做 sanity check ----
  const int probe_rows = 8, probe_cols = 16;
  std::vector<half> hProbe(probe_rows * probe_cols);
  size_t src_pitch = size_t(N) * sizeof(half);
  size_t width_bytes = size_t(probe_cols) * sizeof(half);
  CK(cudaMemcpy2D(hProbe.data(), width_bytes,
                  dC, src_pitch,
                  width_bytes, probe_rows,
                  cudaMemcpyDeviceToHost));

  printf("C[0:%d, 0:%d):\n", probe_rows, probe_cols);
  for (int i=0;i<probe_rows;i++) {
    for (int j=0;j<probe_cols;j++) {
      float v = __half2float(hProbe[i*probe_cols + j]);
      printf("%7.1f ", v);
    }
    printf("\n");
  }

  // ---- clean ----
  cudaFree(dA); cudaFree(dB); cudaFree(dZ); cudaFree(dS);
  cudaFree(dW); cudaFree(dAS); cudaFree(dC);
  return 0;
}

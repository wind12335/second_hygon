// p2p_probe.cu — RTX 4090 x4 互联形态探测（交接第一步的关键证据）
// 输出两样东西：
//   1) cudaDeviceCanAccessPeer 矩阵（0/1）——4090 消费级主板经常禁 P2P，
//      这直接决定 NVSHMEM 走 device 直传还是 host proxy
//   2) 每个有序 GPU 对的 D2D 拷贝带宽（无 P2P 时会塌到走内存的中转路径）
// 编译: nvcc -o p2p_probe p2p_probe.cu
// 运行: CUDA_VISIBLE_DEVICES=0,1,2,3 ./p2p_probe
#include <chrono>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

int main() {
  int ndev = 0;
  CHECK(cudaGetDeviceCount(&ndev));
  printf("device_count=%d\n", ndev);
  std::vector<cudaDeviceProp> props(ndev);
  for (int i = 0; i < ndev; ++i) {
    CHECK(cudaGetDeviceProperties(&props[i], i));
    char pci[32] = {0};
    cudaDeviceGetPCIBusId(pci, sizeof(pci), i);
    printf("gpu%d: %s (%s) sm_%d%d\n", i, props[i].name, pci,
           props[i].major, props[i].minor);
  }
  if (ndev < 2) { printf("fewer than 2 devices; nothing to probe\n"); return 0; }

  printf("\ncan_access_peer[i][j] (row=i accesses j):\n     ");
  for (int j = 0; j < ndev; ++j) printf("%5d", j);
  printf("\n");
  for (int i = 0; i < ndev; ++i) {
    printf("%4d ", i);
    for (int j = 0; j < ndev; ++j) {
      int can = 0;
      if (i != j) cudaDeviceCanAccessPeer(&can, i, j);
      printf("%5d", can);
    }
    printf("\n");
  }

  // 带宽矩阵：每对两个方向，256 MiB，3 warmup + 5 reps 取中位数
  const size_t bytes = 256UL * 1024 * 1024;
  std::vector<void*> buf(ndev, nullptr);
  for (int i = 0; i < ndev; ++i) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&buf[i], bytes));
    CHECK(cudaMemset(buf[i], 0, bytes));
  }
  printf("\npeer_copy_bandwidth_gbps[i->j] (256 MiB, median of 5):\n     ");
  for (int j = 0; j < ndev; ++j) printf("%9d", j);
  printf("\n");
  for (int i = 0; i < ndev; ++i) {
    printf("%4d ", i);
    for (int j = 0; j < ndev; ++j) {
      if (i == j) { printf("%9s", "-"); continue; }
      // cudaMemcpyPeer 在无 P2P 时会自动走 host 中转——带宽塌陷即证据
      for (int r = 0; r < 3; ++r) cudaMemcpyPeer(buf[j], j, buf[i], i, bytes);
      double best = 0.0;
      for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        cudaMemcpyPeer(buf[j], j, buf[i], i, bytes);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gbps = (double)bytes / sec / 1e9;
        if (gbps > best) best = gbps;
      }
      printf("%9.1f", best);
    }
    printf("\n");
  }
  printf("\nnote: >~20 GB/s 且 can_access_peer=1 -> PCIe P2P 直传；"
         "~5-15 GB/s 或 can=0 -> 走 host 中转（NVSHMEM 将用 proxy 路径）\n");
  for (int i = 0; i < ndev; ++i) { cudaSetDevice(i); cudaFree(buf[i]); }
  return 0;
}

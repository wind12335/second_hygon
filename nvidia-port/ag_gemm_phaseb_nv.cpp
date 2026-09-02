// Phase B (NVIDIA port): cross-substrate AllGather-GEMM benchmark for
// RTX 4090 / sm_89 / 4 GPUs / PCIe.
//
// Faithful port of ag_gemm_phaseb.cpp (K500SM_AI / gfx928): same 10 paths,
// same CLI, same CSV schema (analyze_phaseb.py consumes both unchanged).
// Library mapping: RCCL -> NCCL, DUSHMEM -> NVSHMEM, rocBLAS -> cuBLAS.
// See README_NVIDIA_PORT.md for every API decision and open risk.
//
// One binary, identical GEMM/layout/timing/correctness across all paths:
//   NCCL family:
//     comm  COMM_ONLY          sliced NCCL AllGather only (isolated reference)
//     gemm  GEMM_ONLY          sliced GEMM only (fragmentation reference)
//     r0    R0_FULL_SERIAL     full NCCL AllGather -> full GEMM      (was B0)
//     rs    RS_SLICE_SERIAL    sliced AG(i) -> GEMM(i) -> AG(i+1)    (was B1)
//     r1    R1_EVENT_OVERLAP   sliced AG with event-driven release   (was H0)
//   NVSHMEM family:
//     fc    FC_FCOLLECT_ONLY   full nvshmemx fcollect only (isolated reference)
//     dc    DC_PUSHSIG_ONLY    sliced put+signal only, release waits, no GEMM
//     d0    D0_FCOLLECT_SERIAL full fcollect -> full GEMM
//     ds    DS_PUSHSIG_SERIAL  sliced put+signal -> GEMM(i) -> next slice puts
//     d1    D1_PUSHSIG_OVERLAP sliced put+signal, per-slice ready-wait -> GEMM
//
// Slot reuse across iterations is protected by monotonic epoch + credit
// (remote consumers) plus a per-slice event (self consumption), following the
// Phase A admission protocol.

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <cublas_v2.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t _e = (cmd);                                                    \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_e));                                          \
      MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t _e = (cmd);                                                   \
    if (_e != ncclSuccess) {                                                   \
      fprintf(stderr, "NCCL failure %s:%d: %s\n", __FILE__, __LINE__,          \
              ncclGetErrorString(_e));                                         \
      MPI_Abort(MPI_COMM_WORLD, 3);                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(cmd)                                                      \
  do {                                                                         \
    cublasStatus_t _e = (cmd);                                                 \
    if (_e != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS failure %s:%d: %d\n", __FILE__, __LINE__,        \
              static_cast<int>(_e));                                           \
      MPI_Abort(MPI_COMM_WORLD, 4);                                            \
    }                                                                          \
  } while (0)

namespace {

enum class Path {
  kCommOnly, kGemmOnly, kR0, kRS, kR1, kFcOnly, kDcOnly, kD0, kDS, kD1, kD1W
};
enum class Family { kNccl, kNvshmem };

struct Args {
  Path path = Path::kD1;
  int m_local = 1024;
  int n = 1024;
  int k = 1024;
  int q = 1;
  int warmup = 10;
  int iters = 20;
  int verify_every = 1;
  int window_mult = 1;   // symmetric slots = q * window_mult
  int dush_quiet = 0;    // 1 => nvshmemx_quiet_on_stream after each slice's puts
                         // (flag name kept from the HIP version for runner compat)
  std::string output_dir;
  std::string run_id;
  std::string candidate = "C0_DEFAULT";
};

struct ErrorStats {
  unsigned int max_abs_bits;
  unsigned int max_rel_bits;
  unsigned long long mismatch_count;
};

struct Metrics {
  float release_first_us = 0.0f;
  float release_last_us = 0.0f;
  float done_us = 0.0f;
  float gemm_first_start_us = 0.0f;
  float gemm_last_end_us = 0.0f;
  float e2e_us = 0.0f;
  float gemm_interval_us = 0.0f;
};

struct SliceMetrics {
  float release_us = 0.0f;
  float gemm_start_us = 0.0f;
  float gemm_end_us = 0.0f;
  float gemm_duration_us = 0.0f;
};

struct Measurement {
  Metrics totals;
  std::vector<SliceMetrics> slices;
};

__global__ void fill_input(float* dst, size_t count, int rank) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    // The rank term makes an incorrect AllGather ordering observable.
    dst[i] = 0.005f * static_cast<float>(rank + 1) +
             0.0001f * static_cast<float>(i % 251);
  }
  return;
}

__global__ void fill_weight(float* dst, size_t count) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = 0.0002f * static_cast<float>(static_cast<int>((i * 17) % 127) - 63);
  }
  return;
}

// src is [rank][m_chunk][N]. dst is canonical [rank][m_local][N].
__global__ void scatter_chunk(const float* src, float* dst, int ranks,
                              int m_local, int m_chunk, int n, int chunk) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(ranks) * m_chunk * n;
  if (idx >= total) return;
  int col = static_cast<int>(idx % n);
  size_t row_part = idx / n;
  int row = static_cast<int>(row_part % m_chunk);
  int rank = static_cast<int>(row_part / m_chunk);
  size_t out = (static_cast<size_t>(rank) * m_local + chunk * m_chunk + row) * n + col;
  dst[out] = src[idx];
}

__global__ void compare_output(const float* actual, const float* reference,
                               size_t count, float abs_tol, float rel_tol,
                               ErrorStats* stats) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float a = actual[i];
  float r = reference[i];
  float abs_err = fabsf(a - r);
  float rel_err = abs_err / fmaxf(fabsf(r), 1.0e-7f);
  atomicMax(&stats->max_abs_bits, __float_as_uint(abs_err));
  atomicMax(&stats->max_rel_bits, __float_as_uint(rel_err));
  if (abs_err > abs_tol && rel_err > rel_tol) atomicAdd(&stats->mismatch_count, 1ULL);
}

// Device-side signaling used for the D1 credit return. The HIP version used
// dushmemx_signal_op_on_stream (a stream-ordered host API); NVSHMEM's plain
// host signal is NOT stream-ordered, so we keep the stream ordering by
// launching this kernel on the same stream, right after the consumer GEMM.
__global__ void signal_op_kernel(uint64_t* dest, uint64_t value, int pe) {
  nvshmem_uint64_signal(dest, value, pe);
}

const char* path_name(Path path) {
  switch (path) {
    case Path::kCommOnly: return "COMM_ONLY";
    case Path::kGemmOnly: return "GEMM_ONLY";
    case Path::kR0: return "R0_FULL_SERIAL";
    case Path::kRS: return "RS_SLICE_SERIAL";
    case Path::kR1: return "R1_EVENT_OVERLAP";
    case Path::kFcOnly: return "FC_FCOLLECT_ONLY";
    case Path::kDcOnly: return "DC_PUSHSIG_ONLY";
    case Path::kD0: return "D0_FCOLLECT_SERIAL";
    case Path::kDS: return "DS_PUSHSIG_SERIAL";
    case Path::kD1: return "D1_PUSHSIG_OVERLAP";
    case Path::kD1W: return "D1W_WAITSTREAM_OVERLAP";
  }
  return "UNKNOWN";
}

Family path_family(Path path) {
  switch (path) {
    case Path::kFcOnly:
    case Path::kDcOnly:
    case Path::kD0:
    case Path::kDS:
    case Path::kD1:
    case Path::kD1W:
      return Family::kNvshmem;
    default:
      return Family::kNccl;
  }
}

Path parse_path(const std::string& value) {
  if (value == "comm") return Path::kCommOnly;
  if (value == "gemm") return Path::kGemmOnly;
  if (value == "r0") return Path::kR0;
  if (value == "rs") return Path::kRS;
  if (value == "r1") return Path::kR1;
  if (value == "fc") return Path::kFcOnly;
  if (value == "dc") return Path::kDcOnly;
  if (value == "d0") return Path::kD0;
  if (value == "ds") return Path::kDS;
  if (value == "d1") return Path::kD1;
  if (value == "d1w") return Path::kD1W;
  fprintf(stderr, "Unknown --path value: %s\n", value.c_str());
  std::exit(1);
  return Path::kD1;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (key == "--path") args.path = parse_path(require_value("--path"));
    else if (key == "--m-local") args.m_local = std::atoi(require_value("--m-local"));
    else if (key == "--n") args.n = std::atoi(require_value("--n"));
    else if (key == "--k") args.k = std::atoi(require_value("--k"));
    else if (key == "--q") args.q = std::atoi(require_value("--q"));
    else if (key == "--warmup") args.warmup = std::atoi(require_value("--warmup"));
    else if (key == "--iters") args.iters = std::atoi(require_value("--iters"));
    else if (key == "--verify-every") args.verify_every = std::atoi(require_value("--verify-every"));
    else if (key == "--window-mult") args.window_mult = std::atoi(require_value("--window-mult"));
    else if (key == "--dush-quiet") args.dush_quiet = std::atoi(require_value("--dush-quiet"));
    else if (key == "--output-dir") args.output_dir = require_value("--output-dir");
    else if (key == "--run-id") args.run_id = require_value("--run-id");
    else if (key == "--candidate") args.candidate = require_value("--candidate");
    else if (key == "--help") {
      printf("Usage: %s --path {comm|gemm|r0|rs|r1|fc|dc|d0|ds|d1} --m-local M --n N --k K --q Q "
             "--warmup W --iters I --verify-every V --window-mult WM --dush-quiet Q0 "
             "--output-dir DIR --run-id ID --candidate ID\n", argv[0]);
      std::exit(0);
    } else {
      fprintf(stderr, "Unknown option: %s\n", key.c_str());
      std::exit(1);
    }
  }
  if (args.output_dir.empty() || args.run_id.empty() || args.m_local <= 0 || args.n <= 0 ||
      args.k <= 0 || args.q <= 0 || args.m_local % args.q != 0 || args.warmup < 0 ||
      args.iters <= 0 || args.verify_every <= 0 || args.window_mult < 1 ||
      args.dush_quiet < 0 || args.dush_quiet > 1) {
    fprintf(stderr, "Invalid benchmark arguments. m-local must be divisible by q.\n");
    std::exit(1);
  }
  return args;
}

float elapsed_us(cudaEvent_t from, cudaEvent_t to) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, from, to));
  return ms * 1000.0f;
}

float bits_to_float(unsigned int bits) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits), "unexpected float width");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string csv_escape(const std::string& value) {
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '\"') escaped += '\"';
    escaped += c;
  }
  escaped += "\"";
  return escaped;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  Args args = parse_args(argc, argv);

  const char* local_rank_env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int device = local_rank_env ? std::atoi(local_rank_env) : rank;
  // CUDA/NVSHMEM init is a plain cudaSetDevice; the HIP port needed
  // hipInit/hipDevicePrimaryCtxRetain/hipCtxSetCurrent for DUSHMEM, NVSHMEM
  // sets up its own context (see README_NVIDIA_PORT.md).
  CUDA_CHECK(cudaSetDevice(device));

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device < 0 || device >= device_count) {
    fprintf(stderr, "Rank %d mapped to invalid device %d of %d\n", rank, device, device_count);
    MPI_Abort(MPI_COMM_WORLD, 5);
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int m_chunk = args.m_local / args.q;
  const int global_m = ranks * args.m_local;
  const size_t local_elements = static_cast<size_t>(args.m_local) * args.k;
  const size_t full_a_elements = static_cast<size_t>(global_m) * args.k;
  const size_t full_y_elements = static_cast<size_t>(global_m) * args.n;
  const size_t chunk_a_elements = static_cast<size_t>(ranks) * m_chunk * args.k;
  const size_t chunk_y_elements = static_cast<size_t>(ranks) * m_chunk * args.n;
  const size_t slice_bytes = static_cast<size_t>(m_chunk) * args.k * sizeof(float);
  const size_t local_bytes = local_elements * sizeof(float);
  const int slots = args.q * args.window_mult;

  ncclUniqueId id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, ranks, id, rank));

  nvshmemx_init_attr_t init_attr;
  nvshmemx_init_attr_options_t init_options;
  std::memset(&init_attr, 0, sizeof(init_attr));
  std::memset(&init_options, 0, sizeof(init_options));
  MPI_Comm mpi_world = MPI_COMM_WORLD;
  init_options.mpi_comm = &mpi_world;
  init_attr.options = &init_options;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
  if (nvshmem_my_pe() != rank || nvshmem_n_pes() != ranks) {
    fprintf(stderr, "rank=%d NVSHMEM world mismatch: pe=%d npes=%d\n", rank,
            nvshmem_my_pe(), nvshmem_n_pes());
    MPI_Abort(MPI_COMM_WORLD, 92);
  }

  cudaStream_t comm_stream{};
  cudaStream_t compute_stream{};
  cudaStream_t wait_stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));
  cublasHandle_t blas{};
  CUBLAS_CHECK(cublasCreate(&blas));
  CUBLAS_CHECK(cublasSetStream(blas, compute_stream));

  // Symmetric heap: identical buffers on every PE, so NCCL and NVSHMEM paths
  // feed the exact same GEMM inputs at the exact same addresses.
  float* x_local = static_cast<float*>(nvshmem_malloc(local_bytes));
  float* full_a = static_cast<float*>(nvshmem_malloc(full_a_elements * sizeof(float)));
  const size_t signal_count = static_cast<size_t>(ranks) * slots;
  uint64_t* ready = static_cast<uint64_t*>(nvshmem_malloc(signal_count * sizeof(uint64_t)));
  uint64_t* credit = static_cast<uint64_t*>(nvshmem_malloc(signal_count * sizeof(uint64_t)));
  std::vector<float*> gathered(args.q, nullptr);
  for (int i = 0; i < args.q; ++i) {
    gathered[i] = static_cast<float*>(nvshmem_malloc(chunk_a_elements * sizeof(float)));
  }
  float* weights = nullptr;
  float* reference = nullptr;
  float* output = nullptr;
  CUDA_CHECK(cudaMalloc(&weights, static_cast<size_t>(args.k) * args.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&reference, full_y_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output, full_y_elements * sizeof(float)));
  std::vector<float*> chunk_output(args.q, nullptr);
  for (int i = 0; i < args.q; ++i) {
    CUDA_CHECK(cudaMalloc(&chunk_output[i], chunk_y_elements * sizeof(float)));
  }
  if (x_local == nullptr || full_a == nullptr || ready == nullptr || credit == nullptr ||
      gathered.front() == nullptr) {
    fprintf(stderr, "rank=%d symmetric allocation failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 93);
  }

  auto sig_idx = [&](int peer, int slot) -> size_t {
    return static_cast<size_t>(peer) * slots + slot;
  };

  constexpr int threads = 256;
  fill_input<<<(local_elements + threads - 1) / threads, threads, 0, comm_stream>>>(
      x_local, local_elements, rank);
  fill_weight<<<(static_cast<size_t>(args.k) * args.n + threads - 1) / threads, threads, 0,
                compute_stream>>>(weights, static_cast<size_t>(args.k) * args.n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemsetAsync(full_a, 0, full_a_elements * sizeof(float), comm_stream));
  for (int i = 0; i < args.q; ++i) {
    CUDA_CHECK(cudaMemsetAsync(gathered[i], 0, chunk_a_elements * sizeof(float), comm_stream));
  }
  CUDA_CHECK(cudaMemsetAsync(ready, 0, signal_count * sizeof(uint64_t), comm_stream));
  CUDA_CHECK(cudaMemsetAsync(credit, 0, signal_count * sizeof(uint64_t), comm_stream));
  CUDA_CHECK(cudaStreamSynchronize(comm_stream));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
  nvshmemx_barrier_all_on_stream(comm_stream);
  CUDA_CHECK(cudaStreamSynchronize(comm_stream));

  auto gemm = [&](const float* a, float* c, int rows) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             args.n, rows, args.k, &alpha,
                             weights, args.n, a, args.k, &beta, c, args.n));
  };
  auto scatter = [&](const float* src, int chunk_index) {
    scatter_chunk<<<(chunk_y_elements + threads - 1) / threads, threads, 0, compute_stream>>>(
        src, output, ranks, args.m_local, m_chunk, args.n, chunk_index);
    CUDA_CHECK(cudaGetLastError());
  };

  // Ground-truth reference: trusted full NCCL AllGather + full GEMM.
  NCCL_CHECK(ncclAllGather(x_local, full_a, local_elements, ncclFloat, comm, comm_stream));
  CUDA_CHECK(cudaStreamSynchronize(comm_stream));
  gemm(full_a, reference, global_m);
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));

  // GEMM_ONLY consumes precisely the q gathered buffers that RS/R1/DS/D1 use.
  for (int i = 0; i < args.q; ++i) {
    NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                             gathered[i], static_cast<size_t>(m_chunk) * args.k,
                             ncclFloat, comm, comm_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(comm_stream));
  nvshmemx_barrier_all_on_stream(comm_stream);
  CUDA_CHECK(cudaStreamSynchronize(comm_stream));

  cudaEvent_t issue{};
  cudaEvent_t done{};
  cudaEvent_t end{};
  CUDA_CHECK(cudaEventCreate(&issue));
  CUDA_CHECK(cudaEventCreate(&done));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<cudaEvent_t> release(args.q);
  std::vector<cudaEvent_t> gemm_start(args.q);
  std::vector<cudaEvent_t> gemm_end(args.q);
  for (int i = 0; i < args.q; ++i) {
    CUDA_CHECK(cudaEventCreate(&release[i]));
    CUDA_CHECK(cudaEventCreate(&gemm_start[i]));
    CUDA_CHECK(cudaEventCreate(&gemm_end[i]));
  }

  ErrorStats* device_error = nullptr;
  CUDA_CHECK(cudaMalloc(&device_error, sizeof(ErrorStats)));
  const float abs_tol = 1.0e-2f;
  const float rel_tol = 1.0e-4f;

  long long global_call = 0;  // 1-based epoch counter across warmup + timed runs.

  auto run_once = [&]() -> Measurement {
    ++global_call;
    const long long epoch = global_call;
    const uint64_t epoch_sig = static_cast<uint64_t>(epoch);
    CUDA_CHECK(cudaMemsetAsync(output, 0, full_y_elements * sizeof(float), compute_stream));
    CUDA_CHECK(cudaEventRecord(issue, comm_stream));

    if (args.path == Path::kCommOnly) {
      for (int i = 0; i < args.q; ++i) {
        NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                 gathered[i], static_cast<size_t>(m_chunk) * args.k,
                                 ncclFloat, comm, comm_stream));
        CUDA_CHECK(cudaEventRecord(release[i], comm_stream));
      }
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaEventRecord(end, comm_stream));
    } else if (args.path == Path::kFcOnly) {
      // fcollect into full_a once; release measured on the wait stream.
      if (global_call > 1) CUDA_CHECK(cudaStreamWaitEvent(comm_stream, gemm_end[0], 0));
      nvshmemx_fcollectmem_on_stream(NVSHMEM_TEAM_WORLD, full_a, x_local,
                                     local_bytes, comm_stream);
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaStreamWaitEvent(wait_stream, done, 0));
      CUDA_CHECK(cudaEventRecord(release[0], wait_stream));
      CUDA_CHECK(cudaEventRecord(end, wait_stream));
    } else if (args.path == Path::kDcOnly) {
      for (int i = 0; i < args.q; ++i) {
        const long long gs = (global_call - 1) * args.q + i;
        const int slot = static_cast<int>(gs % slots);
        CUDA_CHECK(cudaMemcpyAsync(gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
                                   x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                   slice_bytes, cudaMemcpyDeviceToDevice, comm_stream));
        for (int peer = 0; peer < ranks; ++peer) {
          if (peer == rank) continue;
          nvshmemx_putmem_signal_on_stream(
              gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
              x_local + static_cast<size_t>(i) * m_chunk * args.k, slice_bytes,
              &ready[sig_idx(rank, slot)], epoch_sig, NVSHMEM_SIGNAL_SET, peer, comm_stream);
        }
        if (args.dush_quiet) nvshmemx_quiet_on_stream(comm_stream);
        for (int producer = 0; producer < ranks; ++producer) {
          if (producer == rank) continue;
          nvshmemx_signal_wait_until_on_stream(&ready[sig_idx(producer, slot)],
                                               NVSHMEM_CMP_GE, epoch_sig, wait_stream);
        }
        CUDA_CHECK(cudaEventRecord(release[i], wait_stream));
      }
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaEventRecord(end, wait_stream));
    } else if (args.path == Path::kGemmOnly) {
      for (int i = 0; i < args.q; ++i) {
        CUDA_CHECK(cudaEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        CUDA_CHECK(cudaEventRecord(gemm_end[i], compute_stream));
      }
      CUDA_CHECK(cudaEventRecord(done, compute_stream));
      CUDA_CHECK(cudaEventRecord(end, compute_stream));
    } else if (args.path == Path::kR0) {
      NCCL_CHECK(ncclAllGather(x_local, full_a, local_elements, ncclFloat, comm, comm_stream));
      CUDA_CHECK(cudaEventRecord(release[0], comm_stream));
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaStreamWaitEvent(compute_stream, done, 0));
      CUDA_CHECK(cudaEventRecord(gemm_start[0], compute_stream));
      gemm(full_a, output, global_m);
      CUDA_CHECK(cudaEventRecord(gemm_end[0], compute_stream));
      CUDA_CHECK(cudaEventRecord(end, compute_stream));
    } else if (args.path == Path::kD0) {
      if (global_call > 1) CUDA_CHECK(cudaStreamWaitEvent(comm_stream, gemm_end[0], 0));
      nvshmemx_fcollectmem_on_stream(NVSHMEM_TEAM_WORLD, full_a, x_local,
                                     local_bytes, comm_stream);
      CUDA_CHECK(cudaEventRecord(release[0], comm_stream));
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaStreamWaitEvent(compute_stream, done, 0));
      CUDA_CHECK(cudaEventRecord(gemm_start[0], compute_stream));
      gemm(full_a, output, global_m);
      CUDA_CHECK(cudaEventRecord(gemm_end[0], compute_stream));
      CUDA_CHECK(cudaEventRecord(end, compute_stream));
    } else if (args.path == Path::kRS || args.path == Path::kR1) {
      const bool serial = args.path == Path::kRS;
      for (int i = 0; i < args.q; ++i) {
        NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                 gathered[i], static_cast<size_t>(m_chunk) * args.k,
                                 ncclFloat, comm, comm_stream));
        CUDA_CHECK(cudaEventRecord(release[i], comm_stream));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, release[i], 0));
        CUDA_CHECK(cudaEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        CUDA_CHECK(cudaEventRecord(gemm_end[i], compute_stream));
        if (serial && i + 1 < args.q) CUDA_CHECK(cudaStreamWaitEvent(comm_stream, gemm_end[i], 0));
      }
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaEventRecord(end, compute_stream));
    } else if (args.path == Path::kDS || args.path == Path::kD1 ||
               args.path == Path::kD1W) {
      const bool serial = args.path == Path::kDS;
      // d1w: place the consumer-side ready-waits and the release marker on the
      // dedicated wait_stream (the placement DC uses) and gate compute_stream
      // onto release[i] with an event — isolating wait placement as the
      // variable. Mirrors the HIP-side D1W exactly, including enqueue order
      // (enqueue order is a measured performance variable: switching it moved
      // d1 by up to 34% on the Hygon side — do NOT restructure this loop).
      cudaStream_t& consumer_wait_stream =
          (args.path == Path::kD1W) ? wait_stream : compute_stream;
      // Single per-slice loop: producer side (credit + self-WAR gated local
      // copy + put+signal) then consumer side (ready-wait -> GEMM -> scatter
      // -> credit) for the same slice. The loops are merged on purpose:
      // cudaStreamWaitEvent snapshots the most recent record of the event at
      // enqueue time, so a serial gate placed in a separate producer loop
      // would capture the PREVIOUS iteration's gemm_end[i]. Merging lets the
      // gate below reference this iteration's gemm_end[i] after it has been
      // enqueued. For D1 (serial=false) the merged loop issues exactly the
      // same per-stream op sequence as the old split loops.
      for (int i = 0; i < args.q; ++i) {
        const long long gs = (global_call - 1) * args.q + i;
        const int slot = static_cast<int>(gs % slots);
        if (gs >= slots) {
          const long long prev_epoch = (gs - slots) / args.q + 1;
          for (int consumer = 0; consumer < ranks; ++consumer) {
            if (consumer == rank) continue;
            nvshmemx_signal_wait_until_on_stream(&credit[sig_idx(consumer, slot)],
                                                 NVSHMEM_CMP_GE,
                                                 static_cast<uint64_t>(prev_epoch),
                                                 comm_stream);
          }
          CUDA_CHECK(cudaStreamWaitEvent(comm_stream, gemm_end[i], 0));
        }
        CUDA_CHECK(cudaMemcpyAsync(gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
                                   x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                   slice_bytes, cudaMemcpyDeviceToDevice, comm_stream));
        for (int peer = 0; peer < ranks; ++peer) {
          if (peer == rank) continue;
          nvshmemx_putmem_signal_on_stream(
              gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
              x_local + static_cast<size_t>(i) * m_chunk * args.k, slice_bytes,
              &ready[sig_idx(rank, slot)], epoch_sig, NVSHMEM_SIGNAL_SET, peer, comm_stream);
        }
        if (args.dush_quiet) nvshmemx_quiet_on_stream(comm_stream);
        for (int producer = 0; producer < ranks; ++producer) {
          if (producer == rank) continue;
          nvshmemx_signal_wait_until_on_stream(&ready[sig_idx(producer, slot)],
                                               NVSHMEM_CMP_GE, epoch_sig,
                                               consumer_wait_stream);
        }
        CUDA_CHECK(cudaEventRecord(release[i], consumer_wait_stream));
        if (args.path == Path::kD1W) {
          CUDA_CHECK(cudaStreamWaitEvent(compute_stream, release[i], 0));
        }
        CUDA_CHECK(cudaEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        CUDA_CHECK(cudaEventRecord(gemm_end[i], compute_stream));
        for (int producer = 0; producer < ranks; ++producer) {
          if (producer == rank) continue;
          // Stream-ordered credit: the HIP version used
          // dushmemx_signal_op_on_stream on compute_stream; here a 1-thread
          // kernel on the same stream preserves the ordering.
          signal_op_kernel<<<1, 1, 0, compute_stream>>>(
              &credit[sig_idx(rank, slot)], epoch_sig, producer);
          CUDA_CHECK(cudaGetLastError());
        }
        // Slice-serial semantics (RS-equivalent): the next slice's puts wait
        // for this slice's GEMM. In D1 this gate is absent and the puts of
        // slice i+1 stream out while GEMM(i) runs — that is the overlap.
        if (serial && i + 1 < args.q) {
          CUDA_CHECK(cudaStreamWaitEvent(comm_stream, gemm_end[i], 0));
        }
      }
      CUDA_CHECK(cudaEventRecord(done, comm_stream));
      CUDA_CHECK(cudaEventRecord(end, compute_stream));
    }
    CUDA_CHECK(cudaEventSynchronize(done));
    CUDA_CHECK(cudaEventSynchronize(end));
    Measurement measurement;
    measurement.slices.resize(args.q);
    Metrics& metrics = measurement.totals;
    const bool single_slice = args.path == Path::kR0 || args.path == Path::kD0 ||
                              args.path == Path::kFcOnly;
    const int last_event = single_slice ? 0 : args.q - 1;
    // Paths without an in-loop GEMM never record gemm_* events; paths without
    // collective traffic never record release events. Query only what exists.
    const bool runs_gemm = args.path != Path::kCommOnly && args.path != Path::kFcOnly &&
                           args.path != Path::kDcOnly;
    const bool runs_release = args.path != Path::kGemmOnly;
    if (runs_release) {
      metrics.release_first_us = elapsed_us(issue, release.front());
      metrics.release_last_us = elapsed_us(issue, release[last_event]);
    }
    metrics.done_us = elapsed_us(issue, done);
    if (runs_gemm) {
      metrics.gemm_first_start_us = elapsed_us(issue, gemm_start.front());
      metrics.gemm_last_end_us = elapsed_us(issue, gemm_end[last_event]);
      metrics.gemm_interval_us = elapsed_us(gemm_start.front(), gemm_end[last_event]);
    }
    metrics.e2e_us = elapsed_us(issue, end);
    for (int i = 0; i < args.q; ++i) {
      const bool has_release = args.path != Path::kGemmOnly && (i == 0 || !single_slice);
      const bool has_gemm = args.path != Path::kCommOnly && args.path != Path::kFcOnly &&
                            args.path != Path::kDcOnly && (i == 0 || !single_slice);
      SliceMetrics& slice = measurement.slices[i];
      if (has_release) slice.release_us = elapsed_us(issue, release[i]);
      if (has_gemm) {
        slice.gemm_start_us = elapsed_us(issue, gemm_start[i]);
        slice.gemm_end_us = elapsed_us(issue, gemm_end[i]);
        slice.gemm_duration_us = elapsed_us(gemm_start[i], gemm_end[i]);
      }
    }
    return measurement;
  };

  for (int i = 0; i < args.warmup; ++i) {
    run_once();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::ostringstream rank_path;
  rank_path << args.output_dir << "/raw_rank" << rank << ".csv";
  std::ofstream rank_csv(rank_path.str());
  rank_csv << "run_id,rank,rank_count,device,device_name,gfx_arch,path,family,candidate,M,N,K,q,"
              "slice_bytes,warmup,window_mult,dush_quiet,iteration_index,t_issue_us,"
              "t_release_first_us,t_release_last_us,t_done_us,gemm_first_start_us,"
              "gemm_last_end_us,gemm_interval_us,e2e_us,gemm_tflops,"
              "correctness,max_abs_error,max_rel_error,mismatch_count,status\n";
  std::ostringstream slice_rank_path;
  slice_rank_path << args.output_dir << "/release_slices_rank" << rank << ".csv";
  std::ofstream slice_rank_csv(slice_rank_path.str());
  slice_rank_csv << "run_id,rank,rank_count,device,gfx_arch,path,family,candidate,M,N,K,q,"
                    "slice_index,slice_bytes,iteration_index,t_issue_us,t_release_us,"
                    "t_gemm_start_us,t_gemm_end_us,t_gemm_duration_us,correctness,status\n";
  std::ofstream global_csv;
  std::ofstream slice_global_csv;
  if (rank == 0) {
    global_csv.open(args.output_dir + "/raw_global_samples.csv");
    global_csv << "run_id,rank_count,path,family,candidate,M,N,K,q,slice_bytes,iteration_index,"
                  "t_release_first_max_us,t_release_last_max_us,t_done_max_us,"
                  "gemm_first_start_max_us,gemm_last_end_max_us,gemm_interval_max_us,e2e_max_us,"
                  "correctness_all_ranks,max_abs_error_max,max_rel_error_max,"
                  "mismatch_count_sum,status\n";
    slice_global_csv.open(args.output_dir + "/release_slices_global.csv");
    slice_global_csv << "run_id,rank_count,path,family,candidate,M,N,K,q,slice_index,slice_bytes,"
                        "iteration_index,t_issue_us,t_release_max_us,t_gemm_start_max_us,"
                        "t_gemm_end_max_us,t_gemm_duration_max_us,correctness_all_ranks,status\n";
  }

  int local_failures = 0;
  for (int iter = 0; iter < args.iters; ++iter) {
    MPI_Barrier(MPI_COMM_WORLD);
    Measurement measurement = run_once();
    Metrics& m = measurement.totals;
    ErrorStats host_error{0, 0, 0};
    bool needs_output_check = args.path != Path::kCommOnly && args.path != Path::kFcOnly &&
                              args.path != Path::kDcOnly && ((iter % args.verify_every) == 0);
    if (needs_output_check) {
      CUDA_CHECK(cudaMemsetAsync(device_error, 0, sizeof(ErrorStats), compute_stream));
      compare_output<<<(full_y_elements + threads - 1) / threads, threads, 0, compute_stream>>>(
          output, reference, full_y_elements, abs_tol, rel_tol, device_error);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(&host_error, device_error, sizeof(ErrorStats),
                                 cudaMemcpyDeviceToHost, compute_stream));
      CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }
    float max_abs = bits_to_float(host_error.max_abs_bits);
    float max_rel = bits_to_float(host_error.max_rel_bits);
    bool pass = !needs_output_check || host_error.mismatch_count == 0;
    if (!pass) ++local_failures;
    const double flops = 2.0 * static_cast<double>(global_m) * args.n * args.k;
    const float denom_us = (args.path == Path::kGemmOnly) ? m.gemm_interval_us :
                           (m.gemm_interval_us > 0.0f ? m.gemm_interval_us : 0.0f);
    const double gemm_tflops = denom_us > 0.0f ? flops / (static_cast<double>(denom_us) * 1.0e6) : 0.0;
    rank_csv << args.run_id << ',' << rank << ',' << ranks << ',' << device << ','
             << csv_escape(prop.name) << ",sm_89," << path_name(args.path) << ','
             << (path_family(args.path) == Family::kNvshmem ? "NVSHMEM" : "NCCL") << ','
             << args.candidate << ',' << args.m_local << ',' << args.n << ',' << args.k << ','
             << args.q << ',' << slice_bytes << ',' << args.warmup << ',' << args.window_mult
             << ',' << args.dush_quiet << ',' << iter << ",0," << std::fixed
             << std::setprecision(3) << m.release_first_us << ',' << m.release_last_us << ','
             << m.done_us << ',' << m.gemm_first_start_us << ',' << m.gemm_last_end_us << ','
             << m.gemm_interval_us << ',' << m.e2e_us << ',' << gemm_tflops << ','
             << (pass ? "PASS" : "FAIL") << ',' << max_abs << ',' << max_rel << ','
             << host_error.mismatch_count << ',' << (pass ? "0" : "1") << '\n';
    rank_csv.flush();

    float local_values[7] = {m.release_first_us, m.release_last_us, m.done_us,
                             m.gemm_first_start_us, m.gemm_last_end_us, m.gemm_interval_us,
                             m.e2e_us};
    float reduced_values[7] = {};
    MPI_Reduce(local_values, reduced_values, 7, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    int local_pass = pass ? 1 : 0;
    int all_pass = 0;
    MPI_Reduce(&local_pass, &all_pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    float error_values[2] = {max_abs, max_rel};
    float max_errors[2] = {};
    MPI_Reduce(error_values, max_errors, 2, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    unsigned long long mismatch_sum = 0;
    MPI_Reduce(&host_error.mismatch_count, &mismatch_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      global_csv << args.run_id << ',' << ranks << ',' << path_name(args.path) << ','
                 << (path_family(args.path) == Family::kNvshmem ? "NVSHMEM" : "NCCL") << ','
                 << args.candidate << ',' << args.m_local << ',' << args.n << ',' << args.k << ','
                 << args.q << ',' << slice_bytes << ',' << iter << ',' << std::fixed
                 << std::setprecision(3) << reduced_values[0] << ',' << reduced_values[1] << ','
                 << reduced_values[2] << ',' << reduced_values[3] << ',' << reduced_values[4]
                 << ',' << reduced_values[5] << ',' << reduced_values[6] << ','
                 << (all_pass ? "PASS" : "FAIL") << ',' << max_errors[0] << ','
                 << max_errors[1] << ',' << mismatch_sum << ',' << (all_pass ? "0" : "1") << '\n';
      global_csv.flush();
    }
    for (int slice_index = 0; slice_index < args.q; ++slice_index) {
      const SliceMetrics& slice = measurement.slices[slice_index];
      slice_rank_csv << args.run_id << ',' << rank << ',' << ranks << ',' << device << ",sm_89,"
                     << path_name(args.path) << ','
                     << (path_family(args.path) == Family::kNvshmem ? "NVSHMEM" : "NCCL") << ','
                     << args.candidate << ',' << args.m_local << ',' << args.n << ',' << args.k
                     << ',' << args.q << ',' << slice_index << ',' << slice_bytes << ',' << iter
                     << ",0," << std::fixed << std::setprecision(3) << slice.release_us << ','
                     << slice.gemm_start_us << ',' << slice.gemm_end_us << ','
                     << slice.gemm_duration_us << ',' << (pass ? "PASS" : "FAIL") << ','
                     << (pass ? "0" : "1") << '\n';
      float slice_values[4] = {slice.release_us, slice.gemm_start_us,
                               slice.gemm_end_us, slice.gemm_duration_us};
      float slice_max[4] = {};
      MPI_Reduce(slice_values, slice_max, 4, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        slice_global_csv << args.run_id << ',' << ranks << ',' << path_name(args.path) << ','
                         << (path_family(args.path) == Family::kNvshmem ? "NVSHMEM" : "NCCL")
                         << ',' << args.candidate << ',' << args.m_local << ',' << args.n << ','
                         << args.k << ',' << args.q << ',' << slice_index << ',' << slice_bytes
                         << ',' << iter << ",0," << std::fixed << std::setprecision(3)
                         << slice_max[0] << ',' << slice_max[1] << ',' << slice_max[2] << ','
                         << slice_max[3] << ',' << (all_pass ? "PASS" : "FAIL") << ','
                         << (all_pass ? "0" : "1") << '\n';
        slice_global_csv.flush();
      }
    }
    slice_rank_csv.flush();
  }

  int global_failures = 0;
  MPI_Allreduce(&local_failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::ofstream manifest(args.output_dir + "/manifest.csv");
    manifest << "run_id,platform_id,device_model,gfx_arch,rank_count,path,family,candidate,M,N,K,q,"
                "slice_bytes,warmup,iters,verify_every,window_mult,dush_quiet,"
                "abs_tolerance,rel_tolerance,requested_nccl_algo,requested_nccl_proto,"
                "requested_nccl_min_channels,requested_nccl_max_channels,status,"
                "rank_failure_count\n";
    const char* algo = std::getenv("NCCL_ALGO");
    const char* proto = std::getenv("NCCL_PROTO");
    const char* min_ch = std::getenv("NCCL_MIN_NCHANNELS");
    const char* max_ch = std::getenv("NCCL_MAX_NCHANNELS");
    manifest << args.run_id << ",RTX4090," << csv_escape(prop.name) << ",sm_89," << ranks << ','
             << path_name(args.path) << ','
             << (path_family(args.path) == Family::kNvshmem ? "NVSHMEM" : "NCCL") << ','
             << args.candidate << ',' << args.m_local << ',' << args.n << ',' << args.k << ','
             << args.q << ',' << slice_bytes << ',' << args.warmup << ',' << args.iters << ','
             << args.verify_every << ',' << args.window_mult << ',' << args.dush_quiet << ','
             << abs_tol << ',' << rel_tol << ',' << (algo ? algo : "DEFAULT") << ','
             << (proto ? proto : "DEFAULT") << ',' << (min_ch ? min_ch : "DEFAULT") << ','
             << (max_ch ? max_ch : "DEFAULT") << ','
             << (global_failures == 0 ? "PASS" : "FAIL") << ',' << global_failures << '\n';
    printf("RESULT run_id=%s path=%s family=%s candidate=%s "
           "shape=[P=%d,m_local=%d,N=%d,K=%d,q=%d] status=%s failures=%d\n",
           args.run_id.c_str(), path_name(args.path),
           path_family(args.path) == Family::kNvshmem ? "NVSHMEM" : "NCCL",
           args.candidate.c_str(), ranks, args.m_local, args.n, args.k, args.q,
           global_failures == 0 ? "PASS" : "FAIL", global_failures);
  }

  CUDA_CHECK(cudaFree(device_error));
  for (int i = 0; i < args.q; ++i) {
    CUDA_CHECK(cudaEventDestroy(release[i]));
    CUDA_CHECK(cudaEventDestroy(gemm_start[i]));
    CUDA_CHECK(cudaEventDestroy(gemm_end[i]));
    nvshmem_free(gathered[i]);
    CUDA_CHECK(cudaFree(chunk_output[i]));
  }
  CUDA_CHECK(cudaEventDestroy(issue));
  CUDA_CHECK(cudaEventDestroy(done));
  CUDA_CHECK(cudaEventDestroy(end));
  nvshmem_free(x_local);
  nvshmem_free(full_a);
  nvshmem_free(ready);
  nvshmem_free(credit);
  CUDA_CHECK(cudaFree(weights));
  CUDA_CHECK(cudaFree(reference));
  CUDA_CHECK(cudaFree(output));
  CUBLAS_CHECK(cublasDestroy(blas));
  CUDA_CHECK(cudaStreamDestroy(comm_stream));
  CUDA_CHECK(cudaStreamDestroy(compute_stream));
  CUDA_CHECK(cudaStreamDestroy(wait_stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  nvshmem_finalize();
  MPI_Finalize();
  return global_failures == 0 ? 0 : 10;
}

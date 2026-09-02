// Phase B: cross-substrate AllGather-GEMM benchmark for K500SM_AI/gfx928.
//
// One binary, identical GEMM/layout/timing/correctness across all paths:
//   RCCL family:
//     comm  COMM_ONLY          sliced RCCL AllGather only (isolated reference)
//     gemm  GEMM_ONLY          sliced GEMM only (fragmentation reference)
//     r0    R0_FULL_SERIAL     full RCCL AllGather -> full GEMM      (was B0)
//     rs    RS_SLICE_SERIAL    sliced AG(i) -> GEMM(i) -> AG(i+1)    (was B1)
//     r1    R1_EVENT_OVERLAP   sliced AG with event-driven release   (was H0)
//   DUSHMEM family:
//     fc    FC_FCOLLECT_ONLY   full dushmemx fcollect only (isolated reference)
//     dc    DC_PUSHSIG_ONLY    sliced put+signal only, release waits, no GEMM
//     d0    D0_FCOLLECT_SERIAL full fcollect -> full GEMM
//     ds    DS_PUSHSIG_SERIAL  sliced put+signal -> GEMM(i) -> next slice puts
//     d1    D1_PUSHSIG_OVERLAP sliced put+signal, per-slice ready-wait -> GEMM
//
// Slot reuse across iterations is protected by monotonic epoch + credit
// (remote consumers) plus a per-slice event (self consumption), following the
// Phase A admission protocol.

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>
#include <rocblas/rocblas.h>
#include <dushmem.h>
#include <dushmemx.h>

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

#define HIP_CHECK(cmd)                                                        \
  do {                                                                        \
    hipError_t _e = (cmd);                                                    \
    if (_e != hipSuccess) {                                                   \
      fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(_e));                                         \
      MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                         \
  } while (0)

#define NCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t _e = (cmd);                                                  \
    if (_e != ncclSuccess) {                                                  \
      fprintf(stderr, "RCCL failure %s:%d: %s\n", __FILE__, __LINE__,         \
              ncclGetErrorString(_e));                                        \
      MPI_Abort(MPI_COMM_WORLD, 3);                                           \
    }                                                                         \
  } while (0)

#define ROCBLAS_CHECK(cmd)                                                    \
  do {                                                                        \
    rocblas_status _e = (cmd);                                                \
    if (_e != rocblas_status_success) {                                       \
      fprintf(stderr, "rocBLAS failure %s:%d: %d\n", __FILE__, __LINE__,      \
              static_cast<int>(_e));                                          \
      MPI_Abort(MPI_COMM_WORLD, 4);                                           \
    }                                                                         \
  } while (0)

namespace {

enum class Path {
  kCommOnly, kGemmOnly, kR0, kRS, kR1, kFcOnly, kDcOnly, kD0, kDS, kD1, kD1W
};
enum class Family { kRccl, kDushmem };

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
  int dush_quiet = 0;    // 1 => dushmemx_quiet_on_stream after each slice's puts
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
      return Family::kDushmem;
    default:
      return Family::kRccl;
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

float elapsed_us(hipEvent_t from, hipEvent_t to) {
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, from, to));
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
  HIP_CHECK(hipSetDevice(device));
  // DUSHMEM init requires an active primary context on this HIP stack.
  HIP_CHECK(hipInit(0));
  hipDevice_t hip_device;
  HIP_CHECK(hipDeviceGet(&hip_device, device));
  hipCtx_t primary_context = nullptr;
  HIP_CHECK(hipDevicePrimaryCtxRetain(&primary_context, hip_device));
  HIP_CHECK(hipCtxSetCurrent(primary_context));

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device < 0 || device >= device_count) {
    fprintf(stderr, "Rank %d mapped to invalid device %d of %d\n", rank, device, device_count);
    MPI_Abort(MPI_COMM_WORLD, 5);
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
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

  dushmemx_init_attr_t init_attr = DUSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm mpi_world = MPI_COMM_WORLD;
  init_attr.mpi_comm = &mpi_world;
  const int init_status = dushmemx_init_attr(DUSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
  if (init_status != 0) {
    fprintf(stderr, "rank=%d dushmemx_init_attr returned %d\n", rank, init_status);
    MPI_Abort(MPI_COMM_WORLD, 91);
  }
  if (dushmem_my_pe() != rank || dushmem_n_pes() != ranks) {
    fprintf(stderr, "rank=%d DUSHMEM world mismatch: pe=%d npes=%d\n", rank,
            dushmem_my_pe(), dushmem_n_pes());
    MPI_Abort(MPI_COMM_WORLD, 92);
  }

  hipStream_t comm_stream{};
  hipStream_t compute_stream{};
  hipStream_t wait_stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&comm_stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&wait_stream, hipStreamNonBlocking));
  rocblas_handle blas{};
  ROCBLAS_CHECK(rocblas_create_handle(&blas));
  ROCBLAS_CHECK(rocblas_set_stream(blas, compute_stream));

  // Symmetric heap: identical buffers on every PE, so RCCL and DUSHMEM paths
  // feed the exact same GEMM inputs at the exact same addresses.
  float* x_local = static_cast<float*>(dushmem_malloc(local_bytes));
  float* full_a = static_cast<float*>(dushmem_malloc(full_a_elements * sizeof(float)));
  const size_t signal_count = static_cast<size_t>(ranks) * slots;
  uint64_t* ready = static_cast<uint64_t*>(dushmem_malloc(signal_count * sizeof(uint64_t)));
  uint64_t* credit = static_cast<uint64_t*>(dushmem_malloc(signal_count * sizeof(uint64_t)));
  std::vector<float*> gathered(args.q, nullptr);
  for (int i = 0; i < args.q; ++i) {
    gathered[i] = static_cast<float*>(dushmem_malloc(chunk_a_elements * sizeof(float)));
  }
  float* weights = nullptr;
  float* reference = nullptr;
  float* output = nullptr;
  HIP_CHECK(hipMalloc(&weights, static_cast<size_t>(args.k) * args.n * sizeof(float)));
  HIP_CHECK(hipMalloc(&reference, full_y_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&output, full_y_elements * sizeof(float)));
  std::vector<float*> chunk_output(args.q, nullptr);
  for (int i = 0; i < args.q; ++i) {
    HIP_CHECK(hipMalloc(&chunk_output[i], chunk_y_elements * sizeof(float)));
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
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemsetAsync(full_a, 0, full_a_elements * sizeof(float), comm_stream));
  for (int i = 0; i < args.q; ++i) {
    HIP_CHECK(hipMemsetAsync(gathered[i], 0, chunk_a_elements * sizeof(float), comm_stream));
  }
  HIP_CHECK(hipMemsetAsync(ready, 0, signal_count * sizeof(uint64_t), comm_stream));
  HIP_CHECK(hipMemsetAsync(credit, 0, signal_count * sizeof(uint64_t), comm_stream));
  HIP_CHECK(hipStreamSynchronize(comm_stream));
  HIP_CHECK(hipStreamSynchronize(compute_stream));
  dushmemx_barrier_all_on_stream(comm_stream);
  HIP_CHECK(hipStreamSynchronize(comm_stream));

  auto gemm = [&](const float* a, float* c, int rows) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(blas, rocblas_operation_none, rocblas_operation_none,
                                args.n, rows, args.k, &alpha,
                                weights, args.n, a, args.k, &beta, c, args.n));
  };
  auto scatter = [&](const float* src, int chunk_index) {
    scatter_chunk<<<(chunk_y_elements + threads - 1) / threads, threads, 0, compute_stream>>>(
        src, output, ranks, args.m_local, m_chunk, args.n, chunk_index);
    HIP_CHECK(hipGetLastError());
  };

  // Ground-truth reference: trusted full RCCL AllGather + full GEMM.
  NCCL_CHECK(ncclAllGather(x_local, full_a, local_elements, ncclFloat, comm, comm_stream));
  HIP_CHECK(hipStreamSynchronize(comm_stream));
  gemm(full_a, reference, global_m);
  HIP_CHECK(hipStreamSynchronize(compute_stream));

  // GEMM_ONLY consumes precisely the q gathered buffers that RS/R1/DS/D1 use.
  for (int i = 0; i < args.q; ++i) {
    NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                             gathered[i], static_cast<size_t>(m_chunk) * args.k,
                             ncclFloat, comm, comm_stream));
  }
  HIP_CHECK(hipStreamSynchronize(comm_stream));
  dushmemx_barrier_all_on_stream(comm_stream);
  HIP_CHECK(hipStreamSynchronize(comm_stream));

  hipEvent_t issue{};
  hipEvent_t done{};
  hipEvent_t end{};
  HIP_CHECK(hipEventCreate(&issue));
  HIP_CHECK(hipEventCreate(&done));
  HIP_CHECK(hipEventCreate(&end));
  std::vector<hipEvent_t> release(args.q);
  std::vector<hipEvent_t> gemm_start(args.q);
  std::vector<hipEvent_t> gemm_end(args.q);
  for (int i = 0; i < args.q; ++i) {
    HIP_CHECK(hipEventCreate(&release[i]));
    HIP_CHECK(hipEventCreate(&gemm_start[i]));
    HIP_CHECK(hipEventCreate(&gemm_end[i]));
  }

  ErrorStats* device_error = nullptr;
  HIP_CHECK(hipMalloc(&device_error, sizeof(ErrorStats)));
  const float abs_tol = 1.0e-2f;
  const float rel_tol = 1.0e-4f;

  long long global_call = 0;  // 1-based epoch counter across warmup + timed runs.

  auto run_once = [&]() -> Measurement {
    ++global_call;
    const long long epoch = global_call;
    HIP_CHECK(hipMemsetAsync(output, 0, full_y_elements * sizeof(float), compute_stream));
    HIP_CHECK(hipEventRecord(issue, comm_stream));

    if (args.path == Path::kCommOnly) {
      for (int i = 0; i < args.q; ++i) {
        NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                 gathered[i], static_cast<size_t>(m_chunk) * args.k,
                                 ncclFloat, comm, comm_stream));
        HIP_CHECK(hipEventRecord(release[i], comm_stream));
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipEventRecord(end, comm_stream));
    } else if (args.path == Path::kFcOnly) {
      // fcollect into full_a once; release measured on the wait stream.
      if (global_call > 1) HIP_CHECK(hipStreamWaitEvent(comm_stream, gemm_end[0], 0));
      const int fcret = dushmemx_fcollectmem_on_stream(DUSHMEM_TEAM_WORLD, full_a, x_local,
                                                       local_bytes, comm_stream);
      if (fcret != 0) {
        fprintf(stderr, "rank=%d fcollectmem returned %d\n", rank, fcret);
        MPI_Abort(MPI_COMM_WORLD, 94);
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipStreamWaitEvent(wait_stream, done, 0));
      HIP_CHECK(hipEventRecord(release[0], wait_stream));
      HIP_CHECK(hipEventRecord(end, wait_stream));
    } else if (args.path == Path::kDcOnly) {
      for (int i = 0; i < args.q; ++i) {
        const long long gs = (global_call - 1) * args.q + i;
        const int slot = static_cast<int>(gs % slots);
        HIP_CHECK(hipMemcpyAsync(gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
                                 x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                 slice_bytes, hipMemcpyDeviceToDevice, comm_stream));
        for (int peer = 0; peer < ranks; ++peer) {
          if (peer == rank) continue;
          dushmemx_putmem_signal_on_stream(
              gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
              x_local + static_cast<size_t>(i) * m_chunk * args.k, slice_bytes,
              &ready[sig_idx(rank, slot)], epoch, DUSHMEM_SIGNAL_SET, peer, comm_stream);
        }
        if (args.dush_quiet) dushmemx_quiet_on_stream(comm_stream);
        for (int producer = 0; producer < ranks; ++producer) {
          if (producer == rank) continue;
          dushmemx_signal_wait_until_on_stream(&ready[sig_idx(producer, slot)],
                                               DUSHMEM_CMP_GE, epoch, wait_stream);
        }
        HIP_CHECK(hipEventRecord(release[i], wait_stream));
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipEventRecord(end, wait_stream));
    } else if (args.path == Path::kGemmOnly) {
      for (int i = 0; i < args.q; ++i) {
        HIP_CHECK(hipEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        HIP_CHECK(hipEventRecord(gemm_end[i], compute_stream));
      }
      HIP_CHECK(hipEventRecord(done, compute_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else if (args.path == Path::kR0) {
      NCCL_CHECK(ncclAllGather(x_local, full_a, local_elements, ncclFloat, comm, comm_stream));
      HIP_CHECK(hipEventRecord(release[0], comm_stream));
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipStreamWaitEvent(compute_stream, done, 0));
      HIP_CHECK(hipEventRecord(gemm_start[0], compute_stream));
      gemm(full_a, output, global_m);
      HIP_CHECK(hipEventRecord(gemm_end[0], compute_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else if (args.path == Path::kD0) {
      if (global_call > 1) HIP_CHECK(hipStreamWaitEvent(comm_stream, gemm_end[0], 0));
      const int fcret = dushmemx_fcollectmem_on_stream(DUSHMEM_TEAM_WORLD, full_a, x_local,
                                                       local_bytes, comm_stream);
      if (fcret != 0) {
        fprintf(stderr, "rank=%d fcollectmem returned %d\n", rank, fcret);
        MPI_Abort(MPI_COMM_WORLD, 94);
      }
      HIP_CHECK(hipEventRecord(release[0], comm_stream));
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipStreamWaitEvent(compute_stream, done, 0));
      HIP_CHECK(hipEventRecord(gemm_start[0], compute_stream));
      gemm(full_a, output, global_m);
      HIP_CHECK(hipEventRecord(gemm_end[0], compute_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else if (args.path == Path::kRS || args.path == Path::kR1) {
      const bool serial = args.path == Path::kRS;
      for (int i = 0; i < args.q; ++i) {
        NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                 gathered[i], static_cast<size_t>(m_chunk) * args.k,
                                 ncclFloat, comm, comm_stream));
        HIP_CHECK(hipEventRecord(release[i], comm_stream));
        HIP_CHECK(hipStreamWaitEvent(compute_stream, release[i], 0));
        HIP_CHECK(hipEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        HIP_CHECK(hipEventRecord(gemm_end[i], compute_stream));
        if (serial && i + 1 < args.q) HIP_CHECK(hipStreamWaitEvent(comm_stream, gemm_end[i], 0));
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else if (args.path == Path::kDS || args.path == Path::kD1 || args.path == Path::kD1W) {
      const bool serial = args.path == Path::kDS;
      // d1w: place the consumer-side ready-waits and the release marker on the
      // dedicated wait_stream (the placement DC uses) and gate compute_stream
      // onto release[i] with an event. Motivation (formal N512/q8): with waits
      // on compute_stream (d1) the first release defers to 14443us — after ALL
      // transports complete (DC last = 11382) — and slices then pace at GEMM
      // cadence, i.e. no true overlap. DC's wait_stream placement resolves its
      // first release at 1373us. d1w isolates wait placement as the variable.
      hipStream_t& consumer_wait_stream = (args.path == Path::kD1W) ? wait_stream : compute_stream;
      // Single per-slice loop: producer side (credit + self-WAR gated local
      // copy + put+signal) then consumer side (ready-wait -> GEMM -> scatter
      // -> credit) for the same slice. The loops are merged on purpose:
      // hipStreamWaitEvent snapshots the most recent record of the event at
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
            dushmemx_signal_wait_until_on_stream(&credit[sig_idx(consumer, slot)],
                                                 DUSHMEM_CMP_GE, prev_epoch, comm_stream);
          }
          HIP_CHECK(hipStreamWaitEvent(comm_stream, gemm_end[i], 0));
        }
        HIP_CHECK(hipMemcpyAsync(gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
                                 x_local + static_cast<size_t>(i) * m_chunk * args.k,
                                 slice_bytes, hipMemcpyDeviceToDevice, comm_stream));
        for (int peer = 0; peer < ranks; ++peer) {
          if (peer == rank) continue;
          dushmemx_putmem_signal_on_stream(
              gathered[i] + static_cast<size_t>(rank) * m_chunk * args.k,
              x_local + static_cast<size_t>(i) * m_chunk * args.k, slice_bytes,
              &ready[sig_idx(rank, slot)], epoch, DUSHMEM_SIGNAL_SET, peer, comm_stream);
        }
        if (args.dush_quiet) dushmemx_quiet_on_stream(comm_stream);
        for (int producer = 0; producer < ranks; ++producer) {
          if (producer == rank) continue;
          dushmemx_signal_wait_until_on_stream(&ready[sig_idx(producer, slot)],
                                               DUSHMEM_CMP_GE, epoch, consumer_wait_stream);
        }
        HIP_CHECK(hipEventRecord(release[i], consumer_wait_stream));
        if (args.path == Path::kD1W) {
          HIP_CHECK(hipStreamWaitEvent(compute_stream, release[i], 0));
        }
        HIP_CHECK(hipEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        HIP_CHECK(hipEventRecord(gemm_end[i], compute_stream));
        for (int producer = 0; producer < ranks; ++producer) {
          if (producer == rank) continue;
          dushmemx_signal_op_on_stream(&credit[sig_idx(rank, slot)], epoch,
                                       DUSHMEM_SIGNAL_SET, producer, compute_stream);
        }
        // Slice-serial semantics (RS-equivalent): the next slice's puts wait
        // for this slice's GEMM. In D1 this gate is absent and the puts of
        // slice i+1 stream out while GEMM(i) runs — that is the overlap.
        if (serial && i + 1 < args.q) {
          HIP_CHECK(hipStreamWaitEvent(comm_stream, gemm_end[i], 0));
        }
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    }
    HIP_CHECK(hipEventSynchronize(done));
    HIP_CHECK(hipEventSynchronize(end));
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
      HIP_CHECK(hipMemsetAsync(device_error, 0, sizeof(ErrorStats), compute_stream));
      compare_output<<<(full_y_elements + threads - 1) / threads, threads, 0, compute_stream>>>(
          output, reference, full_y_elements, abs_tol, rel_tol, device_error);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpyAsync(&host_error, device_error, sizeof(ErrorStats),
                               hipMemcpyDeviceToHost, compute_stream));
      HIP_CHECK(hipStreamSynchronize(compute_stream));
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
             << csv_escape(prop.name) << ",gfx928," << path_name(args.path) << ','
             << (path_family(args.path) == Family::kDushmem ? "DUSHMEM" : "RCCL") << ','
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
                 << (path_family(args.path) == Family::kDushmem ? "DUSHMEM" : "RCCL") << ','
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
      slice_rank_csv << args.run_id << ',' << rank << ',' << ranks << ',' << device << ",gfx928,"
                     << path_name(args.path) << ','
                     << (path_family(args.path) == Family::kDushmem ? "DUSHMEM" : "RCCL") << ','
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
                         << (path_family(args.path) == Family::kDushmem ? "DUSHMEM" : "RCCL")
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
    manifest << args.run_id << ",K500SM_AI," << csv_escape(prop.name) << ",gfx928," << ranks << ','
             << path_name(args.path) << ','
             << (path_family(args.path) == Family::kDushmem ? "DUSHMEM" : "RCCL") << ','
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
           path_family(args.path) == Family::kDushmem ? "DUSHMEM" : "RCCL",
           args.candidate.c_str(), ranks, args.m_local, args.n, args.k, args.q,
           global_failures == 0 ? "PASS" : "FAIL", global_failures);
  }

  HIP_CHECK(hipFree(device_error));
  for (int i = 0; i < args.q; ++i) {
    HIP_CHECK(hipEventDestroy(release[i]));
    HIP_CHECK(hipEventDestroy(gemm_start[i]));
    HIP_CHECK(hipEventDestroy(gemm_end[i]));
    dushmem_free(gathered[i]);
    HIP_CHECK(hipFree(chunk_output[i]));
  }
  HIP_CHECK(hipEventDestroy(issue));
  HIP_CHECK(hipEventDestroy(done));
  HIP_CHECK(hipEventDestroy(end));
  dushmem_free(x_local);
  dushmem_free(full_a);
  dushmem_free(ready);
  dushmem_free(credit);
  HIP_CHECK(hipFree(weights));
  HIP_CHECK(hipFree(reference));
  HIP_CHECK(hipFree(output));
  ROCBLAS_CHECK(rocblas_destroy_handle(blas));
  HIP_CHECK(hipStreamDestroy(comm_stream));
  HIP_CHECK(hipStreamDestroy(compute_stream));
  HIP_CHECK(hipStreamDestroy(wait_stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  dushmem_finalize();
  MPI_Finalize();
  return global_failures == 0 ? 0 : 10;
}

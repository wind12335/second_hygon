// Correctness-first RCCL AllGather-GEMM release benchmark for K500SM_AI/gfx928.
// Paths: COMM_ONLY, GEMM_ONLY, B0_FULL_SERIAL, B1_SLICE_SERIAL, H0_EVENT_OVERLAP.

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>
#include <rocblas/rocblas.h>

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
      fprintf(stderr, "HIP failure %s:%d: %s\n", __FILE__, __LINE__,       \
              hipGetErrorString(_e));                                        \
      MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                         \
  } while (0)

#define NCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t _e = (cmd);                                                  \
    if (_e != ncclSuccess) {                                                  \
      fprintf(stderr, "RCCL failure %s:%d: %s\n", __FILE__, __LINE__,      \
              ncclGetErrorString(_e));                                       \
      MPI_Abort(MPI_COMM_WORLD, 3);                                           \
    }                                                                         \
  } while (0)

#define ROCBLAS_CHECK(cmd)                                                    \
  do {                                                                        \
    rocblas_status _e = (cmd);                                               \
    if (_e != rocblas_status_success) {                                      \
      fprintf(stderr, "rocBLAS failure %s:%d: %d\n", __FILE__, __LINE__,  \
              static_cast<int>(_e));                                         \
      MPI_Abort(MPI_COMM_WORLD, 4);                                           \
    }                                                                         \
  } while (0)

namespace {

enum class Path { kCommOnly, kGemmOnly, kB0, kB1, kH0 };

struct Args {
  Path path = Path::kH0;
  int m_local = 1024;
  int n = 1024;
  int k = 1024;
  int q = 1;
  int warmup = 10;
  int iters = 20;
  int verify_every = 1;
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
    case Path::kB0: return "B0_FULL_SERIAL";
    case Path::kB1: return "B1_SLICE_SERIAL";
    case Path::kH0: return "H0_EVENT_OVERLAP";
  }
  return "UNKNOWN";
}

Path parse_path(const std::string& value) {
  if (value == "comm") return Path::kCommOnly;
  if (value == "gemm") return Path::kGemmOnly;
  if (value == "b0") return Path::kB0;
  if (value == "b1") return Path::kB1;
  if (value == "h0") return Path::kH0;
  fprintf(stderr, "Unknown --path value: %s\n", value.c_str());
  std::exit(1);
  return Path::kH0;
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
    else if (key == "--output-dir") args.output_dir = require_value("--output-dir");
    else if (key == "--run-id") args.run_id = require_value("--run-id");
    else if (key == "--candidate") args.candidate = require_value("--candidate");
    else if (key == "--help") {
      printf("Usage: %s --path {comm|gemm|b0|b1|h0} --m-local M --n N --k K --q Q "
             "--warmup W --iters I --verify-every V --output-dir DIR --run-id ID --candidate ID\n", argv[0]);
      std::exit(0);
    } else {
      fprintf(stderr, "Unknown option: %s\n", key.c_str());
      std::exit(1);
    }
  }
  if (args.output_dir.empty() || args.run_id.empty() || args.m_local <= 0 || args.n <= 0 ||
      args.k <= 0 || args.q <= 0 || args.m_local % args.q != 0 || args.warmup < 0 ||
      args.iters <= 0 || args.verify_every <= 0) {
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

  ncclUniqueId id{};
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, ranks, id, rank));

  hipStream_t comm_stream{};
  hipStream_t compute_stream{};
  HIP_CHECK(hipStreamCreateWithFlags(&comm_stream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking));
  rocblas_handle blas{};
  ROCBLAS_CHECK(rocblas_create_handle(&blas));
  ROCBLAS_CHECK(rocblas_set_stream(blas, compute_stream));

  float* x_local = nullptr;
  float* full_a = nullptr;
  float* weights = nullptr;
  float* reference = nullptr;
  float* output = nullptr;
  HIP_CHECK(hipMalloc(&x_local, local_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&full_a, full_a_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&weights, static_cast<size_t>(args.k) * args.n * sizeof(float)));
  HIP_CHECK(hipMalloc(&reference, full_y_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&output, full_y_elements * sizeof(float)));
  std::vector<float*> gathered(args.q, nullptr);
  std::vector<float*> chunk_output(args.q, nullptr);
  for (int i = 0; i < args.q; ++i) {
    HIP_CHECK(hipMalloc(&gathered[i], chunk_a_elements * sizeof(float)));
    HIP_CHECK(hipMalloc(&chunk_output[i], chunk_y_elements * sizeof(float)));
  }

  constexpr int threads = 256;
  fill_input<<<(local_elements + threads - 1) / threads, threads, 0, comm_stream>>>(x_local, local_elements, rank);
  fill_weight<<<(static_cast<size_t>(args.k) * args.n + threads - 1) / threads, threads, 0, compute_stream>>>(
      weights, static_cast<size_t>(args.k) * args.n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(comm_stream));
  HIP_CHECK(hipStreamSynchronize(compute_stream));

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

  // Build a per-process B0 reference with the exact same input, dtype, GEMM and output layout.
  NCCL_CHECK(ncclAllGather(x_local, full_a, local_elements, ncclFloat, comm, comm_stream));
  HIP_CHECK(hipStreamSynchronize(comm_stream));
  gemm(full_a, reference, global_m);
  HIP_CHECK(hipStreamSynchronize(compute_stream));

  // GEMM_ONLY uses precisely the same q gathered buffers and scatter as B1/H0.
  for (int i = 0; i < args.q; ++i) {
    NCCL_CHECK(ncclAllGather(x_local + static_cast<size_t>(i) * m_chunk * args.k,
                             gathered[i], static_cast<size_t>(m_chunk) * args.k,
                             ncclFloat, comm, comm_stream));
  }
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

  auto run_once = [&]() -> Metrics {
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
    } else if (args.path == Path::kGemmOnly) {
      for (int i = 0; i < args.q; ++i) {
        HIP_CHECK(hipEventRecord(gemm_start[i], compute_stream));
        gemm(gathered[i], chunk_output[i], ranks * m_chunk);
        scatter(chunk_output[i], i);
        HIP_CHECK(hipEventRecord(gemm_end[i], compute_stream));
      }
      HIP_CHECK(hipEventRecord(done, compute_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else if (args.path == Path::kB0) {
      NCCL_CHECK(ncclAllGather(x_local, full_a, local_elements, ncclFloat, comm, comm_stream));
      HIP_CHECK(hipEventRecord(release[0], comm_stream));
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipStreamWaitEvent(compute_stream, done, 0));
      HIP_CHECK(hipEventRecord(gemm_start[0], compute_stream));
      gemm(full_a, output, global_m);
      HIP_CHECK(hipEventRecord(gemm_end[0], compute_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else if (args.path == Path::kB1) {
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
        if (i + 1 < args.q) HIP_CHECK(hipStreamWaitEvent(comm_stream, gemm_end[i], 0));
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    } else {  // H0: comm stream can continue after each release while compute consumes it.
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
      }
      HIP_CHECK(hipEventRecord(done, comm_stream));
      HIP_CHECK(hipEventRecord(end, compute_stream));
    }
    HIP_CHECK(hipEventSynchronize(done));
    HIP_CHECK(hipEventSynchronize(end));
    Metrics metrics;
    if (args.path == Path::kCommOnly) {
      metrics.release_first_us = elapsed_us(issue, release.front());
      metrics.release_last_us = elapsed_us(issue, release.back());
      metrics.done_us = elapsed_us(issue, done);
      metrics.e2e_us = elapsed_us(issue, end);
    } else if (args.path == Path::kGemmOnly) {
      metrics.gemm_first_start_us = elapsed_us(issue, gemm_start.front());
      metrics.gemm_last_end_us = elapsed_us(issue, gemm_end.back());
      metrics.gemm_interval_us = elapsed_us(gemm_start.front(), gemm_end.back());
      metrics.done_us = elapsed_us(issue, done);
      metrics.e2e_us = elapsed_us(issue, end);
    } else {
      const int last_event = args.path == Path::kB0 ? 0 : args.q - 1;
      metrics.release_first_us = elapsed_us(issue, release.front());
      metrics.release_last_us = elapsed_us(issue, release[last_event]);
      metrics.done_us = elapsed_us(issue, done);
      metrics.gemm_first_start_us = elapsed_us(issue, gemm_start.front());
      metrics.gemm_last_end_us = elapsed_us(issue, gemm_end[last_event]);
      metrics.gemm_interval_us = elapsed_us(gemm_start.front(), gemm_end[last_event]);
      metrics.e2e_us = elapsed_us(issue, end);
    }
    return metrics;
  };

  // Warmup includes the full path, so RCCL and rocBLAS initialization is outside timed samples.
  for (int i = 0; i < args.warmup; ++i) {
    run_once();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::ostringstream rank_path;
  rank_path << args.output_dir << "/raw_rank" << rank << ".csv";
  std::ofstream rank_csv(rank_path.str());
  rank_csv << "run_id,rank,rank_count,device,device_name,gfx_arch,path,candidate,M,N,K,q,"
              "slice_bytes,warmup,iteration_index,t_issue_us,t_release_first_us,t_release_last_us,"
              "t_done_us,gemm_first_start_us,gemm_last_end_us,gemm_interval_us,e2e_us,gemm_tflops,"
              "correctness,max_abs_error,max_rel_error,mismatch_count,status\n";
  std::ofstream global_csv;
  if (rank == 0) {
    global_csv.open(args.output_dir + "/raw_global_samples.csv");
    global_csv << "run_id,rank_count,path,candidate,M,N,K,q,slice_bytes,iteration_index,"
                  "t_release_first_max_us,t_release_last_max_us,t_done_max_us,"
                  "gemm_first_start_max_us,gemm_last_end_max_us,gemm_interval_max_us,e2e_max_us,"
                  "gemm_tflops_min,correctness_all_ranks,max_abs_error_max,max_rel_error_max,"
                  "mismatch_count_sum,status\n";
  }

  int local_failures = 0;
  for (int iter = 0; iter < args.iters; ++iter) {
    MPI_Barrier(MPI_COMM_WORLD);
    Metrics m = run_once();
    ErrorStats host_error{0, 0, 0};
    bool needs_output_check = args.path != Path::kCommOnly && ((iter % args.verify_every) == 0);
    if (needs_output_check) {
      HIP_CHECK(hipMemsetAsync(device_error, 0, sizeof(ErrorStats), compute_stream));
      compare_output<<<(full_y_elements + threads - 1) / threads, threads, 0, compute_stream>>>(
          output, reference, full_y_elements, abs_tol, rel_tol, device_error);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipMemcpyAsync(&host_error, device_error, sizeof(ErrorStats), hipMemcpyDeviceToHost, compute_stream));
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
             << csv_escape(prop.name) << ",gfx928," << path_name(args.path) << ',' << args.candidate << ','
             << args.m_local << ',' << args.n << ',' << args.k << ',' << args.q << ',' << slice_bytes << ','
             << args.warmup << ',' << iter << ",0," << std::fixed << std::setprecision(3)
             << m.release_first_us << ',' << m.release_last_us << ',' << m.done_us << ','
             << m.gemm_first_start_us << ',' << m.gemm_last_end_us << ',' << m.gemm_interval_us << ','
             << m.e2e_us << ',' << gemm_tflops << ',' << (pass ? "PASS" : "FAIL") << ','
             << max_abs << ',' << max_rel << ',' << host_error.mismatch_count << ','
             << (pass ? "0" : "1") << '\n';
    rank_csv.flush();

    float local_values[8] = {m.release_first_us, m.release_last_us, m.done_us,
                             m.gemm_first_start_us, m.gemm_last_end_us, m.gemm_interval_us,
                             m.e2e_us, static_cast<float>(gemm_tflops)};
    float reduced_values[8] = {};
    MPI_Reduce(local_values, reduced_values, 7, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    // For throughput, the slowest rank is the relevant distributed operation rate.
    float negative_tflops = -local_values[7];
    float min_negative_tflops = 0.0f;
    MPI_Reduce(&negative_tflops, &min_negative_tflops, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
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
      global_csv << args.run_id << ',' << ranks << ',' << path_name(args.path) << ',' << args.candidate << ','
                 << args.m_local << ',' << args.n << ',' << args.k << ',' << args.q << ',' << slice_bytes << ','
                 << iter << ',' << std::fixed << std::setprecision(3) << reduced_values[0] << ','
                 << reduced_values[1] << ',' << reduced_values[2] << ',' << reduced_values[3] << ','
                 << reduced_values[4] << ',' << reduced_values[5] << ',' << reduced_values[6] << ','
                 << -min_negative_tflops << ',' << (all_pass ? "PASS" : "FAIL") << ',' << max_errors[0] << ','
                 << max_errors[1] << ',' << mismatch_sum << ',' << (all_pass ? "0" : "1") << '\n';
      global_csv.flush();
    }
  }

  int global_failures = 0;
  MPI_Allreduce(&local_failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::ofstream manifest(args.output_dir + "/manifest.csv");
    manifest << "run_id,platform_id,device_model,gfx_arch,rank_count,path,candidate,M,N,K,q,"
                "slice_bytes,warmup,iters,verify_every,abs_tolerance,rel_tolerance,"
                "requested_nccl_algo,requested_nccl_proto,requested_nccl_min_channels,"
                "requested_nccl_max_channels,status,rank_failure_count\n";
    const char* algo = std::getenv("NCCL_ALGO");
    const char* proto = std::getenv("NCCL_PROTO");
    const char* min_ch = std::getenv("NCCL_MIN_NCHANNELS");
    const char* max_ch = std::getenv("NCCL_MAX_NCHANNELS");
    manifest << args.run_id << ",K500SM_AI," << csv_escape(prop.name) << ",gfx928," << ranks << ','
             << path_name(args.path) << ',' << args.candidate << ',' << args.m_local << ',' << args.n << ','
             << args.k << ',' << args.q << ',' << slice_bytes << ',' << args.warmup << ',' << args.iters << ','
             << args.verify_every << ',' << abs_tol << ',' << rel_tol << ','
             << (algo ? algo : "DEFAULT") << ',' << (proto ? proto : "DEFAULT") << ','
             << (min_ch ? min_ch : "DEFAULT") << ',' << (max_ch ? max_ch : "DEFAULT") << ','
             << (global_failures == 0 ? "PASS" : "FAIL") << ',' << global_failures << '\n';
    printf("RESULT run_id=%s path=%s candidate=%s shape=[P=%d,m_local=%d,N=%d,K=%d,q=%d] status=%s failures=%d\n",
           args.run_id.c_str(), path_name(args.path), args.candidate.c_str(), ranks, args.m_local, args.n,
           args.k, args.q, global_failures == 0 ? "PASS" : "FAIL", global_failures);
  }

  HIP_CHECK(hipFree(device_error));
  for (int i = 0; i < args.q; ++i) {
    HIP_CHECK(hipEventDestroy(release[i]));
    HIP_CHECK(hipEventDestroy(gemm_start[i]));
    HIP_CHECK(hipEventDestroy(gemm_end[i]));
    HIP_CHECK(hipFree(gathered[i]));
    HIP_CHECK(hipFree(chunk_output[i]));
  }
  HIP_CHECK(hipEventDestroy(issue));
  HIP_CHECK(hipEventDestroy(done));
  HIP_CHECK(hipEventDestroy(end));
  HIP_CHECK(hipFree(x_local));
  HIP_CHECK(hipFree(full_a));
  HIP_CHECK(hipFree(weights));
  HIP_CHECK(hipFree(reference));
  HIP_CHECK(hipFree(output));
  ROCBLAS_CHECK(rocblas_destroy_handle(blas));
  HIP_CHECK(hipStreamDestroy(comm_stream));
  HIP_CHECK(hipStreamDestroy(compute_stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return global_failures == 0 ? 0 : 10;
}

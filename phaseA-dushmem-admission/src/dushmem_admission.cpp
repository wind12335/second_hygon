// Phase A DUSHMEM admission benchmark for K500SM_AI / gfx928 / 4 GPUs / PCIe.
// It intentionally measures only primitive correctness and release behavior.
// It is not an AG-GEMM performance benchmark.

#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>
#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string case_id;
  std::string mode = "put_signal";
  std::string outdir;
  size_t payload_bytes = 4096;
  uint64_t epochs = 100;
  int slots = 1;
  bool credit = false;
  bool quiet = false;
  bool credit_quiet = true;
  int expected_pes = 4;
};

struct DeviceError {
  unsigned long long mismatch_count;
  unsigned long long first_index;
  uint32_t expected;
  uint32_t observed;
  int producer;
  int first_set;
};

__device__ __forceinline__ uint32_t pattern_word(int producer, uint64_t epoch,
                                                  uint64_t index) {
  uint32_t x = static_cast<uint32_t>(index) * 747796405u;
  x ^= static_cast<uint32_t>(epoch) * 2891336453u;
  x ^= static_cast<uint32_t>(epoch >> 32) * 277803737u;
  x ^= static_cast<uint32_t>(producer + 1) * 2246822519u;
  x ^= x >> 16;
  x *= 2246822519u;
  x ^= x >> 13;
  return x;
}

__global__ void fill_pattern_kernel(uint32_t* dst, size_t words, int producer,
                                    uint64_t epoch) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < words) dst[index] = pattern_word(producer, epoch, index);
}

__global__ void check_pattern_kernel(const uint32_t* src, size_t words, int producer,
                                     uint64_t epoch, DeviceError* error) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= words) return;
  const uint32_t expected = pattern_word(producer, epoch, index);
  const uint32_t observed = src[index];
  if (observed == expected) return;

  atomicAdd(&error->mismatch_count, 1ULL);
  if (atomicCAS(&error->first_set, 0, 1) == 0) {
    error->first_index = index;
    error->expected = expected;
    error->observed = observed;
    error->producer = producer;
  }
}

void usage(const char* program) {
  std::cerr
      << "Usage: " << program << " --case-id ID --outdir DIR [options]\n"
      << "  --mode put_signal|fcollect\n"
      << "  --payload-bytes N --epochs N --slots N\n"
      << "  --credit 0|1 --quiet 0|1 --credit-quiet 0|1\n"
      << "  --expected-pes N\n";
}

bool parse_bool(const std::string& value, bool* result) {
  if (value == "0" || value == "false" || value == "False") {
    *result = false;
    return true;
  }
  if (value == "1" || value == "true" || value == "True") {
    *result = true;
    return true;
  }
  return false;
}

bool parse_options(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      std::cerr << "Missing value for " << key << "\n";
      return false;
    }
    const std::string value = argv[++i];
    try {
      if (key == "--case-id") options->case_id = value;
      else if (key == "--mode") options->mode = value;
      else if (key == "--outdir") options->outdir = value;
      else if (key == "--payload-bytes") options->payload_bytes = std::stoull(value);
      else if (key == "--epochs") options->epochs = std::stoull(value);
      else if (key == "--slots") options->slots = std::stoi(value);
      else if (key == "--expected-pes") options->expected_pes = std::stoi(value);
      else if (key == "--credit") {
        if (!parse_bool(value, &options->credit)) throw std::invalid_argument("bool");
      } else if (key == "--quiet") {
        if (!parse_bool(value, &options->quiet)) throw std::invalid_argument("bool");
      } else if (key == "--credit-quiet") {
        if (!parse_bool(value, &options->credit_quiet)) throw std::invalid_argument("bool");
      } else {
        std::cerr << "Unknown option: " << key << "\n";
        return false;
      }
    } catch (const std::exception&) {
      std::cerr << "Invalid value for " << key << ": " << value << "\n";
      return false;
    }
  }

  if (options->case_id.empty() || options->outdir.empty() || options->payload_bytes == 0 ||
      options->epochs == 0 || options->slots <= 0 || options->expected_pes <= 1) {
    std::cerr << "Missing or invalid required option.\n";
    return false;
  }
  if (options->payload_bytes % sizeof(uint32_t) != 0) {
    std::cerr << "payload-bytes must be a multiple of " << sizeof(uint32_t) << "\n";
    return false;
  }
  if (options->mode != "put_signal" && options->mode != "fcollect") {
    std::cerr << "Unsupported mode: " << options->mode << "\n";
    return false;
  }
  if (options->mode == "fcollect" && (options->credit || options->quiet || options->slots != 1)) {
    std::cerr << "fcollect requires slots=1, credit=0, quiet=0.\n";
    return false;
  }
  return true;
}

void hip_or_abort(hipError_t status, const char* what, MPI_Comm comm, int rank) {
  if (status == hipSuccess) return;
  std::fprintf(stderr, "rank=%d HIP failure in %s: %s\n", rank, what,
               hipGetErrorString(status));
  std::fflush(stderr);
  MPI_Abort(comm, 90);
}

double elapsed_us(hipEvent_t start, hipEvent_t end, MPI_Comm comm, int rank) {
  float ms = -1.0f;
  hip_or_abort(hipEventElapsedTime(&ms, start, end), "hipEventElapsedTime", comm, rank);
  return static_cast<double>(ms) * 1000.0;
}

size_t buffer_offset_words(int slot, int producer, int npes, size_t words) {
  return (static_cast<size_t>(slot) * npes + producer) * words;
}

size_t signal_offset(int endpoint_rank, int slot, int slots) {
  return static_cast<size_t>(endpoint_rank) * slots + slot;
}

void write_csv_header(std::ofstream* out) {
  *out << "case_id,mode,rank,epoch,slot,payload_bytes,world_size,slots,credit_enabled,quiet,"
          "issue_to_release_us,issue_to_comm_stream_complete_us,issue_to_checked_us,"
          "max_rank_checked_us,checksum_mismatches,first_bad_index,first_expected,"
          "first_observed,first_producer,fcollect_return,iteration_status\n";
}

void write_capability_header(std::ofstream* out) {
  *out << "case_id,rank,world_size,device,device_name,reported_arch,dushmem_version,"
          "peer_rank,peer_device,hip_p2p_access,dushmem_ptr_available\n";
}

std::string csv_string(const char* value) {
  std::string result = value == nullptr ? "" : value;
  std::replace(result.begin(), result.end(), '"', '\'');
  return "\"" + result + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(argv[0]);
    return 64;
  }

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpi_rank = -1;
  int mpi_size = 0;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  const char* local_rank_env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  const int device = local_rank_env == nullptr ? mpi_rank : std::atoi(local_rank_env);
  hip_or_abort(hipSetDevice(device), "hipSetDevice", comm, mpi_rank);
  hip_or_abort(hipInit(0), "hipInit", comm, mpi_rank);

  hipDevice_t hip_device;
  hip_or_abort(hipDeviceGet(&hip_device, device), "hipDeviceGet", comm, mpi_rank);
  hipCtx_t primary_context = nullptr;
  hip_or_abort(hipDevicePrimaryCtxRetain(&primary_context, hip_device),
               "hipDevicePrimaryCtxRetain", comm, mpi_rank);
  hip_or_abort(hipCtxSetCurrent(primary_context), "hipCtxSetCurrent", comm, mpi_rank);

  dushmemx_init_attr_t init_attr = DUSHMEMX_INIT_ATTR_INITIALIZER;
  init_attr.mpi_comm = &comm;
  const int init_status = dushmemx_init_attr(DUSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
  if (init_status != 0) {
    std::fprintf(stderr, "rank=%d dushmemx_init_attr returned %d\n", mpi_rank, init_status);
    MPI_Abort(comm, 91);
  }

  const int rank = dushmem_my_pe();
  const int npes = dushmem_n_pes();
  if (npes != options.expected_pes || npes != mpi_size) {
    std::fprintf(stderr,
                 "rank=%d invalid world: mpi_size=%d dushmem_n_pes=%d expected=%d\n",
                 rank, mpi_size, npes, options.expected_pes);
    std::fflush(stderr);
    dushmem_finalize();
    MPI_Finalize();
    return 65;
  }

  hipDeviceProp_t properties{};
  hip_or_abort(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties", comm, rank);
  int vendor_major = 0;
  int vendor_minor = 0;
  int vendor_patch = 0;
  dushmemx_vendor_get_version_info(&vendor_major, &vendor_minor, &vendor_patch);

  std::vector<int> rank_devices(npes, -1);
  MPI_Allgather(&device, 1, MPI_INT, rank_devices.data(), 1, MPI_INT, comm);

  const size_t words = options.payload_bytes / sizeof(uint32_t);
  const size_t recv_words = static_cast<size_t>(options.slots) * npes * words;
  const size_t signal_count = static_cast<size_t>(npes) * options.slots;

  auto* source = static_cast<uint32_t*>(dushmem_malloc(options.payload_bytes));
  auto* recv = static_cast<uint32_t*>(dushmem_malloc(recv_words * sizeof(uint32_t)));
  auto* ready = static_cast<uint64_t*>(dushmem_malloc(signal_count * sizeof(uint64_t)));
  auto* credit = static_cast<uint64_t*>(dushmem_malloc(signal_count * sizeof(uint64_t)));
  if (source == nullptr || recv == nullptr || ready == nullptr || credit == nullptr) {
    std::fprintf(stderr, "rank=%d symmetric allocation failed\n", rank);
    MPI_Abort(comm, 92);
  }

  DeviceError* device_error = nullptr;
  hip_or_abort(hipMalloc(&device_error, sizeof(DeviceError)), "hipMalloc(device_error)", comm, rank);

  hipStream_t comm_stream = nullptr;
  hipStream_t wait_stream = nullptr;
  hipStream_t ack_stream = nullptr;
  hip_or_abort(hipStreamCreateWithFlags(&comm_stream, hipStreamNonBlocking), "create comm stream", comm,
               rank);
  hip_or_abort(hipStreamCreateWithFlags(&wait_stream, hipStreamNonBlocking), "create wait stream", comm,
               rank);
  hip_or_abort(hipStreamCreateWithFlags(&ack_stream, hipStreamNonBlocking), "create ack stream", comm,
               rank);

  hipEvent_t issue_event = nullptr;
  hipEvent_t local_copy_event = nullptr;
  hipEvent_t comm_done_event = nullptr;
  hipEvent_t release_event = nullptr;
  hipEvent_t checked_event = nullptr;
  for (hipEvent_t* event : {&issue_event, &local_copy_event, &comm_done_event, &release_event,
                            &checked_event}) {
    hip_or_abort(hipEventCreate(event), "create event", comm, rank);
  }

  const std::string rank_csv_path = options.outdir + "/raw/rank_" + std::to_string(rank) + ".csv";
  const std::string capability_path =
      options.outdir + "/raw/capability_rank_" + std::to_string(rank) + ".csv";
  std::ofstream rank_csv(rank_csv_path);
  std::ofstream capability_csv(capability_path);
  if (!rank_csv || !capability_csv) {
    std::fprintf(stderr, "rank=%d cannot create CSV below %s\n", rank, options.outdir.c_str());
    MPI_Abort(comm, 93);
  }
  write_csv_header(&rank_csv);
  write_capability_header(&capability_csv);

  for (int peer = 0; peer < npes; ++peer) {
    int hip_p2p = 0;
    if (peer != rank) {
      hip_or_abort(hipDeviceCanAccessPeer(&hip_p2p, device, rank_devices[peer]),
                   "hipDeviceCanAccessPeer", comm, rank);
    }
    void* direct_ptr = peer == rank ? recv : dushmem_ptr(recv, peer);
    capability_csv << options.case_id << ',' << rank << ',' << npes << ',' << device << ','
                   << csv_string(properties.name) << ',' << csv_string(properties.gcnArchName) << ','
                   << vendor_major << '.' << vendor_minor << '.' << vendor_patch << ',' << peer << ','
                   << rank_devices[peer] << ',' << hip_p2p << ',' << (direct_ptr == nullptr ? 0 : 1)
                   << '\n';
  }
  capability_csv.flush();

  hip_or_abort(hipMemsetAsync(recv, 0, recv_words * sizeof(uint32_t), comm_stream), "clear recv", comm,
               rank);
  hip_or_abort(hipMemsetAsync(ready, 0, signal_count * sizeof(uint64_t), comm_stream), "clear ready",
               comm, rank);
  hip_or_abort(hipMemsetAsync(credit, 0, signal_count * sizeof(uint64_t), comm_stream), "clear credit",
               comm, rank);
  hip_or_abort(hipStreamSynchronize(comm_stream), "sync initial clear", comm, rank);
  dushmemx_barrier_all_on_stream(comm_stream);
  hip_or_abort(hipStreamSynchronize(comm_stream), "sync initial barrier", comm, rank);

  int local_failed = 0;
  if (rank == 0) {
    std::cout << "ADMISSION_START case=" << options.case_id << " mode=" << options.mode
              << " payload_bytes=" << options.payload_bytes << " epochs=" << options.epochs
              << " slots=" << options.slots << " credit=" << options.credit
              << " quiet=" << options.quiet << " npes=" << npes << " arch=gfx928\n";
    std::cout.flush();
  }

  const int threads = 256;
  const int blocks = static_cast<int>((words + threads - 1) / threads);
  for (uint64_t epoch = 1; epoch <= options.epochs; ++epoch) {
    const int slot = static_cast<int>((epoch - 1) % static_cast<uint64_t>(options.slots));

    if (options.credit && epoch > static_cast<uint64_t>(options.slots)) {
      const uint64_t reusable_epoch = epoch - static_cast<uint64_t>(options.slots);
      for (int consumer = 0; consumer < npes; ++consumer) {
        if (consumer == rank) continue;
        dushmemx_signal_wait_until_on_stream(
            &credit[signal_offset(consumer, slot, options.slots)], DUSHMEM_CMP_GE,
            reusable_epoch, comm_stream);
      }
      hip_or_abort(hipStreamSynchronize(comm_stream), "wait reusable credit", comm, rank);
    }

    fill_pattern_kernel<<<blocks, threads, 0, comm_stream>>>(source, words, rank, epoch);
    hip_or_abort(hipGetLastError(), "launch fill_pattern_kernel", comm, rank);
    hip_or_abort(hipEventRecord(issue_event, comm_stream), "record issue", comm, rank);

    int fcollect_return = 0;
    if (options.mode == "put_signal") {
      const size_t local_offset = buffer_offset_words(slot, rank, npes, words);
      hip_or_abort(hipMemcpyAsync(recv + local_offset, source, options.payload_bytes,
                                  hipMemcpyDeviceToDevice, comm_stream),
                   "copy local payload", comm, rank);
      hip_or_abort(hipEventRecord(local_copy_event, comm_stream), "record local copy", comm, rank);

      for (int peer = 0; peer < npes; ++peer) {
        if (peer == rank) continue;
        const size_t remote_offset = buffer_offset_words(slot, rank, npes, words);
        dushmemx_putmem_signal_on_stream(recv + remote_offset, source, options.payload_bytes,
                                         &ready[signal_offset(rank, slot, options.slots)], epoch,
                                         DUSHMEM_SIGNAL_SET, peer, comm_stream);
      }
      if (options.quiet) dushmemx_quiet_on_stream(comm_stream);
      hip_or_abort(hipEventRecord(comm_done_event, comm_stream), "record comm done", comm, rank);

      hip_or_abort(hipStreamWaitEvent(wait_stream, local_copy_event, 0), "wait local copy", comm,
                   rank);
      for (int producer = 0; producer < npes; ++producer) {
        if (producer == rank) continue;
        dushmemx_signal_wait_until_on_stream(
            &ready[signal_offset(producer, slot, options.slots)], DUSHMEM_CMP_GE, epoch,
            wait_stream);
      }
    } else {
      const size_t local_offset = buffer_offset_words(slot, 0, npes, words);
      fcollect_return = dushmemx_fcollectmem_on_stream(DUSHMEM_TEAM_WORLD, recv + local_offset,
                                                        source, options.payload_bytes, comm_stream);
      hip_or_abort(hipEventRecord(comm_done_event, comm_stream), "record fcollect done", comm,
                   rank);
      hip_or_abort(hipStreamWaitEvent(wait_stream, comm_done_event, 0), "wait fcollect", comm,
                   rank);
    }

    hip_or_abort(hipEventRecord(release_event, wait_stream), "record release", comm, rank);
    hip_or_abort(hipMemsetAsync(device_error, 0, sizeof(DeviceError), wait_stream), "clear error",
                 comm, rank);
    for (int producer = 0; producer < npes; ++producer) {
      const size_t offset = buffer_offset_words(slot, producer, npes, words);
      check_pattern_kernel<<<blocks, threads, 0, wait_stream>>>(recv + offset, words, producer, epoch,
                                                                  device_error);
      hip_or_abort(hipGetLastError(), "launch check_pattern_kernel", comm, rank);
    }
    hip_or_abort(hipEventRecord(checked_event, wait_stream), "record checked", comm, rank);
    hip_or_abort(hipEventSynchronize(checked_event), "sync checked", comm, rank);

    DeviceError host_error{};
    hip_or_abort(hipMemcpy(&host_error, device_error, sizeof(DeviceError), hipMemcpyDeviceToHost),
                 "copy error", comm, rank);
    const int iteration_bad = (host_error.mismatch_count != 0 || fcollect_return != 0) ? 1 : 0;
    int global_bad = 0;
    MPI_Allreduce(&iteration_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);

    const double issue_to_release_us = elapsed_us(issue_event, release_event, comm, rank);
    const double issue_to_comm_us = elapsed_us(issue_event, comm_done_event, comm, rank);
    const double issue_to_checked_us = elapsed_us(issue_event, checked_event, comm, rank);
    double max_rank_checked_us = 0.0;
    MPI_Allreduce(&issue_to_checked_us, &max_rank_checked_us, 1, MPI_DOUBLE, MPI_MAX, comm);

    const char* status = global_bad == 0 ? "PASS" : "FAIL_CHECKSUM_OR_COLLECTIVE";
    rank_csv << options.case_id << ',' << options.mode << ',' << rank << ',' << epoch << ',' << slot
             << ',' << options.payload_bytes << ',' << npes << ',' << options.slots << ','
             << (options.credit ? 1 : 0) << ',' << (options.quiet ? 1 : 0) << ',' << std::fixed
             << std::setprecision(3) << issue_to_release_us << ',' << issue_to_comm_us << ','
             << issue_to_checked_us << ',' << max_rank_checked_us << ','
             << host_error.mismatch_count << ',' << host_error.first_index << ','
             << host_error.expected << ',' << host_error.observed << ',' << host_error.producer << ','
             << fcollect_return << ',' << status << '\n';
    rank_csv.flush();

    if (global_bad != 0) {
      local_failed = 1;
      std::fprintf(stderr,
                   "ADMISSION_FAILURE case=%s rank=%d epoch=%" PRIu64
                   " slot=%d mismatches=%llu first_index=%llu producer=%d expected=%u observed=%u "
                   "fcollect_return=%d\n",
                   options.case_id.c_str(), rank, epoch, slot, host_error.mismatch_count,
                   host_error.first_index, host_error.producer, host_error.expected,
                   host_error.observed, fcollect_return);
      std::fflush(stderr);
      break;
    }

    if (options.credit) {
      for (int producer = 0; producer < npes; ++producer) {
        if (producer == rank) continue;
        dushmemx_signal_op_on_stream(&credit[signal_offset(rank, slot, options.slots)], epoch,
                                     DUSHMEM_SIGNAL_SET, producer, ack_stream);
      }
      if (options.credit_quiet) dushmemx_quiet_on_stream(ack_stream);
      hip_or_abort(hipStreamSynchronize(ack_stream), "sync credit publication", comm, rank);
    } else {
      MPI_Barrier(comm);
    }
  }

  int global_failed = 0;
  MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, comm);
  hip_or_abort(hipDeviceSynchronize(), "final device synchronize", comm, rank);
  dushmem_barrier_all();

  rank_csv.close();
  capability_csv.close();
  hipEventDestroy(issue_event);
  hipEventDestroy(local_copy_event);
  hipEventDestroy(comm_done_event);
  hipEventDestroy(release_event);
  hipEventDestroy(checked_event);
  hipStreamDestroy(ack_stream);
  hipStreamDestroy(wait_stream);
  hipStreamDestroy(comm_stream);
  hipFree(device_error);
  dushmem_free(credit);
  dushmem_free(ready);
  dushmem_free(recv);
  dushmem_free(source);
  dushmem_finalize();
  MPI_Finalize();

  if (rank == 0) {
    std::cout << "ADMISSION_END case=" << options.case_id
              << " status=" << (global_failed == 0 ? "PASS" : "FAIL") << "\n";
    std::cout.flush();
  }
  return global_failed == 0 ? 0 : 2;
}

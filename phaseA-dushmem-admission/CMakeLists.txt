cmake_minimum_required(VERSION 3.21)

# DUSHMEM v3.2.5's static device archive must be linked by hipcc, not by the
# lower-level clang++ command CMake normally emits for HIP. Keep CMake as a
# valid convenience entry point, but use a custom two-stage hipcc build that
# matches the DTK manual exactly.
project(dushmem_phase4_admission LANGUAGES NONE)

set(DUSHMEM_ADMISSION_ARCH gfx928 CACHE STRING
    "Hygon GPU architecture for this K500SM_AI experiment")
set(DTK_ROOT /opt/dtk CACHE PATH "DTK root containing hipcc and DUSHMEM")
set(MPI_ROOT /opt/mpi CACHE PATH "MPI root")

set(HIPCC "${DTK_ROOT}/bin/hipcc")
set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/dushmem_admission.cpp")
set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/dushmem_admission.o")
set(BINARY "${CMAKE_CURRENT_BINARY_DIR}/dushmem_admission")

if(NOT EXISTS "${HIPCC}")
  message(FATAL_ERROR "hipcc not found: ${HIPCC}")
endif()

add_custom_command(
  OUTPUT "${BINARY}"
  BYPRODUCTS "${OBJECT}"
  COMMAND "${HIPCC}" -O2 -g -std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
          --offload-arch=${DUSHMEM_ADMISSION_ARCH} -mcode-object-version=4 -fgpu-rdc
          -I${DTK_ROOT}/dushmem/include -I${MPI_ROOT}/include
          -c "${SOURCE}" -o "${OBJECT}"
  COMMAND "${HIPCC}" --offload-arch=${DUSHMEM_ADMISSION_ARCH}
          -mcode-object-version=4 -fgpu-rdc --hip-link "${OBJECT}" -o "${BINARY}"
          -L${DTK_ROOT}/dushmem/lib -ldushmem_device -ldushmem_host
          -L${MPI_ROOT}/lib -lmpi
          -Wl,-rpath,${DTK_ROOT}/dushmem/lib -Wl,-rpath,${MPI_ROOT}/lib
  DEPENDS "${SOURCE}"
  COMMENT "Building DUSHMEM admission benchmark with hipcc for ${DUSHMEM_ADMISSION_ARCH}"
  VERBATIM)

add_custom_target(build_dushmem_admission ALL DEPENDS "${BINARY}")

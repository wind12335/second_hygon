# Phase B cross-substrate benchmark build (NVIDIA port).
# Target: RTX 4090 / sm_89 / 4 GPUs / PCIe.
#
# nvcc handles RDC linking in a single command (unlike the two-stage hipcc
# build the HIP version needs). -rdc=true is REQUIRED for the NVSHMEM device
# API (signal_op_kernel -> nvshmem_uint64_signal).
#
# Override *_HOME on the command line for non-default installs, e.g.:
#   make NVSHMEM_HOME=/opt/nvshmem NCCL_HOME=/usr/local/nccl

CUDA_HOME    ?= /usr/local/cuda
NCCL_HOME    ?= /usr
NVSHMEM_HOME ?= /opt/nvshmem
MPI_HOME     ?= /usr

NVCC         ?= $(CUDA_HOME)/bin/nvcc

ARCH_FLAGS    = -arch=sm_89 -rdc=true
CXX_FLAGS     = -O3 -g -std=c++17 -lineinfo
INCLUDES      = -I$(NCCL_HOME)/include -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include
LIB_DIRS      = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib -L$(NVSHMEM_HOME)/lib -L$(MPI_HOME)/lib
# NVSHMEM CUDA builds link the shipped device + host libraries (see NVSHMEM
# Getting Started: -lnvshmem_device -lnvshmem_host). If your install exposes a
# single -lnvshmem, drop the missing names (README_NVIDIA_PORT.md risk #4).
LIBS          = -lnvshmem_device -lnvshmem_host -lnccl -lcublas -lmpi
RPATHS        = -Wl,-rpath,$(NCCL_HOME)/lib -Wl,-rpath,$(NVSHMEM_HOME)/lib -Wl,-rpath,$(MPI_HOME)/lib

SRC    := ag_gemm_phaseb_nv.cpp
TARGET := ag_gemm_phaseb_nv

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXX_FLAGS) $(ARCH_FLAGS) $(INCLUDES) $< -o $@ $(LIB_DIRS) $(LIBS) $(RPATHS)

clean:
	rm -f $(TARGET)

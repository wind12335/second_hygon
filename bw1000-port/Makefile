# Phase B cross-substrate benchmark build — bw1000 8-GPU port (2026-09-02).
# Two-stage hipcc, required for linking the DUSHMEM device runtime with RCCL
# and rocBLAS in one binary (direct CMake clang++ linking fails).
#
# ★ 架构必须按 env_check 报告覆盖，不要盲用默认值：
#     make ARCH=gfx9XX COV=4        # gfx 名字从 env_check_report_bw1000.txt 抄
# 所有路径变量均可被环境覆盖（不同 DTK 安装位置）。

HIPCC        ?= hipcc
DTK_ROOT     ?= /opt/dtk
DUSHMEM_ROOT ?= $(DTK_ROOT)/dushmem
MPI_ROOT     ?= /opt/mpi

ARCH         ?= gfx928
COV          ?= 4
ARCH_FLAGS    = --offload-arch=$(ARCH) -mcode-object-version=$(COV) -fgpu-rdc
CXX_FLAGS     = -O3 -g -std=c++17 -DHIP_ENABLE_WARP_SYNC_BUILTINS
INCLUDES      = -I$(DTK_ROOT)/include -I$(DUSHMEM_ROOT)/include -I$(MPI_ROOT)/include
LIB_DIRS      = -L$(DTK_ROOT)/lib -L$(DUSHMEM_ROOT)/lib -L$(MPI_ROOT)/lib
LIBS          = -ldushmem_device -ldushmem_host -lrccl -lrocblas -lamdhip64 -lmpi
RPATHS        = -Wl,-rpath,$(DTK_ROOT)/lib -Wl,-rpath,$(DUSHMEM_ROOT)/lib -Wl,-rpath,$(MPI_ROOT)/lib

SRC    := ag_gemm_phaseb.cpp
OBJ    := $(SRC:.cpp=.o)
TARGET := ag_gemm_phaseb

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(HIPCC) $(ARCH_FLAGS) -fgpu-rdc --hip-link $< -o $@ $(LIB_DIRS) $(LIBS) $(RPATHS)

%.o: %.cpp
	$(HIPCC) $(CXX_FLAGS) $(ARCH_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

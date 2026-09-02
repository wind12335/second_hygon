HIPCC ?= /opt/dtk/bin/hipcc
ARCH ?= gfx928
MPI_HOME ?= /opt/mpi
DTK_HOME ?= /opt/dtk
TARGET := ag_gemm_rccl

.PHONY: all clean

all: $(TARGET)

$(TARGET): ag_gemm_rccl.cpp
	$(HIPCC) -O3 -std=c++17 --offload-arch=$(ARCH) $< -o $@ \
		-I$(DTK_HOME)/include -I$(MPI_HOME)/include \
		-L$(DTK_HOME)/lib -lrccl -lrocblas -lamdhip64 \
		-L$(MPI_HOME)/lib -lmpi \
		-Wl,-rpath,$(DTK_HOME)/lib -Wl,-rpath,$(MPI_HOME)/lib

clean:
	rm -f $(TARGET)

EXE = fpga_ocl_reduction
CC  = gcc
CXX = g++

OPTFLAGS = -O2
CFLAGS   = $(OPTFLAGS) -Wall -g -std=gnu99 -fopenmp
CXXFLAGS = $(OPTFLAGS) -Wall -g -fopenmp

FPGA_CC = aoc
FPGA_EMULATOR = -march=emulator
FPGA_CL_FLAGS = -v -board=p385a_min_ax115 $(FPGA_EMULATOR)

# INC is obtained via "aocl compile-config", LDFLAGS is obtained via "aocl link-config"
INC      = -I/net/tools/reconfig/intel/17.1/hld/host/include
LDFLAGS  = -L/net/tools/reconfig/intel/17.1/hld/host/linux64/lib -lOpenCL 

INC     += -I./host
LDFLAGS += -fopenmp

OBJS = bin/FPGA_OpenCL_utils.o bin/OpenCL_reduction.o
AOCX = bin/my_reduction.aocx

all: $(EXE) $(AOCX)

$(EXE): $(OBJS) $(AOCX)
	$(CXX) $(OPTFLAGS) $(OBJS) -o bin/$(EXE) $(LDFLAGS)
	cp bin/$(EXE) ./
	cp $(AOCX)    ./

bin/my_reduction.aocx: device/my_reduction.cl
	$(FPGA_CC) $(FPGA_CL_FLAGS) device/my_reduction.cl -o bin/my_reduction.aocx
	
bin/FPGA_OpenCL_utils.o: host/FPGA_OpenCL_utils.h host/FPGA_OpenCL_utils.c
	$(CC)  $(CFLAGS)   $(INC) host/FPGA_OpenCL_utils.c  -c -o bin/FPGA_OpenCL_utils.o
	
bin/OpenCL_reduction.o: host/FPGA_OpenCL_utils.h host/OpenCL_reduction.cpp
	$(CXX) $(CXXFLAGS) $(INC) host/OpenCL_reduction.cpp -c -o bin/OpenCL_reduction.o

clean:
	$(RM) $(OBJS) $(AOCX) $(EXE)
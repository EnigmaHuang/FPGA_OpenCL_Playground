EXE = fpga_ocl_sgemm
CC  = gcc
CXX = g++

OPTFLAGS = -O2 -march=native
CFLAGS   = $(OPTFLAGS) -Wall -g -std=gnu99 -fopenmp
CXXFLAGS = $(OPTFLAGS) -Wall -g -fopenmp

FPGA_CC = aoc
FPGA_EMULATOR = -march=emulator
FPGA_CL_FLAGS = -v -board=p385a_min_ax115 $(FPGA_EMULATOR)

# INC is obtained via "aocl compile-config", LDFLAGS is obtained via "aocl link-config"
INC     = -I/net/tools/reconfig/intel/17.1/hld/host/include
LDFLAGS = -L/net/tools/reconfig/intel/17.1/hld/host/linux64/lib -lOpenCL 

INC     += -I./host
LDFLAGS += -fopenmp

OBJS = bin/FPGA_OpenCL_utils.o bin/test_sgemm.o bin/main.o
AOCX = bin/my_sgemm.aocx

all: $(EXE) $(AOCX)

$(EXE): $(OBJS) $(AOCX)
	$(CXX) $(OPTFLAGS) $(OBJS) -o bin/$(EXE) $(LDFLAGS)
	cp bin/$(EXE) ./
	cp $(AOCX)    ./

bin/my_sgemm.aocx: device/my_sgemm.cl
	$(FPGA_CC) $(FPGA_CL_FLAGS) device/my_sgemm.cl -o bin/my_sgemm.aocx
	
bin/FPGA_OpenCL_utils.o: host/FPGA_OpenCL_utils.h host/FPGA_OpenCL_utils.c
	$(CC)  $(CFLAGS)   $(INC) host/FPGA_OpenCL_utils.c -c -o bin/FPGA_OpenCL_utils.o
	
bin/test_sgemm.o: host/test_sgemm.c
	$(CC)  $(CFLAGS)   $(INC) host/test_sgemm.c -c -o bin/test_sgemm.o

bin/main.o: host/main.c host/test_sgemm.h host/FPGA_OpenCL_utils.h
	$(CC)  $(CFLAGS)   $(INC) host/main.c -c -o bin/main.o
	
clean:
	$(RM) $(OBJS) $(AOCX) $(EXE)
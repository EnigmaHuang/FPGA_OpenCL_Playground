EXE = fpga_ocl_boys
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

OBJS = bin/FPGA_OpenCL_utils.o bin/OpenCL_boys.o bin/boys_func_host.o
AOCX = bin/my_boys_func.aocx

all: $(EXE) $(AOCX)

$(EXE): $(OBJS) $(AOCX)
	$(CXX) $(OPTFLAGS) $(OBJS) -o bin/$(EXE) $(LDFLAGS)
	cp bin/$(EXE) ./
	cp $(AOCX)    ./

bin/my_boys_func.aocx: device/my_boys_func.cl device/vector_config.h device/boys_consts.h 
	$(FPGA_CC) $(FPGA_CL_FLAGS) device/my_boys_func.cl -o bin/my_boys_func.aocx
	
bin/FPGA_OpenCL_utils.o: host/FPGA_OpenCL_utils.h host/FPGA_OpenCL_utils.c
	$(CC) $(CFLAGS) $(INC) host/FPGA_OpenCL_utils.c -c -o bin/FPGA_OpenCL_utils.o
	
bin/boys_func_host.o: device/vector_config.h host/boys_consts_host.h host/boys_func_host.h
	$(CC) $(CFLAGS) $(INC) host/boys_func_host.c -c -o bin/boys_func_host.o
	
bin/OpenCL_boys.o: device/vector_config.h host/FPGA_OpenCL_utils.h host/OpenCL_boys.c
	$(CC) $(CFLAGS) $(INC) host/OpenCL_boys.c -c -o bin/OpenCL_boys.o

clean:
	$(RM) $(OBJS) $(AOCX) $(EXE)
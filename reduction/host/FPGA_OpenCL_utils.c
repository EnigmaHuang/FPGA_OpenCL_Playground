#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "FPGA_OpenCL_utils.h"

// Get platform from platform lists
int getCLPlatform(cl_platform_id *platform, const int platform_id)
{
	cl_uint numPlatforms; 
	cl_int  status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		printf("[Error] clGetPlatformIDs() error, returned status id = %d.\n", status);
		*platform = NULL;
		return status;
	}
	
	if (numPlatforms > 0)
	{
		cl_platform_id *platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if (platform_id < numPlatforms)
		{
			*platform = platforms[platform_id];
		} else {
			printf("[WARNING] Cannot find platform with id = %d (max %d), \
					fall back to platform 0.\n", platform_id, numPlatforms - 1);
			*platform = platforms[0];
		}
		free(platforms);
		return status;
	} else {
		printf("[ERROR] clGetPlatformIDs() returns 0 available platform.\n");
		*platform = NULL;
		return -1;
	}
}

// Query the platform and choose all FPGA devices
int getCLFPGADevicesID(const cl_platform_id platform, cl_device_id **device, cl_uint *numDevices)
{
	cl_device_id *_device = NULL;
	cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, numDevices);
	if ((*numDevices) > 0) 
	{
		_device = (cl_device_id*) malloc((*numDevices) * sizeof(cl_device_id));
		assert(_device != NULL);
		status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, (*numDevices), _device, NULL);
		*device = _device;
		return status;
	} else {
		printf("[ERROR] clGetDeviceIDs() returns 0 available device.\n");
		*device = NULL;
		return -1;
	}
}

// Read kernel binary file into a string
int readCLBinaryKernelFile(const char *file_name, size_t *file_size, unsigned char **file_content) 
{
	// Open the file
	FILE *inf = fopen(file_name, "r");
	if (inf == NULL) 
	{
		printf("[Error] Cannot opening kernel binary file %s\n", file_name);
		file_content = NULL;
		file_size = 0;
		return -1;
	}
	
	// Get its size
	fseek(inf, 0, SEEK_END);
	size_t size = ftell(inf);
	*file_size = size;
	rewind(inf);

	// Read the kernel code as a string
	unsigned char *_content = (unsigned char*) malloc(size * sizeof(unsigned char));
	size_t read_binary_count = fread(_content, size, 1, inf);
	assert(read_binary_count == 1);
	*file_content = _content;
	
	fclose(inf);
	return 0;
}

// Initialize with 1 device, 1 queue and 1 program, for simple tasks
int initCLFPGASimpleEnvironment(
	cl_device_id **FPGA_devices, cl_uint *numDevices, 
	cl_context *context, cl_command_queue *queue, 
	cl_program *program, const char *FPGA_bin_file_name
)
{
	int ret;
	
	// OpenCL extra step 1: get all platforms and choose the first available one
	cl_platform_id _platform;
	ret = getCLPlatform(&_platform, 0);
	if (ret != 0) return ret;
	
	// OpenCL extra step 2: query the platform and get device
	cl_device_id *_FPGA_devices;
	cl_uint _numDevices; 
	getCLFPGADevicesID(_platform, &_FPGA_devices, &_numDevices);
	if (ret != 0) return ret;
	
	// OpenCL extra step 3: create context (on first device)
	cl_context _context;
	_context = clCreateContext(NULL, 1, _FPGA_devices, NULL, NULL, NULL);
	
	// OpenCL extra step 4: create command queue associate with the context
	// _FPGA_devices[0] means we use the first FPGA device
	cl_command_queue _queue;
	_queue = clCreateCommandQueue(_context, _FPGA_devices[0], 0, NULL);
	
	// OpenCL extra step 5: create program object
	cl_int binary_status, errcode;
	cl_program _program;
	size_t binary_size;
	unsigned char *binary_content;
	
	ret = readCLBinaryKernelFile(FPGA_bin_file_name, &binary_size, &binary_content);
	if (ret != 0) return ret;
	
	_program = clCreateProgramWithBinary(_context, 1, _FPGA_devices, &binary_size, 
										(const unsigned char **) &binary_content, &binary_status, &errcode);
	if ((binary_status != CL_SUCCESS) || (errcode != CL_SUCCESS))
	{
		printf("[ERROR] clCreateProgramWithBinary() failed.\n");
		return -1;
	}
	
	// OpenCL extra step 6: build program
	// The 2nd and 3rd parameters mean that we use the first FPGA 
	errcode = clBuildProgram(_program, 1, _FPGA_devices, NULL, NULL, NULL);
	if (errcode != CL_SUCCESS)
	{
		printf("[ERROR] clBuildProgram() failed, returned status = %d\n", errcode);
		return -1;
	}
	
	// Set return values
	*FPGA_devices = _FPGA_devices;
	*numDevices   = _numDevices;
	*context = _context;
	*queue   = _queue;
	*program = _program;
	return 0;
}
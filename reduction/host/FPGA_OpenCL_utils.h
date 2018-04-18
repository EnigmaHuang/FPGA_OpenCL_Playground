#ifndef __FPGA_OPENCL_UTILS_H__
#define __FPGA_OPENCL_UTILS_H__

#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get platform from platform lists
int getCLPlatform(cl_platform_id *platform, const int platform_id);

// Query the platform and choose all FPGA devices
int getCLFPGADevicesID(const cl_platform_id platform, cl_device_id **device, cl_uint *numDevices);

// Read kernel binary file into a string
int readCLBinearyKernelFile(const char *file_name, size_t *file_size, unsigned char **file_content);

// Initialize with 1 device, 1 queue and 1 program, for simple tasks
int initCLFPGASimpleEnvironment(
	cl_device_id **FPGA_devices, cl_uint *numDevices, 
	cl_context *context, cl_command_queue *queue, 
	cl_program *program, const char *FPGA_bin_file_name
);

#ifdef __cplusplus
}
#endif

#endif
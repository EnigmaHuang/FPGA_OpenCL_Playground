//For clarity,error checking has been omitted.
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "FPGA_OpenCL_utils.h"

int main(int argc, char **argv)
{
	int n = atoi(argv[1]);
	n = (n + 63) / 64 * 64;
	size_t size_n = (size_t) n;
	size_t nBytes = sizeof(int) * (size_t) n;
	printf("Vector add, length = %d\n", n);
	
	// Allocate memory on host
	int *h_a, *h_b;
	h_a = (int*) malloc(nBytes);
	h_b = (int*) malloc(nBytes);
	
	// Init data on host
	for (int i = 0; i < n; i++)
	{
		h_a[i] = 114 + i;
		h_b[i] = 514 - i;
	}
	
	// Initialize Intel FPGA OpenCL environment
	cl_device_id *FPGA_devices;
	cl_uint numDevices;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	initCLFPGASimpleEnvironment(
		&FPGA_devices, &numDevices, &context, 
		&queue, &program, "vector_add.aocx"
	);
	
	// OpenCL extra step 7: create kernel object
	cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

	// Allocate memory on device
	cl_int err;
	cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes, NULL, &err);
	cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes, NULL, &err);
	
	// Copy data to device
	err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, nBytes, h_a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, nBytes, h_b, 0, NULL, NULL);

	// Set kernel arguments and launch kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &d_a);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &d_b);
	const size_t threads_in_workgroup[1] = {64};
	const size_t workspace_threads[1]	 = {size_n};
	cl_event event;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, workspace_threads, threads_in_workgroup, 0, NULL, &event);
	clWaitForEvents(1, &event);
	clReleaseEvent(event);

	// Generate result on host
	for (int i = 0; i < n; i++)	h_b[i] += h_a[i];
	
	// Copy result from device to host
	err = clEnqueueReadBuffer(queue, d_a, CL_TRUE, 0, nBytes, h_a, 0, NULL, NULL);
	
	// Check the results
	for (int i = 0; i < n; i++) assert(h_a[i] == h_b[i]);
	printf("Result is correct.\n");

	// Free host memory
	free(h_a);
	free(h_b);
	free(FPGA_devices);
	
	// Free device resources
	err = clReleaseKernel(kernel);      // Release kernel
	err = clReleaseProgram(program);	// Release the program object
	err = clReleaseMemObject(d_a);      // Release memory object
	err = clReleaseMemObject(d_b);      // Release memory object
	err = clReleaseCommandQueue(queue); // Release Command queue
	err = clReleaseContext(context);    // Release context
	
	return 0;
}
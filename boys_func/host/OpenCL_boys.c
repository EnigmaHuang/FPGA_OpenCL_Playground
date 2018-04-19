#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "../device/vector_config.h"
#include "FPGA_OpenCL_utils.h"
#include "boys_func_host.h"

void testBoysFunction(int order, FLOAT_TYPE *x, cl_context context, cl_command_queue queue, cl_program program)
{
	printf("Testing boys function with order %d\n", order);
	cl_kernel kernel = clCreateKernel(program, "boys_function", NULL);
	
	// Allocate host memory
	size_t x_mem_size = sizeof(FLOAT_TYPE) * BATCH_SIZE;
	size_t F_mem_size = sizeof(FLOAT_TYPE) * BATCH_SIZE * (order + 1);
	FLOAT_TYPE *h_F = (FLOAT_TYPE *) malloc(F_mem_size);
	FLOAT_TYPE *hdF = (FLOAT_TYPE *) malloc(F_mem_size);
	
	// Get reference result
	boys_function_host(order, x, h_F);
	
	// Allocate memory on device
	cl_int err;
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, x_mem_size, NULL, &err);
	cl_mem d_F = clCreateBuffer(context, CL_MEM_READ_WRITE, F_mem_size, NULL, &err);
	
	// Copy data to device
	cl_event h2d_copy;
	err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, x_mem_size, x, 0, NULL, &h2d_copy);
	clWaitForEvents(1, &h2d_copy);
	
	// Set kernel arguments and launch kernel
	cl_event kernel_exec;
	err = clSetKernelArg(kernel, 0, sizeof(int),    (void*) &order);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &d_x);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &d_F);
	err = clEnqueueTask(queue, kernel, 0, NULL, &kernel_exec);
	clWaitForEvents(1, &kernel_exec);
	
	// Copy result back to host
	cl_event d2h_copy;
	err = clEnqueueReadBuffer(queue, d_F, CL_TRUE, 0, F_mem_size, hdF, 0, NULL, &d2h_copy);
	clWaitForEvents(1, &d2h_copy);
	
	// Check result
	int passed = 1;
	for (int i = 0; i <= order; i++)
	{
		for (int j = 0; j < BATCH_SIZE; j++)
		{
			FLOAT_TYPE host_res = h_F[i * BATCH_SIZE + j];
			FLOAT_TYPE dev_res  = hdF[i * BATCH_SIZE + j];
			FLOAT_TYPE abs_diff = fabs(dev_res - host_res);
			FLOAT_TYPE rel_diff = abs_diff / fabs(host_res);
			if (rel_diff > 1e-6)
			{
				passed = 0;
				printf("Order %d, x = %f,\tref res = %e, dev res = %e, rel diff = %e \n", i, x[j], host_res, dev_res, rel_diff);
			} else {
				printf("Order %d, x = %f,\tref res = %e, dev res = %e\n", i, x[j], host_res, dev_res);
			}
		}
	}
	if (passed) printf("Check passed\n"); else printf("Check failed\n");
	
	// Release resources
	err = clReleaseKernel(kernel);      
	err = clReleaseMemObject(d_x);
	err = clReleaseMemObject(d_F);
	
	// Free host space
	free(h_F);
	free(hdF);
}

int main(int argc, char **argv)
{
	FLOAT_TYPE x[BATCH_SIZE] = {1.2, 3.4, 5.6, 7.8, 41.1, 42.2, 43.3, 44.4};
	
	// Initialize Intel FPGA OpenCL environment
	cl_device_id *FPGA_devices;
	cl_uint numDevices;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	initCLFPGASimpleEnvironment(
		&FPGA_devices, &numDevices, &context, 
		&queue, &program, "my_boys_func.aocx"
	);

	// Test boys function on device, order < 4 and > 4 has different code path
	testBoysFunction(3, x, context, queue, program);
	testBoysFunction(6, x, context, queue, program);
	
	// Free device resources
	clReleaseProgram(program);    // Release the program object
	clReleaseCommandQueue(queue); // Release Command queue
	clReleaseContext(context);    // Release context
	
	return 0;
}
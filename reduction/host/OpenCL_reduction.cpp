#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "FPGA_OpenCL_utils.h"
#include "../device/my_reduction.h"

void testReductionNDKernel(
	int *h_x, int n, size_t nBytes, int refres, 
	cl_context context, cl_command_queue queue, cl_program program
)
{
	printf("Testing NDRange kernel\n");
	cl_kernel kernel = clCreateKernel(program, "reduction_NDRange", NULL);
	
	// Allocate memory on device
	cl_int err;
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes,      NULL, &err);
	cl_mem res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	
	// Copy data to device
	cl_event h2d_copy;
	err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, nBytes, h_x, 0, NULL, &h2d_copy);
	clWaitForEvents(1, &h2d_copy);
	
	// Set kernel arguments and launch kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &d_x);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &res);
	err = clSetKernelArg(kernel, 2, sizeof(int),    (void*) &n);
	const size_t kernel_wg_size[1] = {WG_SIZE};
	const size_t kernel_ws_size[1] = {WG_SIZE};
	cl_event kernel_exec;
	double st = omp_get_wtime();
	for (int i = 0; i < 20; i++)
	{
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, kernel_ws_size, kernel_wg_size, 0, NULL, &kernel_exec);
		clWaitForEvents(1, &kernel_exec);
	}
	double et = omp_get_wtime();
	double ut = et - st;
	double bw = nBytes * 20.0 / (ut * 1000000000.0);
	printf("20 runs used time = %lf (s), effective bandwidth = %lf GB/s \n", ut, bw);
	
	// Copy result back to host
	int dev_res;
	cl_event d2h_copy;
	err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, sizeof(int), &dev_res, 0, NULL, &d2h_copy);
	clWaitForEvents(1, &d2h_copy);
	
	// Check result
	float abserr = fabs(dev_res - refres);
	float relerr = abserr / fabs(refres);
	if (relerr < 1e-10)
	{
		printf("Check passed, ref res = %d, device res = %d, rel err = %e\n", refres, dev_res, relerr);
	} else {
		printf("Check failed, ref res = %d, device res = %d, rel err = %e\n", refres, dev_res, relerr);
	}
	
	// Release resources
	err = clReleaseKernel(kernel);      
	err = clReleaseMemObject(d_x);
	err = clReleaseMemObject(res);
}

void testReductionSingleTask(
	int *h_x, int n, size_t nBytes, int refres, 
	cl_context context, cl_command_queue queue, cl_program program
)
{
	printf("Testing single single work-item kernel\n");
	cl_kernel kernel = clCreateKernel(program, "reduction_task", NULL);
	
	// Allocate memory on device
	cl_int err;
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes,      NULL, &err);
	cl_mem res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	
	// Copy data to device
	cl_event h2d_copy;
	err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, nBytes, h_x, 0, NULL, &h2d_copy);
	clWaitForEvents(1, &h2d_copy);
	
	// Set kernel arguments and launch kernel
	int zero = 0;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &d_x);
	err = clSetKernelArg(kernel, 1, sizeof(int),    (void*) &zero);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &res);
	err = clSetKernelArg(kernel, 3, sizeof(int),    (void*) &zero);
	err = clSetKernelArg(kernel, 4, sizeof(int),    (void*) &n);
	cl_event kernel_exec;
	double st = omp_get_wtime();
	for (int i = 0; i < 20; i++)
	{
		err = clEnqueueTask(queue, kernel, 0, NULL, &kernel_exec);
		clWaitForEvents(1, &kernel_exec);
	}
	double et = omp_get_wtime();
	double ut = et - st;
	double bw = nBytes * 20.0 / (ut * 1000000000.0);
	printf("20 runs used time = %lf (s), effective bandwidth = %lf GB/s \n", ut, bw);
	
	// Copy result back to host
	int dev_res;
	cl_event d2h_copy;
	err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, sizeof(int), &dev_res, 0, NULL, &d2h_copy);
	clWaitForEvents(1, &d2h_copy);
	
	// Check result
	float abserr = fabs(dev_res - refres);
	float relerr = abserr / fabs(refres);
	if (relerr < 1e-10)
	{
		printf("Check passed, ref res = %d, device res = %d, rel err = %e\n", refres, dev_res, relerr);
	} else {
		printf("Check failed, ref res = %d, device res = %d, rel err = %e\n", refres, dev_res, relerr);
	}
	
	// Release resources
	err = clReleaseKernel(kernel);      
	err = clReleaseMemObject(d_x);
	err = clReleaseMemObject(res);
}

void testReductionMultiTask(
	int *h_x, int n, size_t nBytes, int refres, int nthreads, 
	cl_context context, cl_command_queue queue, cl_program program
)
{
	printf("Testing parallel single work-item kernel\n");
	// Create kernels for each thread
	cl_kernel *kernels = (cl_kernel*) malloc(sizeof(cl_kernel) * nthreads);
	for (int i = 0; i < nthreads; i++) 
		kernels[i] = clCreateKernel(program, "reduction_task", NULL);
	
	// Allocate memory on device
	cl_int err;
	size_t res_bytes = sizeof(int) * nthreads;
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes,    NULL, &err);
	cl_mem res = clCreateBuffer(context, CL_MEM_READ_WRITE, res_bytes, NULL, &err);
	
	// Copy data to device
	cl_event h2d_copy;
	err = clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, nBytes, h_x, 0, NULL, &h2d_copy);
	clWaitForEvents(1, &h2d_copy);
	
	// Set kernel arguments and launch kernels
	double ut = 0.0;
	#pragma omp parallel num_threads(nthreads) reduction(max:ut)
	{
		int tid  = omp_get_thread_num();
		long long _spos = (long long) n;
		_spos *= tid;
		_spos /= nthreads;
		long long _epos = (long long) n;
		_epos *= (tid + 1);
		_epos /= nthreads;
		int spos = (int) _spos;
		int epos = (int) _epos;
		int leng = epos - spos;
		
		clSetKernelArg(kernels[tid], 0, sizeof(cl_mem), (void*) &d_x);
		clSetKernelArg(kernels[tid], 1, sizeof(int),    (void*) &spos);
		clSetKernelArg(kernels[tid], 2, sizeof(cl_mem), (void*) &res);
		clSetKernelArg(kernels[tid], 3, sizeof(int),    (void*) &tid);
		clSetKernelArg(kernels[tid], 4, sizeof(int),    (void*) &leng);
		
		#pragma omp barrier
		cl_event kernel_exec;
		double t_ut = 0.0;
		for (int i = 0; i < 20; i++)
		{
			#pragma omp barrier
			double st = omp_get_wtime();
			err = clEnqueueTask(queue, kernels[tid], 0, NULL, &kernel_exec);
			clWaitForEvents(1, &kernel_exec);
			#pragma omp barrier
			double et = omp_get_wtime();
			t_ut += et - st;
		}
		
		ut = t_ut > ut ? t_ut : ut;
	}
	double bw = nBytes * 20.0 / (ut * 1000000000.0);
	printf("20 runs used time = %lf (s), effective bandwidth = %lf GB/s \n", ut, bw);
	
	// Copy result back to host
	int *dev_res = (int*) malloc(res_bytes);
	cl_event d2h_copy;
	err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, res_bytes, dev_res, 0, NULL, &d2h_copy);
	clWaitForEvents(1, &d2h_copy);
	int devres = 0;
	for (int i = 0; i < nthreads; i++) devres += dev_res[i];
	
	// Check result
	float abserr = fabs(devres - refres);
	float relerr = abserr / fabs(refres);
	if (relerr < 1e-10)
	{
		printf("Check passed, ref res = %d, device res = %d, rel err = %e\n", refres, devres, relerr);
	} else {
		printf("Check failed, ref res = %d, device res = %d, rel err = %e\n", refres, devres, relerr);
	}
	
	// Release resources
	for (int i = 0; i < nthreads; i++) clReleaseKernel(kernels[i]);  
	err = clReleaseMemObject(d_x);
	err = clReleaseMemObject(res);
}

int main(int argc, char **argv)
{
	int n = atoi(argv[1]);
	size_t nBytes = sizeof(int) * (size_t) n;
	printf("Reduction, length = %d\n", n);
	
	// Allocate memory on host
	int *x = (int*) malloc(nBytes);
	
	int refres = 0.0;
	srand(time(NULL));
	for (int i = 0; i < n; i++)
	{
		x[i] = rand() % 10;
		refres += x[i];
	}
	
	// Initialize Intel FPGA OpenCL environment
	cl_device_id *FPGA_devices;
	cl_uint numDevices;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	initCLFPGASimpleEnvironment(
		&FPGA_devices, &numDevices, &context, 
		&queue, &program, "my_reduction.aocx"
	);
	
	// Test traditional NDRange kernel
	//testReductionNDKernel(x, n, nBytes, refres, context, queue, program);
	
	// Test single work-item kernel with 1 thread
	testReductionSingleTask(x, n, nBytes, refres, context, queue, program);
	
	// Test single work-item kernel with 1 thread
	testReductionMultiTask(x, n, nBytes, refres, PARA_TASKS, context, queue, program);
	
	// Free device resources
	clReleaseProgram(program);    // Release the program object
	clReleaseCommandQueue(queue); // Release Command queue
	clReleaseContext(context);    // Release context
	
	free(x);
	
	return 0;
}
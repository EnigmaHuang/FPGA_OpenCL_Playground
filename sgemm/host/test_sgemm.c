
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#include "test_sgemm.h"
#include "../device/my_sgemm.h"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define testKrnlParam1	C_height, C_width, comm_dim, alpha, beta, \
						h_A, h_B, h_C, context, queue, program

#define testKrnlParam2	const unsigned int C_height, const unsigned int C_width, \
						const unsigned int comm_dim, const float alpha, const float beta, \
						const float *h_A, const float *h_B, float *h_C, \
						cl_context context, cl_command_queue queue, cl_program program, \
						const char *kernel_name, const size_t *kernel_wg_size, const size_t *kernel_ws_size
						
void testKernel(testKrnlParam2)
{
	printf("Target kernel: %s\n", kernel_name);
	
	cl_kernel padzero_krnl   = clCreateKernel(program, "padZeros_rm", NULL);
	cl_kernel unpadzero_krnl = clCreateKernel(program, "removePadZeros_rm", NULL);
	cl_kernel sgemm_kernel   = clCreateKernel(program, kernel_name, NULL);
	
	unsigned int pad_C_height = CEIL_DIV(C_height, TILE_SIZE) * TILE_SIZE;
	unsigned int pad_C_width  = CEIL_DIV(C_width,  TILE_SIZE) * TILE_SIZE;
	unsigned int pad_comm_dim = CEIL_DIV(comm_dim, TILE_SIZE) * TILE_SIZE;
	unsigned int A_mem_size = C_height * comm_dim * sizeof(float);
	unsigned int B_mem_size = comm_dim * C_width  * sizeof(float);
	unsigned int C_mem_size = C_height * C_width  * sizeof(float);
	unsigned int padA_mem_size = pad_C_height * pad_comm_dim * sizeof(float);
	unsigned int padB_mem_size = pad_comm_dim * pad_C_width  * sizeof(float);
	unsigned int padC_mem_size = pad_C_height * pad_C_width  * sizeof(float);
	
	// Allocate memory on device
	cl_int err;
	cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, A_mem_size, NULL, &err);
	cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_WRITE, B_mem_size, NULL, &err);
	cl_mem d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, C_mem_size, NULL, &err);
	cl_mem d_padA = clCreateBuffer(context, CL_MEM_READ_WRITE, padA_mem_size, NULL, &err);
	cl_mem d_padB = clCreateBuffer(context, CL_MEM_READ_WRITE, padB_mem_size, NULL, &err);
	cl_mem d_padC = clCreateBuffer(context, CL_MEM_READ_WRITE, padC_mem_size, NULL, &err);
	
	printf("Test case size (%d, %d, %d) --padding--> (%d, %d, %d)\n", 
			C_height, C_width, comm_dim, pad_C_height, pad_C_width, pad_comm_dim);
	
	double st = omp_get_wtime();
	
	for (int itest = 0; itest < 20; itest++)
	{
		// Copy data to device
		cl_event h2d_copy[3];
		err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, A_mem_size, h_A, 0, NULL, &h2d_copy[0]);
		err = clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, B_mem_size, h_B, 0, NULL, &h2d_copy[1]);
		err = clEnqueueWriteBuffer(queue, d_C, CL_TRUE, 0, C_mem_size, h_C, 0, NULL, &h2d_copy[2]);
		
		// Launch kernels for zero padding
		cl_event dev_pad0[3];
		const size_t wg_size[2] = {TILE_SIZE, TILE_SIZE};
		// Pad zero for A
		err = clSetKernelArg(padzero_krnl, 0, sizeof(unsigned int), (void*) &C_height);
		err = clSetKernelArg(padzero_krnl, 1, sizeof(unsigned int), (void*) &comm_dim);
		err = clSetKernelArg(padzero_krnl, 2, sizeof(unsigned int), (void*) &pad_C_height);
		err = clSetKernelArg(padzero_krnl, 3, sizeof(unsigned int), (void*) &pad_comm_dim);
		err = clSetKernelArg(padzero_krnl, 4, sizeof(cl_mem), (void*) &d_A);
		err = clSetKernelArg(padzero_krnl, 5, sizeof(cl_mem), (void*) &d_padA);
		const size_t ws_sizeA[2] = {pad_comm_dim, pad_C_height};
		err = clEnqueueNDRangeKernel(queue, padzero_krnl, 2, NULL, ws_sizeA, wg_size, 1, &h2d_copy[0], &dev_pad0[0]);
		// Pad zero for B
		err = clSetKernelArg(padzero_krnl, 0, sizeof(unsigned int), (void*) &comm_dim);
		err = clSetKernelArg(padzero_krnl, 1, sizeof(unsigned int), (void*) &C_width);
		err = clSetKernelArg(padzero_krnl, 2, sizeof(unsigned int), (void*) &pad_comm_dim);
		err = clSetKernelArg(padzero_krnl, 3, sizeof(unsigned int), (void*) &pad_C_width);
		err = clSetKernelArg(padzero_krnl, 4, sizeof(cl_mem), (void*) &d_B);
		err = clSetKernelArg(padzero_krnl, 5, sizeof(cl_mem), (void*) &d_padB);
		const size_t ws_sizeB[2] = {pad_C_width, pad_comm_dim};
		err = clEnqueueNDRangeKernel(queue, padzero_krnl, 2, NULL, ws_sizeB, wg_size, 1, &h2d_copy[1], &dev_pad0[1]);
		// Pad zero for C
		err = clSetKernelArg(padzero_krnl, 0, sizeof(unsigned int), (void*) &C_height);
		err = clSetKernelArg(padzero_krnl, 1, sizeof(unsigned int), (void*) &C_width);
		err = clSetKernelArg(padzero_krnl, 2, sizeof(unsigned int), (void*) &pad_C_height);
		err = clSetKernelArg(padzero_krnl, 3, sizeof(unsigned int), (void*) &pad_C_width);
		err = clSetKernelArg(padzero_krnl, 4, sizeof(cl_mem), (void*) &d_C);
		err = clSetKernelArg(padzero_krnl, 5, sizeof(cl_mem), (void*) &d_padC);
		const size_t ws_sizeC[2] = {pad_C_width, pad_C_height};
		err = clEnqueueNDRangeKernel(queue, padzero_krnl, 2, NULL, ws_sizeC, wg_size, 1, &h2d_copy[2], &dev_pad0[2]);
		
		// Launch compute kernel
		cl_event sgemm_event;
		err = clSetKernelArg(sgemm_kernel, 0,  sizeof(cl_mem), (void*) &d_padA);
		err = clSetKernelArg(sgemm_kernel, 1,  sizeof(unsigned int), (void*) &pad_comm_dim);
		err = clSetKernelArg(sgemm_kernel, 2,  sizeof(cl_mem), (void*) &d_padB);
		err = clSetKernelArg(sgemm_kernel, 3,  sizeof(unsigned int), (void*) &pad_C_width);
		err = clSetKernelArg(sgemm_kernel, 4,  sizeof(cl_mem), (void*) &d_padC);
		err = clSetKernelArg(sgemm_kernel, 5,  sizeof(unsigned int), (void*) &pad_C_width);
		err = clSetKernelArg(sgemm_kernel, 6,  sizeof(float), (void*) &alpha);
		err = clSetKernelArg(sgemm_kernel, 7,  sizeof(float), (void*) &beta);
		err = clSetKernelArg(sgemm_kernel, 8,  sizeof(unsigned int), (void*) &pad_comm_dim);
		err = clSetKernelArg(sgemm_kernel, 9,  sizeof(unsigned int), (void*) &pad_C_height);
		err = clSetKernelArg(sgemm_kernel, 10, sizeof(unsigned int), (void*) &pad_C_width);
		err = clEnqueueNDRangeKernel(queue, sgemm_kernel, 2, NULL, kernel_ws_size, kernel_wg_size, 3, &dev_pad0[0], &sgemm_event);
		
		// Launch kernels for removing padded zeros
		cl_event unpadC_event;
		err = clSetKernelArg(unpadzero_krnl, 0, sizeof(unsigned int), (void*) &pad_C_height);
		err = clSetKernelArg(unpadzero_krnl, 1, sizeof(unsigned int), (void*) &pad_C_width);
		err = clSetKernelArg(unpadzero_krnl, 2, sizeof(unsigned int), (void*) &C_height);
		err = clSetKernelArg(unpadzero_krnl, 3, sizeof(unsigned int), (void*) &C_width);
		err = clSetKernelArg(unpadzero_krnl, 4, sizeof(cl_mem), (void*) &d_padC);
		err = clSetKernelArg(unpadzero_krnl, 5, sizeof(cl_mem), (void*) &d_C);
		err = clEnqueueNDRangeKernel(queue, unpadzero_krnl, 2, NULL, ws_sizeC, wg_size, 1, &sgemm_event, &unpadC_event);
		
		// Copy C back to the host
		cl_event d2h_copy;
		err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, C_mem_size, h_C, 1, &unpadC_event, &d2h_copy);
		clWaitForEvents(1, &d2h_copy);
	}
	
	double et = omp_get_wtime();
	double ut = et - st;
	double valid_gflops = 2.0 * C_height * C_width * comm_dim * 20.0;
	double real_gflops  = 2.0 * pad_C_height * pad_C_width * pad_comm_dim * 20.0;
	valid_gflops /= 1000000000.0 * ut;
	real_gflops  /= 1000000000.0 * ut;
	printf("20 runs used time = %lf (s), valid GFlops = %lf, real GFlops = %lf\n", ut, valid_gflops, real_gflops);
	
	// Free device memory
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	clReleaseMemObject(d_padA);
	clReleaseMemObject(d_padB);
	clReleaseMemObject(d_padC);
	
	// Free device kernel
	err = clReleaseKernel(padzero_krnl);
	err = clReleaseKernel(sgemm_kernel);
	err = clReleaseKernel(unpadzero_krnl);
}

void testKernel1(testKernelParam)
{
	unsigned int pad_C_height = CEIL_DIV(C_height, TILE_SIZE) * TILE_SIZE;
	unsigned int pad_C_width  = CEIL_DIV(C_width,  TILE_SIZE) * TILE_SIZE;
	const size_t kernel1_wg_size[2] = {TILE_SIZE, TILE_SIZE};
	const size_t kernel1_ws_size[2] = {pad_C_width, pad_C_height};
	testKernel(testKrnlParam1, "sgemm_1_naive", kernel1_wg_size, kernel1_ws_size);
}

void testKernel2(testKernelParam)
{
	unsigned int pad_C_height = CEIL_DIV(C_height, TILE_SIZE) * TILE_SIZE;
	unsigned int pad_C_width  = CEIL_DIV(C_width,  TILE_SIZE) * TILE_SIZE;
	const size_t kernel2_wg_size[2] = {TILE_SIZE, TILE_SIZE};
	const size_t kernel2_ws_size[2] = {pad_C_width, pad_C_height};
	testKernel(testKrnlParam1, "sgemm_2_tiling", kernel2_wg_size, kernel2_ws_size);
}

void testKernel3(testKernelParam)
{
	unsigned int pad_C_height = CEIL_DIV(C_height, TILE_SIZE) * TILE_SIZE;
	unsigned int pad_C_width  = CEIL_DIV(C_width,  TILE_SIZE) * TILE_SIZE;
	const size_t kernel3_wg_size[2] = {TILE_SIZE / WPTN, TILE_SIZE / WPTM};
	const size_t kernel3_ws_size[2] = {pad_C_width / WPTN, pad_C_height / WPTM};
	testKernel(testKrnlParam1, "sgemm_3_2Dreg", kernel3_wg_size, kernel3_ws_size);
}

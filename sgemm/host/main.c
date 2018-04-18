#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include "FPGA_OpenCL_utils.h"
#include "test_sgemm.h"

int check_result(float *ref, float *target, int n)
{
	int cnt = 0;
	for (int i = 0; i < n; i++)
	{
		float diff = fabs(ref[i] - target[i]);
		float rdiff = diff / fabs(ref[i]);
		if (rdiff > 1e-7)
		{
			printf("ERROR: position %d, ref = %e, target = %e, abs diff = %e, rel diff = %e\n", 
					i, ref[i], target[i], diff, rdiff);
			cnt++;
			if (cnt == 10) return 0;
		}
	}
	return 1;
}

int main(int argc, char **argv)
{
	int M, N, K;
	M = atoi(argv[1]);
	N = atoi(argv[2]);
	K = atoi(argv[3]);
	
	float *h_A, *h_B, *h_C, *C_ref;
	h_A   = (float*) malloc(sizeof(float) * M * K);
	h_B   = (float*) malloc(sizeof(float) * K * N);
	h_C   = (float*) malloc(sizeof(float) * M * N);
	C_ref = (float*) malloc(sizeof(float) * M * N);
	
	// Generate random input
	for (int i = 0; i < M * K; i++) h_A[i] = (float) (i % K);
	for (int i = 0; i < K * N; i++) h_B[i] = (float) (i % N);
	for (int i = 0; i < M * N; i++) 
	{
		h_C[i]   = (float) (i % M);
		C_ref[i] = (float) (i % M);
	}
	float alpha = 1.0, beta = 0.0;  // Set beta = 0 to allow both multiple run and result check
	
	// Compute reference results
	#pragma omp parallel for 
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		{
			register float accu = 0.0;
			for (int k = 0; k < K; k++) accu += h_A[i * K + k] * h_B[k * N + j];
			C_ref[i * N + j] = beta * C_ref[i * N + j] + alpha * accu;
		}
		
	// Initialize Intel FPGA OpenCL environment
	cl_device_id *FPGA_devices;
	cl_uint numDevices;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	initCLFPGASimpleEnvironment(
		&FPGA_devices, &numDevices, &context, 
		&queue, &program, "my_sgemm.aocx"
	);
	
	// Test kernel 2
	testKernel2(
		M, N, K, alpha, beta, h_A, h_B, h_C,
		context, queue, program
	);
	
	// Check result
	if (check_result(C_ref, h_C, M * N)) printf("Check passed\n"); else printf("Check failed\n");
	
	// Test kernel 3
	testKernel3(
		M, N, K, alpha, beta, h_A, h_B, h_C,
		context, queue, program
	);
	
	// Check result
	if (check_result(C_ref, h_C, M * N)) printf("Check passed\n"); else printf("Check failed\n");
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(C_ref);
	
	return 0;
}
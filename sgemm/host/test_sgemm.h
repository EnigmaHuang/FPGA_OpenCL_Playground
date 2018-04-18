#ifndef __TEST_SGEMM_H__
#define __TEST_SGEMM_H__

#include <CL/cl.h>

#define testKernelParam const unsigned int C_height, const unsigned int C_width, \
						const unsigned int comm_dim, const float alpha, const float beta, \
						const float *h_A, const float *h_B, float *h_C, \
						cl_context context, cl_command_queue queue, cl_program program

#ifdef __cplusplus
extern "C" {
#endif

void testKernel1(testKernelParam);

void testKernel2(testKernelParam);

void testKernel3(testKernelParam);

#ifdef __cplusplus
}
#endif

#endif 


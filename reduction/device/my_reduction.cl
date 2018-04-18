
#include "my_reduction.h"

#define WARP_REDUCE(x)	if (tid < (x)) buffer[tid] += buffer[tid + (x)]; \
						barrier(CLK_LOCAL_MEM_FENCE); 

__kernel 
__attribute((reqd_work_group_size(WG_SIZE, 1, 1)))
void reduction_NDRange(__global int * restrict x, __global int * restrict res, int length)
{
	__local int buffer[WG_SIZE];
	
	int tid   = get_local_id(0);
	int nstep = length / WG_SIZE;
	int idx   = tid;
	
	int sum = 0;
	#pragma unroll 4
	for (int i = 0; i < nstep; i++)
	{
		idx = tid + i * WG_SIZE;
		sum += x[idx];
	}
	if (idx + WG_SIZE < length) sum += x[idx + WG_SIZE];
	buffer[tid] = sum;
	barrier(CLK_LOCAL_MEM_FENCE); 
	
	WARP_REDUCE(4096);
	WARP_REDUCE(2048);
	WARP_REDUCE(1024);
	WARP_REDUCE(512);
	WARP_REDUCE(256);
	WARP_REDUCE(128);
	WARP_REDUCE(64);
	WARP_REDUCE(32);
	WARP_REDUCE(16);
	WARP_REDUCE(8);
	WARP_REDUCE(4);
	WARP_REDUCE(2);
	
	if (tid == 0) res[0] = buffer[0] + buffer[1];
}

__attribute__((task)) 
__attribute__((num_compute_units(PARA_TASKS)))
kernel
void reduction_task(
	__global int * restrict x,   int x_offset,
	__global int * restrict res, int res_offset, int length
)
{
	int sum = 0;
	
	for (int i = 0; i < length; i++)
	{
		int xi = x[x_offset + i];
		sum += xi;
	}
	
	res[res_offset] = sum;
}

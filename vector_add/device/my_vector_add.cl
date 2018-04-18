
#include "my_vector_add.h"

__kernel void vector_add(__global int * restrict a, __global int * restrict b)
{
	int global_thread_id = get_group_id(0) * get_local_size(0) + get_local_id(0);
	int result = a[global_thread_id] + b[global_thread_id];
	a[global_thread_id] = result;
}


#include "my_sgemm.h"

#define KernelParameters __global const float * restrict A, const unsigned int lda, \
	__global const float * restrict B, const unsigned int ldb, \
	__global float * restrict C, const unsigned int ldc, \
	const float alpha, const float beta, \
	const unsigned int common_dim, \
	const unsigned int c_height, \
	const unsigned int c_width


/* ---------- Auxiliary kernels ---------- */ 

// Padding row-major matrix with 0 on boundary to given size
__kernel
void padZeros_rm(
	const unsigned int rows, const unsigned int columns,
	const unsigned int pad_rows, const unsigned int pad_columns,
	__global const float * restrict input, __global float * restrict output
)
{
	const unsigned int col = get_global_id(0);
	const unsigned int row = get_global_id(1);
	if (col < pad_columns && row < pad_rows)
	{
		float val = 0.0f;
		if (col < columns && row < rows) val = input[row * columns + col];
		output[row * pad_columns + col] = val;
	}
}

// Inverse operation of padZeros(), the given matrix is row-major
__kernel
void removePadZeros_rm(
	const unsigned int pad_rows, const unsigned int pad_columns,
	const unsigned int rows, const unsigned int columns,
	__global const float * restrict input, __global float * restrict output
)
{
	const unsigned int col = get_global_id(0);
	const unsigned int row = get_global_id(1);
	if (col < columns && row < rows)
	{
		output[row * columns + col] = input[row * pad_columns + col];
	}
}
	
/* ---------- Basic kernels ---------- */
__kernel
void sgemm_1_naive(KernelParameters)
{
	const unsigned int globalRow = get_global_id(1); // Row ID of C (0..c_height)
	const unsigned int globalCol = get_global_id(0); // Col ID of C (0..c_width)
	
	float accu = 0.0f;
	
	for (unsigned int k = 0; k < common_dim; k++)
		accu += A[globalRow * lda + k] * B[k * ldb + globalCol];
	
	C[globalRow * ldc + globalCol] = alpha * accu + beta * C[globalRow * ldc + globalCol];
}


__kernel
__attribute((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__attribute((num_simd_work_items(4)))
void sgemm_2_tiling(KernelParameters)
{
	const unsigned int row = get_local_id(1); // Local row ID (max: TILE_SIZE)
	const unsigned int col = get_local_id(0); // Local col ID (max: TILE_SIZE)
	const unsigned int globalRow = get_global_id(1); // Row ID of C (0..c_height)
	const unsigned int globalCol = get_global_id(0); // Col ID of C (0..c_width)
	
	__local float Asub[TILE_SIZE][TILE_SIZE];
	__local float Bsub[TILE_SIZE][TILE_SIZE];
	
	float accu = 0.0f;
	
	const unsigned int numTiles = common_dim / TILE_SIZE;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		// Load one tile of A and B into local memory
		const unsigned int tiledRow = TILE_SIZE * t + row;
		const unsigned int tiledCol = TILE_SIZE * t + col;
		Asub[row][col] = A[globalRow * lda + tiledCol];
		Bsub[row][col] = B[tiledRow * ldb + globalCol];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Accumulation 
		#pragma unroll
		for (unsigned int k = 0; k < TILE_SIZE; k++)
			accu += Asub[row][k] * Bsub[k][col];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	C[globalRow * ldc + globalCol] = alpha * accu + beta * C[globalRow * ldc + globalCol];
}

/* ---------- Advanced kernels ---------- */
// Corresponding to kernel 6 in https://github.com/EnigmaHuang/my_CUDA_SGEMM
__kernel
__attribute((reqd_work_group_size(TILE_SIZE / WPTN, TILE_SIZE / WPTM, 1)))
__attribute((num_simd_work_items(4)))
void sgemm_3_2Dreg(KernelParameters)
{	
	// Thread identifiers
	const unsigned int col = get_local_id(0); // Local col ID (max: TILE_SIZE/WPTN == RTSN)
	const unsigned int row = get_local_id(1); // Local row ID (max: TILE_SIZE/WPTM == RTSM)
	const unsigned int col_block_id = get_group_id(0); // == blockIdx.y in CUDA
	const unsigned int row_block_id = get_group_id(1); // == blockIdx.x in CUDA
	const unsigned int globalCol = col_block_id * TILE_SIZE + col;   
	const unsigned int globalRow = row_block_id * TILE_SIZE + row;   

	// Local memory to fit a tile of TS*TS elements of A and B
	__local float As[TILE_SIZE][TILE_SIZE];
	__local float Bs[TILE_SIZE][TILE_SIZE];

	// Initialize the accumulation registers
	float acc[WPTM][WPTN];
	float Areg[WPTM], Breg[WPTN];
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++) 
		for (unsigned int wn = 0; wn < WPTN; wn++) 
			acc[wm][wn] = 0;
	
	// Loop over all tiles
	const unsigned int numTiles = common_dim / TILE_SIZE;
	for (unsigned int t = 0; t < numTiles; t++) 
	{
		const unsigned int read_A_topleft = TILE_SIZE * row_block_id * lda + t * TILE_SIZE;
		const unsigned int read_B_topleft = TILE_SIZE * t * ldb + TILE_SIZE * col_block_id;
		
		// Load A tile and B tile to the shm
		#pragma unroll
		for (unsigned int wm = 0; wm < WPTM; wm++)
		{
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTN; wn++)
			{
				unsigned int block_row = wm * RTSM + row;
				unsigned int block_col = wn * RTSN + col;
				unsigned int A_idx = read_A_topleft + block_row * lda + block_col;
				unsigned int B_idx = read_B_topleft + block_row * ldb + block_col;
				As[block_row][block_col] = A[A_idx];
				Bs[block_row][block_col] = B[B_idx];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		#pragma unroll
		for (unsigned int k = 0; k < TILE_SIZE; k++) 
		{
			// Cache the values of As in registers
			#pragma unroll
			for (unsigned int wm = 0; wm < WPTM; wm++) 
			{
				unsigned int irow = row + wm * RTSM;
				Areg[wm] = As[irow][k];
			}
			
			// Cache the values of Bs in registers
			#pragma unroll
			for (unsigned int wn = 0; wn < WPTN; wn++) 
			{
				unsigned int icol = col + wn * RTSN;
				Breg[wn] = Bs[k][icol];
			}
			
			// Perform the computation
			#pragma unroll
			for (unsigned int wm = 0; wm < WPTM; wm++) 
			{
				#pragma unroll
				for (unsigned int wn = 0; wn < WPTN; wn++)
				{
					acc[wm][wn] += Areg[wm] * Breg[wn];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Store the final results in C
	#pragma unroll
	for (unsigned int wm = 0; wm < WPTM; wm++)
	{
		unsigned int c_dim1 = (globalRow + wm * RTSM) * ldc;
		#pragma unroll
		for (unsigned int wn = 0; wn < WPTN; wn++)
		{
			unsigned int c_dim2 = globalCol + wn * RTSN;
			C[c_dim1 + c_dim2] = alpha * acc[wm][wn] + beta * C[c_dim1 + c_dim2];
		}
	}
}


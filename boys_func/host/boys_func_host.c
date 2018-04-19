#include <math.h>

#include "../device/vector_config.h"
#include "boys_func_host.h"
#include "boys_consts_host.h"

// TODO: Check if we should use native_{exp, pow} in OpenCL

static inline
void boys_F_split_small_n(int order, FLOAT_TYPE * restrict x, FLOAT_TYPE * restrict F)	
{
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		if (x[i] < BOYS_SHORTGRID_MAXX)  // boys_F_taylor(&F[i], x[i], order);
		{
			const int lookup_idx = (int)(BOYS_SHORTGRID_LOOKUPFAC*(x[i]+BOYS_SHORTGRID_LOOKUPFAC2));
			const FLOAT_TYPE xi = ((FLOAT_TYPE)lookup_idx * BOYS_SHORTGRID_SPACE);
			const FLOAT_TYPE dx = xi - x[i];
			
			int grid_offset0 = lookup_idx * (BOYS_SHORTGRID_MAXN + 1);

			for (int j = 0; j <= order; ++j)
			{
				int grid_offset = grid_offset0 + j;

				F[j * BATCH_SIZE + i] = boys_shortgrid[grid_offset]
								   + dx * (                  boys_shortgrid[grid_offset + 1]
								   + dx * ( (1.0/2.0   )   * boys_shortgrid[grid_offset + 2]
								   + dx * ( (1.0/6.0   )   * boys_shortgrid[grid_offset + 3]
								   + dx * ( (1.0/24.0  )   * boys_shortgrid[grid_offset + 4]
								   + dx * ( (1.0/120.0 )   * boys_shortgrid[grid_offset + 5]
								   + dx * ( (1.0/720.0 )   * boys_shortgrid[grid_offset + 6]
								   + dx * ( (1.0/5040.0)   * boys_shortgrid[grid_offset + 7]
								   )))))));
			}
		}
		else  // boys_F_long(&F[i], x[i], order);
		{
			FLOAT_TYPE x1 = 1.0 / x[i];
			FLOAT_TYPE x2 = sqrt(x1);
			int F_idx = i;

			for (int j = 0; j <= order; j++)
			{
				F[F_idx] = boys_longfac[j] * x2;
				F_idx += BATCH_SIZE;
				x2 *= x1;
			}
		}
	}
}

static inline
void boys_F_split_large_n(int order, FLOAT_TYPE * restrict x, FLOAT_TYPE * restrict F)	
{
	// Order is large - do only the highest, then recur down
	
	int top_offset = order * BATCH_SIZE;
	
	for(int i = 0; i < BATCH_SIZE; i++)
	{
		if (x[i] < BOYS_SHORTGRID_MAXX)  // F[top_offset + i] = boys_F_taylor_single(x[i], order);
		{
			const int lookup_idx = (int)(BOYS_SHORTGRID_LOOKUPFAC*(x[i]+BOYS_SHORTGRID_LOOKUPFAC2));
			const FLOAT_TYPE xi = ((FLOAT_TYPE)lookup_idx * BOYS_SHORTGRID_SPACE);
			const FLOAT_TYPE dx = xi - x[i];

			int grid_offset = lookup_idx * (BOYS_SHORTGRID_MAXN + 1) + order;

			F[top_offset + i] = boys_shortgrid[grid_offset]
				   + dx * (                  boys_shortgrid[grid_offset + 1]
				   + dx * ( (1.0/2.0   )   * boys_shortgrid[grid_offset + 2]
				   + dx * ( (1.0/6.0   )   * boys_shortgrid[grid_offset + 3]
				   + dx * ( (1.0/24.0  )   * boys_shortgrid[grid_offset + 4]
				   + dx * ( (1.0/120.0 )   * boys_shortgrid[grid_offset + 5]
				   + dx * ( (1.0/720.0 )   * boys_shortgrid[grid_offset + 6]
				   + dx * ( (1.0/5040.0)   * boys_shortgrid[grid_offset + 7]
				   )))))));
		}
		else  // F[top_offset + i] = boys_F_long_single(x[i], order);
		{
			FLOAT_TYPE p = -(2 * order + 1);
			FLOAT_TYPE x2 = pow(x[i], p);
			F[top_offset + i] = boys_longfac[order] * sqrt(x2);
		}
	}

    // factors for the recursion
	FLOAT_TYPE x2[BATCH_SIZE];
	FLOAT_TYPE ex[BATCH_SIZE];
	
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		x2[i] = 2.0 * x[i];
		ex[i] = exp(-x[i]);
	}

    // Recur down
    for (int n2 = order - 1; n2 >= 0; n2--)
    {
		FLOAT_TYPE den = 1.0 / (2.0 * n2 + 1);
		int offset0 = n2 * BATCH_SIZE;
		int offset1 = (n2 + 1) * BATCH_SIZE;
		
		// F[n2] = den * (x2 * F[(n2+1)] + ex)
		// TODO: Use shift reg to hold F
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			FLOAT_TYPE tmp1 = x2[i] * F[offset1 + i];
			FLOAT_TYPE tmp2 = tmp1 + ex[i];
			F[offset0 + i]  = den * tmp2;
		}
    }
}

void boys_function_host(int order, FLOAT_TYPE * restrict x, FLOAT_TYPE * restrict F)
{
	if (order < 4) boys_F_split_small_n(order, x, F);
	else           boys_F_split_large_n(order, x, F);
}

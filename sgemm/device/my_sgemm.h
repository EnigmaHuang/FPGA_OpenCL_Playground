#ifndef __MY_SGEMM_H__
#define __MY_SGEMM_H__

#define TILE_SIZE  64  // Tile size for loading into shared memory
#define WPTM       8   // Work per thread on dimension M (the height of C), == TILE_SIZE / RTSM
#define WPTN       4   // Work per thread on dimension N (the width of C),  == TILE_SIZE / RTSN
#define RTSM       8   // Reduced tile size on dimension M 
#define RTSN       16  // Reduced tile size on dimension N, should not be smaller than 16 for coalesced memory accessing 

#endif

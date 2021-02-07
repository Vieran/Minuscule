#include <stdio.h>
#include <stdlib.h>
#define TILE_DIM 10
//actually, I don't really understand what TILE_DIM is?
//and in line 25, what is the relationship between TILE_DIM and i?
//how does the matrix multi perform?

__global__ void simpleMultiply(float *a, float *b, float *c, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	for (int i = 0; i < TILE_DIM; i++)
		sum += a[row*TILE_DIM+i] * b[i*n+col];
	c[row*n+col] = sum;
}


__global__ void coalescedMutiply(float *a, float *b, float *c, int n) {
	__shared__ float aTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
	__syncwarp();
	for (int i = 0; i < TILE_DIM; i++)
		sum += aTile[threadIdx.y][i] * b[i*n+col];
	c[row*n+col] = sum;
}


__global__ void sharedABMultiply(float *a, float *b, float *c, int n) {
	__shared__ float aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
	bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*n+col];
	__syncthreads();
	for (int i = 0; i < TILE_DIM; i++)
		sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
	c[row*n+col] = sum;
}

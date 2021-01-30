#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

struct Matrix {
	int width;
	int heigth;
	float *elements;
};

//get matrix A's row and column
__device__ float getElement(Matrix *A, int row, int column) {
	return A->elements[row * A->width + column];
}

//assign elements for matrix A(row, column)
__device__ void setElement(Matrix *A, int row, int column, float value) {
	A->elements[row * A->width + column] = value;
}

//kernel matrix multi, each thread computes one element
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C) {
	float cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int column = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; i++)
		cvalue += getElement(A, row, i) * getElement(B, i, column);
	setElement(C, row, column, cvalue);
}

int main() {
	int width = 1 << 10;
	int heigth = 1 << 10;
	Matrix *A, *B, *C;

	//alloc memory
	cudaMallocManaged((void**)&A, sizeof(Matrix));
	cudaMallocManaged((void**)&B, sizeof(Matrix));
	cudaMallocManaged((void**)&C, sizeof(Matrix));
	int nBytes = width * heigth * sizeof(float);
	cudaMallocManaged((void**)&A->elements, nBytes);
	cudaMallocManaged((void**)&B->elements, nBytes); 
	cudaMallocManaged((void**)&C->elements, nBytes);

	//initial data
	A->heigth = B->heigth = C->heigth = heigth;
	A->width = B->width = C->width = width;
	for (int i = 0; i < width * heigth; i++) {
		A->elements[i] = 1.0;
		B->elements[i] = 2.0;
	}

	//setting for kernel
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (heigth + blockSize.y - 1) / blockSize.y);

	//execute kernel
	matMulKernel <<< gridSize, blockSize >>> (A, B, C);

	//synchronization
	cudaDeviceSynchronize();
	float maxError = 0.0;
	for (int i = 0; i < width * heigth; i++)
		maxError = fmaxf(maxError, fabs(C->elements[i] - 2 * width));
	printf("maxError: %f\n", maxError);

	return 0;
}

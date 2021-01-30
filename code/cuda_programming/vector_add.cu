#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

//from the output of this program, we can see that the memory alloc gets sth wrong

__global__ void add(float* x, float* y, float* z, int n) {
	//access the global index
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//threads per grid
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] + y[i];
}

int main() {
	int N = 1 << 20;
	int nBytes = N * sizeof(float);

	//alloc memory for host
	float *x, *y, *z;
	x = (float*)malloc(nBytes);
	y = (float*)malloc(nBytes);
	z = (float*)malloc(nBytes);

	//initial the data
	for (int i = 0; i < N; i++) {
		x[i] = 10.0;
		y[i] = 20.0;
	}

	//alloc memory for device
	float *d_x, *d_y, *d_z;
	cudaMalloc((void**)&d_x, nBytes);
	cudaMalloc((void**)&d_y, nBytes);
	cudaMalloc((void**)&d_z, nBytes);
	
	//copy the memory from host to device
	cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
	//setting for kernel running
	dim3 blockSzie(256);
	dim3 gridSize((N + blockSzie.x - 1) / blockSzie.x);
	//execute kernel
	add <<<gridSize, blockSzie >>> (d_x, d_y, d_z, N);

	//copy the memory from device to host
	cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyHostToDevice);

	//check the result
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = fmaxf(maxError, fabs(z[i] - 30.0));
	printf("maxError: %f\n", maxError);

	//free the memory of device
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	//free memory of host
	free(x);
	free(y);
	free(z);
	return 0;
}

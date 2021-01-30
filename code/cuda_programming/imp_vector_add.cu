#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

//cuda6.0引入统一内存（Unified Memory）避免再host和device上进行数据的深拷贝
//CUDA中使用cudaMallocManaged函数分配托管内存：
//cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flag=0);

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
	cudaMallocManaged((void**)&x, nBytes);
	cudaMallocManaged((void**)&y, nBytes);
	cudaMallocManaged((void**)&z, nBytes);

	//initial the data
	for (int i = 0; i < N; i++) {
		x[i] = 10.0;
		y[i] = 20.0;
	}

	//setting for kernel running
	dim3 blockSzie(256);
	dim3 gridSize((N + blockSzie.x - 1) / blockSzie.x);
	//execute kernel
	add <<<gridSize, blockSzie >>> (x, y, z, N);

	//同步device，确保结果能够正确访问
	cudaDeviceSynchronize();

	//check the result
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = fmaxf(maxError, fabs(z[i] - 30.0));
	printf("maxError: %f\n", maxError);

	//free the memory of device
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	return 0;
}

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
	int dev = 0;
	cudaDeviceProp devProp;
	//CHECK(cudaGetDeviceProperties(&devProp, dev));
	cudaGetDeviceProperties(&devProp, dev);
	printf("use GPU device %d: %s\n", dev, devProp.name);
	printf("number of SM: %d\n", devProp.multiProcessorCount);
	printf("the maximum shared memory in each thread block: %ld KB\n", devProp.sharedMemPerBlock / 1024);
	printf("the max num of threads in each thread block: %d\n", devProp.maxThreadsPerBlock);
	printf("the max num of thread in each EM: %d\n", devProp.maxThreadsPerMultiProcessor);
	printf("the max num of warp in each EM: %d\n", devProp.maxThreadsPerMultiProcessor / 32);
	return 0;
}

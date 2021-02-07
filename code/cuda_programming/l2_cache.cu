#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void kernel(int *data_persistent, int *data_stream, int dataSize, int freqSize) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize];
	data_stream[tid % dataSize] = 2 * data_stream[tid % dataSize];
}

int main() {
	// create cuda stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// cuda device properties variable
	cudaDeviceProp prop;
	int device_id;
	cudaGetDeviceProperties(&prop, device_id);
	size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
	int freqSize = 1LL << 10;
	int *data_persistent;
	cudaMallocManaged(&data_persistent, freqSize * (1LL << 20) * sizeof(int));
	cudaStreamAttrValue stream_attribute;
	stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(data_persistent);
	stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int);
	stream_attribute.accessPolicyWindow.hitRatio = 1.0;
	return 0;
}

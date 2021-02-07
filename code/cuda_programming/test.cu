#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

//this program is used to test the printf function in cuda kernel function
//printf in cuda use flush to pop it's containment to the stdout at some specific time, ie. when call cudaDeviceSynchronize()
//but i also find that, not using cudaDeviceSynchronize(), but just make this program wait for about 3 minutes also help printf works
//maybe it is sth relative to the lower structure of the machine

__global__ void multithread() {
	printf("threadid: %d\n", threadIdx.x);
}
__global__ void get_arr(int *a) {
	//fprintf(stderr, "in get_arr\n");
	printf("in get_arr\n");
	int b = a[0];
	printf("b = %d\n", b);
	printf("sinf4.736990305652013 = %lf\n", sinf(4.736990305652013));
	printf("__sinf4.736990305652013 = %lf\n", __sinf(4.736990305652013));
	printf("sinpi4.736990305652013 = %lf\n", sinpi(4.736990305652013));
	multithread<<<4,4>>>();
}


int main() {
	printf("sin4.736990305652013 in cpu = %lf\n", sin(4.736990305652013));
	int *a ;//= (int*)malloc(3*sizeof(int));
	cudaMallocManaged(&a, 3 * sizeof(int));
	a[0] = 3;
	a[1] = 2;
	a[2] = 1;
	get_arr<<<4,4>>>(a);
	//sleep(3);
	cudaDeviceSynchronize();
	cudaError_t cudastatus = cudaGetLastError();
	if (cudastatus != cudaSuccess) {
		fprintf(stderr, "error %s\n", cudaGetErrorString(cudastatus));
	}
	return 0;
}

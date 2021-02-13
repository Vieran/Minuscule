#include <stdio.h>
#include <stdlib.h>

//this program is mainly used to check function pointer support in cuda
/* result:
    call __global__ func pointer in host is ok
	call __global__ func pointer in device is not allow
	call __device__ func pointer in host is not allow
	call __device__ func pointer in device is ok
	all above require architecture above 35
	of cource you can never call a host func in device
*/


typedef struct gateobj {
	void (*func)(struct gateobj*);
	void (*func2)(struct gateobj*, int);
	struct gateobj* next;
} gateobj;

__global__ void func1(gateobj *g) {
	printf("func1\n");
}

__global__ void func2(gateobj *g, int a) {
	printf("func2\n");
}

__global__ void func3() {
	printf("func3\n");
}

__device__ void pointer_device(gateobj *g) {
	printf("in pointer_device\n");
}
__global__ void handler(gateobj *g) {
	printf("handler, call all funcs\n");
	/*
	switch(g->func2) {
		case 1:
			func1<<<2,1>>>(g);
			break;
		case 2:
			func2<<<2,1>>>(g, 4);
			break;
		case 3:
			func3<<<2,4>>>();
			break;
	}
	*/
	g->func(g);
	cudaDeviceSynchronize();
	if (g->func == pointer_device)
		printf("assert\n");
}

__host__ __device__ void get_func_p(gateobj *g) {
	g->func = pointer_device;
}

void register_gate() {
	gateobj *g;
	cudaMallocManaged(&g, sizeof(gateobj));
	get_func_p(g);
}

int main() {
	printf("in cpu\n");
	gateobj *g;
	cudaMallocManaged(&g, sizeof(gateobj));
	g->func = pointer_device;
//	g->func2 = pointer_device;
	handler<<<3,1>>>(g);
	cudaDeviceSynchronize();
	return 0;
}

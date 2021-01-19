#include <stdio.h>
#include <omp.h>

//don't forget to use -fopenmp as a flag
//remember to set OMP_NUM_THREADS in the environment

int main() {
	int num_threads, thread_id;
#pragma omp parallel private(num_threads, thread_id)
{
	thread_id = omp_get_thread_num();
	printf("hello world from thread %d\n", thread_id);
	if (thread_id == 0) {
		num_threads = omp_get_num_threads();
		printf("total %d threads\n", num_threads);
	}
}
	return 0;
}

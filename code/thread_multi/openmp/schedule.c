#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 20

//this program add array a and b and store it in result
//dynamically control each thread's workload
int main() {
	srand(time(NULL));
	int threads_num, thread_id;
	double a[N], b[N], result[N];

	//initialize
	for (int i = 0; i < N; i++) {
		a[i] = rand() / RAND_MAX;
		b[i] = rand() / RAND_MAX;
	}

	//set each thread's workload as 3
	//you can set OMP_NUM_THREADS=4 to see what happens
	int chunk = 3;

#pragma omp parallel private(thread_id)
{
	thread_id = omp_get_thread_num();
#pragma omp for schedule(static, chunk)
	for (int i = 0; i < N; i++) {
		result[i] = a[i] + b[i];
		printf("thread_id = %d working on index = %d\n", thread_id, i);
	}
}
return 0;
}

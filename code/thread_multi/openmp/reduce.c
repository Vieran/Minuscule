#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 53

int main() {
	srand(time(NULL));
	int n, chunk = 5;
	double a[N], b[N], result;

	for (int i = 0; i < N; i++) {
		a[i] = rand() * 0.0000001;
		b[i] = rand() * 0.0000001;
	}

	int i;
	int thread_id;
#pragma omp parallel for default(shared) private(thread_id) schedule(static, chunk) reduction(+ : result)
	for (i = 0; i < N; i++) {
		result += a[i] * b[i];
		thread_id = omp_get_thread_num();
		printf("thread_id = %d work on index = %d\n", thread_id, i);
	}

	printf("result = %lf", result);
	return 0;
}

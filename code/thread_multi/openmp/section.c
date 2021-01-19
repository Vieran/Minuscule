#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#define N 100

//use section to precisely decide each thread's work
//different sections usually work in different thread
//but, there is one thing strange, when not using sleep, all the section will only work in thread 0 ?
int main() {
	srand(time(NULL));
	double a[N], sum = 0, sum2 = 0;
	int thread_id;

#pragma omp parallel for
	for (int i = 0; i < N; i++)
		a[i] = rand() * 0.0000001;
	double max = a[0], min = a[0];

//fork several different threads
#pragma omp parallel sections private(thread_id)
{
	#pragma omp section
	{//get the max and the min of array a
		for (int i = 0; i < N; i++) {
			if (a[i] > max)
				max = a[i];
			if (a[i] < min)
				min = a[i];
		}
		thread_id = omp_get_thread_num();
		printf("thread_id = %d, max = %lf, min = %lf\n", thread_id, max, min);
		sleep(1);
	}
	#pragma omp section
	{//calculate the sum of array a
		for (int i = 0; i < N; i++)
			sum = sum + a[i];
		thread_id = omp_get_thread_num();
		printf("thread_id = %d, sum = %lf\n", thread_id, sum);
		sleep(1);
	}
	#pragma omp section
	{//calculate the square of array a
		for (int i = 0; i < N; i++)
			sum2 += a[i] * a[i];
		thread_id = omp_get_thread_num();
		printf("thread_id = %d, sum2 = %lf\n", thread_id, sum2);
		sleep(1);
	}
}
//jion
	return 0;
}

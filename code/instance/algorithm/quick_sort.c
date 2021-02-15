/* Quick Sort
 * this is the quick sort algorithm
 * including base as well as opt version
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#define length 10000000


double get_wall_time();

int divide(int arr[length], int low, int high) {
	int tmp = arr[low];
	do {
		while (low < high && arr[high] >= tmp)
			--high;
		if (low < high) {
			arr[low] = arr[high];
			++low;
		}

		while (low < high && arr[low] <= tmp)
			++low;
		if (low < high) {
			arr[high] = arr[low];
			--high;
		}
	} while (low != high);
	arr[low] = tmp;
	return low;
}

void quick_sort(int arr[length], int low, int high) {
	int mid;
	if (low >= high)
		return;
	mid = divide(arr, low, high);
	quick_sort(arr, low, mid - 1);
	quick_sort(arr, mid + 1, high);
}

void output(int arr[length]) {
	for (int i = 0; i < length; i++)
		printf("%d ", arr[i]);
	printf("\n");
}

int cmp(const void* num1, const void* num2) {
	return (*(int *)num1 - *(int *)num2);
}
int main() {
	int arr[length], arr_default[length];
	srand(time(NULL));
	for (int i = 0; i < length; i++) {
		arr[i] = rand()%1001;
		arr_default[i] = arr[i];
	}

	double t1 = get_wall_time();
	quick_sort(arr, 0, length - 1);
	double t2 = get_wall_time();
	qsort(arr_default, length, sizeof(int), cmp);
	double t3 = get_wall_time();

	printf("%12lf --- quick_sort\n", t2 - t1);
	printf("%12lf --- qsort\n", t3 - t2);

	return 0;
}


double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

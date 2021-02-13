/* global opimal solution --- random_walk
 * it is easy to drop into the local opimal solution
 * when searching the minimum point.
 * one of the way to avoid local opimal solution is random_walk
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define dimension 2
#define epsilon 1E-09
#define iteration_times 10000
#define num_vector 100
#define large_enough 1E10
double step = 10;

// standardize the vector u
void standardize(double *u) {
	double modulus = 0;
	for (int i = 0; i < dimension; i++)
		modulus += u[i] * u[i];

	for (int i = 0; i < dimension; i++)
		u[i] = u[i] / modulus;
}

// update next point
void update_next_point(double *start_point, double *next_point, double *u) {
	for (int i = 0; i < dimension; i++)
		next_point[i] = start_point[i] + step * u[i];
}

// define the target function
double func(double *variable) {
	// take f = sin(r)/r + 1, r = sqrt((x-50)^2+(y-50)^2)+e for example
	double r = sqrt(pow(variable[0]-50, 2) + pow(variable[1]-50, 2)) + M_E;
	double f = sin(r) / r + 1;
	return -f;
}

// update start point
void update_start_point(double *start_point, double *next_point) {
	for (int i = 0; i < dimension; i++)
		start_point[i] = next_point[i];
}

int main() {
	srand(time(NULL));
	double start_point[2] = {10, 10}; // start point
	double next_point[2] = {10, 10}; // next point

	int counter, walk_num = 0;
	while (step > epsilon) {
		counter = 1;
		while (counter < iteration_times) {
			// create and standardize num_vector vector u
			double u[dimension];
			double tmp_point[dimension], min = large_enough;
			for (int i = 0; i < num_vector; i++) {
				for (int k = 0; k < dimension; k++)
					u[k] = (double)(rand()%201)/100-1;
				standardize(u);

				// update location
				update_next_point(start_point, tmp_point, u);
				if (func(tmp_point) <= min) {
					update_start_point(next_point, tmp_point);
					min = func(next_point);
				}
			}

			// find better point
			if (func(next_point) < func(start_point)) {
				counter = 1;
				update_start_point(start_point, next_point);
			} else {
				counter++;
			}
		}
		step = step / 2;
		walk_num++;
		printf("random walk times: %d\n", walk_num);
	}

	printf("walk_num final: %d\n", walk_num);
	printf("final point: [%lf, %lf]\n", start_point[0], start_point[1]);
	printf("final f: %lf\n", func(start_point));
	return 0;
}

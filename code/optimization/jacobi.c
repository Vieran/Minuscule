#include <stdlib.h>
#include <stdio.h>
#define imax
#define kmax

int jacobi(int x, int y, int z, double phi[imax][kmax][2]) {
	int t0 = 0;
	int t1 = 1;
	int len, j;
	for (int i = 1; i <= len; i++) {
		for (int k = 1; k <= kmax; k++)
			for(j = 1; j <= imax; j++)
				phi[j][k][i] = (phi[j+1][k][t0] + phi[j-1][k][t0] + phi[j][k+1][t0] + phi[j][k-1][t0]) * 0.25;
	j = t0;
	t0 = t1;
	t1 = i;
	}
}

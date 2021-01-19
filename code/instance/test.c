#include <stdio.h>

void dis_m(double** m, int row, int column) {
	for (int i = 0; i< row; i++)
		for (int k = 0; k < column; k++)
			printf("%lf ", *((double*)m+i*row+k));
}

int main() {
	double m[10][10];
	for (int i = 0; i< 10; i++)
		for (int k = 0; k < 10; k++)
			m[i][k] = i*10 + k;

	dis_m(m, 10, 10);
	return 0;
}

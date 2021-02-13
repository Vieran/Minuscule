#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define row1 10000
#define row2 10000
#define column1 10000
#define column2 10000

int main() {
    srand((unsigned)time(NULL));
    //double (*matrix1)[column1] = (double (*)[column1])malloc(row1 * column1 * sizeof(double));
    //double (*matrix2)[column2] = (double (*)[column2])malloc(row2 * column2 * sizeof(double));
	double matrix1[row1][column1], matrix2[row2][column2];
	for (int i = 0; i < row1; i++)
		for (int j = 0; j < column1; j++)
			matrix1[i][j] = rand() * 0.0000001;
	for (int i = 0; i < row2; i++)
		for (int j = 0; j < column2; j++)
			matrix2[i][j] = rand() * 0.0000001;


    clock_t start_t, end_t;
    //double (* matrix3)[column2] = (double (*)[column2])malloc(row1 * column2 * sizeof(double));
	double matrix3[row1][column2];
    start_t = clock();
    for (int i = 0; i < row1; i++)
        for (int j = 0; j < column2; j++) {
			matrix3[i][j] = 0;
            for (int k = 0; k < row2; k++)
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j];
        }
   
    end_t = clock();
    printf("time = %lf", (double)(end_t - start_t)/CLOCKS_PER_SEC);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int row1 = 0, column1 = 0, row2 = 0, column2 = 0;
    scanf("%d %d %d", &row1, &column1, &column2);
    row2 = column1;
    double (*matrix1)[column1] = (double (*)[column1])malloc(row1 * column1 * sizeof(double));
    double (*matrix2)[column2] = (double (*)[column2])malloc(row2 * column2 * sizeof(double));

    srand((unsigned)time(NULL));

    for (int i = 0; i < row1; i++) {
        //printf("\n");
        for (int j = 0; j < column1; j++) {
            matrix1[i][j] = rand() * 0.001;
            //printf("%lf ", matrix1[i][j]);
        }
    }

    for (int i = 0; i < row2; i++) {
        //printf("\n");
        for (int j = 0; j < column2; j++) {
            matrix2[i][j] = rand() * 0.001;
            //printf("%lf ", matrix2[i][j]);
        }
    }

    clock_t start_t, end_t;
    double (* matrix3)[column2] = (double (*)[column2])malloc(row1 * column2 * sizeof(double));
    for (int i = 0; i < row1; i++)
        for (int j = 0; j < column2; j++)
            matrix3[i][j] = 0;

    start_t = clock();
    for (int i = 0; i < row1; i++) {
        //printf("\n");
        for (int j = 0; j < column2; j++) {
            for (int k = 0; k < row2; k++)
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j];
            //printf("%lf ", matrix3[i][j]);
        }
    }
    end_t = clock();
    printf("time = %lf", (double)(end_t - start_t)/CLOCKS_PER_SEC);

    free(matrix1);
    free(matrix2);
    free(matrix3);
    
    return 0;
}

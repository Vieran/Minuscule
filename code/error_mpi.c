#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define row1 100
#define row2 100
#define column1 100
#define column2 100


//we assume the row1 can be divided equally by the size
//make every process count row1/size for the matrix3 in row
int main(int argc, char** argv) {
    srand((unsigned)time(NULL));

	int root = 0;
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int job = row1 / size;
	printf("rank=%d, row1=%d, column1=%d, row2=%d, column2=%d, job=%d\n", rank, row1, column1, row2, column2, job);

	//initial matrix2 in all processes
	double matrix2[row2][column2];
	for (int i = 0; i < row2; i++) {
		//printf("\n");
		for (int j = 0; j < column2; j++) {
			matrix2[i][j] = rand() * 0.001;
			//printf("%lf ", matrix2[i][j]);
		}
	}

	if (rank == root) {
		//initial matrix1 in root process
		double matrix1[row1][column1];
		for (int i = 0; i < row1; i++) {
			//printf("\n");
			for (int j = 0; j < column1; j++) {
				matrix1[i][j] = rand() * 0.001;
				//printf("%lf ", matrix1[i][j]);
			}
		}
		//scatter part of matrix1 to all other processes
		MPI_Scatter(&matrix1[0][0], job * column1, MPI_DOUBLE, &matrix1[0][0], job * column1, MPI_DOUBLE, root, MPI_COMM_WORLD);

		//start computing
		double start_t, end_t;
		start_t = MPI_Wtime();
		double matrix3[row1][column2]; //store the result
		for (int i = 0; i < job; i++) {
			//printf("\n");
			for (int j = 0; j < column2; j++) {
				matrix3[rank * job + i][j] = 0;
				for (int k = 0; k < row2; k++)
					matrix3[rank * job + i][j] += matrix1[rank * job + i][k] * matrix2[k][j];
				//printf("%lf ", matrix3[i][j]);
			}
    	}

		//receive all the data from other process
		MPI_Gather(NULL, job * column2, MPI_DOUBLE, &matrix3, job * column2, MPI_DOUBLE, root, MPI_COMM_WORLD);
			
		end_t = MPI_Wtime();
    	printf("time = %lf", (double)(end_t - start_t));

	} else {
		double matrix1[job][column1];
		double matrix3[job][column2];
		//scatter part of matrix1 to all other processes
		MPI_Scatter(&matrix1[0][0], job * column1, MPI_DOUBLE, &matrix1[0][0], job * column1, MPI_DOUBLE, root, MPI_COMM_WORLD);
		for (int i = 0; i < job; i++) {
			//printf("\n");
			for (int j = 0; j < column2; j++) {
				matrix3[i][j] = 0;
				for (int k = 0; k < row2; k++)
					matrix3[i][j] += matrix1[i][k] * matrix2[k][j];
				//printf("%lf ", matrix3[i][j]);
			}
		}
		//receive all the data from other process
		MPI_Gather(&matrix3, job * column2, MPI_DOUBLE, NULL, job * column2, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}

	MPI_Finalize();
    return 0;
}

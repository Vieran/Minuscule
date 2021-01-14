#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

//we assume the row1 can be divided equally by the size
//make every process count row1/size for the matrix3 in row
int main(int argc, char** argv) {
    int row1, column1, row2, column2;
    scanf("%d %d %d", &row1, &column1, &column2);
    row2 = column1;

    srand((unsigned)time(NULL));

	MPI_Init(NULL, NULL);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int root = 0;
	int job;
	if (rank == root)
		job = row1 / size;
	
	MPI_Bcast(&job, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&row1, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&column1, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&row2, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&column2, 1, MPI_INT, root, MPI_COMM_WORLD);

	double (*matrix1)[column1] = (double (*)[column1])malloc(job * column1 * sizeof(double));
	double (*matrix2)[column2] = (double (*)[column2])malloc(row2 * column2 * sizeof(double));

	//initial matrix1
    for (int i = 0; i < job; i++) {
        //printf("\n");
        for (int j = 0; j < column1; j++) {
            matrix1[i][j] = rand() * 0.001;
            //printf("%lf ", matrix1[i][j]);
        }
    }

	//initial matrix2
    for (int i = 0; i < row2; i++) {
        //printf("\n");
        for (int j = 0; j < column2; j++) {
            matrix2[i][j] = rand() * 0.001;
            //printf("%lf ", matrix2[i][j]);
        }
    }
	printf("rank=%d, row1=%d, column1=%d, row2=%d, column2=%d, job=%d\n", rank, row1, column1, row2, column2, job);

	//counting and MPI
	if (rank == root) {
		//setting for counting time
		double start_t, end_t;
		start_t = MPI_Wtime();

		//create and initialize matrix3
    	double (* matrix3)[column2] = (double (*)[column2])malloc(row1 * column2 * sizeof(double));
		for (int i = rank; i < row1; i++)
			for (int j = 0; j < column2; j++)
				matrix3[i][j] = 0;

		//counting
		for (int i = 0; i < job; i++) {
			//printf("\n");
			for (int j = 0; j < column2; j++) {
				for (int k = 0; k < row2; k++)
					matrix3[rank * job + i][j] += matrix1[i][k] * matrix2[k][j];
				//printf("%lf ", matrix3[i][j]);
			}
    	}

		//use status to confirm which process send data
		MPI_Status status;
		//create matrix to receive data;
    	double (* matrix4)[column2] = (double (*)[column2])malloc(job * column2 * sizeof(double));
		//put data into matrix3 according to the send rank
		for (int i = 1; i < size; i++) {
			MPI_Recv(&*matrix4, job * column1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
			for (int j = 0; j < job; j++)
				for (int k = 0; k < column2; k++)
					matrix3[i * job + j][k] = matrix4[j][k];
		}
		
		end_t = MPI_Wtime();
    	printf("time = %lf", (double)(end_t - start_t)/CLOCKS_PER_SEC);
    	free(matrix3);

	} else {
    	double (* matrix3)[column2] = (double (*)[column2])malloc(job * column2 * sizeof(double));
		for (int i = 0; i < job; i++)
			for (int j = 0; j < column2; j++)
				matrix3[i][j] = 0;
		for (int i = 0; i < job; i++) {
			//printf("\n");
			for (int j = 0; j < column2; j++) {
				for (int k = 0; k < row2; k++)
					matrix3[i][j] += matrix1[i][k] * matrix2[k][j];
				//printf("%lf ", matrix3[i][j]);
			}
		}
		MPI_Send(&*matrix3, job * column2, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
    	free(matrix3);
	}


    free(matrix1);
    free(matrix2);
	MPI_Finalize();
    
    return 0;
}

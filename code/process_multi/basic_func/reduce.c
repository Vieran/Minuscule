/*show the basic usage of reduce*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//获取本地向量的大小
	int local_vector_size = atoi(argv[1]);

	//计算全局向量的大小
	int global_vector_size = local_vector_size * size;

	//初始化向量
	double *a, *b;
	a = (double*)malloc(local_vector_size * sizeof(double));
	b = (double*)malloc(local_vector_size * sizeof(double));
	for (int i = 0; i < local_vector_size; i++) {
		a[i] = 3.14 * rank;
		b[i] = 6.67 * rank;
	}

	//每个进程进行计算
	double partial_sum = 0;
	for (int i = 0; i < local_vector_size; i++)
		partial_sum += a[i] * b[i];

	//进行归约，计算总和
	int root = 0;
	double sum = 0;
	MPI_Reduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	if (rank == root)
		printf("the dot product is %g\n", sum);
	else
		printf("the partial_sum of rank %d is %g\n", rank, partial_sum);
	
	free(a);
	free(b);
	MPI_Finalize();
	return 0;
}
//Allreduce的使用类似reduce，只是不需要指定root

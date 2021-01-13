/*show the basic usage of gather*/
//分布式计算：广播、分散、收集、全局收集
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 4) {
		printf("this program need 4 processes");
		MPI_Finalize();
		exit(0);
	}

	//a is the send buffer, b is the receive buffer
	int a[4], b[4];
	for (int i = 0; i < 4; i++) {
		a[i] = 0;
		b[i] = 0;
	}
	a[0] = rank; //对于每一个进程，初始化其数组的第一个元素（也即发送的值

	int root = 0;

	//收集，并打印每个进程中的数组
	MPI_Gather(a, 1, MPI_INT, b, 1, MPI_INT, root, MPI_COMM_WORLD);
	printf("rank %d b[0] = %d, b[1] = %d, b[2] = %d, b[3] = %d\n", rank, b[0], b[1], b[2], b[3]);

	MPI_Finalize();
	return 0;
}

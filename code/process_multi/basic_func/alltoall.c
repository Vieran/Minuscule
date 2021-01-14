/*show the basic usage of alltoall*/
//alltoall是allgather在科学计算中的一个重要扩展
//这种模式中，不同的数据被发送到每个接收器，同时每个发送器也是接收器（每个进程都进行了发送和接收的过程，并且发送是发送给所有进程），看起来像是进行了矩阵转置
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
		a[i] = i + 1 + 4 * rank;
	}
	printf("rank %d a[0] = %d, a[1] = %d, a[2] = %d, a[3] = %d\n", rank, a[0], a[1], a[2], a[3]);
	MPI_Barrier(MPI_COMM_WORLD);

	//全局收集，隐含目标即全部进程，故不需要root参数
	MPI_Alltoall(a, 1, MPI_INT, b, 1, MPI_INT, MPI_COMM_WORLD);
	printf("rank %d b[0] = %d, b[1] = %d, b[2] = %d, b[3] = %d\n", rank, b[0], b[1], b[2], b[3]);

	MPI_Finalize();
	return 0;
}

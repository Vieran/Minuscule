/*show how broadcast work*/
//分布式计算：广播、分散、收集、全局收集
//广播与通信器的所有其他进程共享存在于一个进程上下文的值或结构，每个进程复制一份完整的数据
//分散聚合通信模式也是共享数据，但是每个进程只复制部分数据
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

	int a[4];
	for (int i = 0; i < 4; i++)
		a[i] = 0;

	int root = 0;
	//如果是根进程，则初始化数组
	if (rank == root) {
		a[0] = 3;
		a[1] = 5;
		a[2] = 8;
		a[3] = 4;
	}

	//广播，并打印每个进程中的数组（这个数组应该和root上的一样
	MPI_Bcast(a, 4, MPI_INT, root, MPI_COMM_WORLD);
	printf("rank %d a[0] = %d, a[1] = %d, a[2] = %d, a[3] = %d\n", rank, a[0], a[1], a[2], a[3]);

	MPI_Finalize();
	return 0;
}

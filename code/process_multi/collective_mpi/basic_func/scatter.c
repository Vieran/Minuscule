/*show the basic usage of scatter*/
//分布式计算：广播、分散、收集、全局收集
//广播与通信器的所有其他进程共享存在于一个进程上下文的值或结构，每个进程复制一份完整的数据
//分散聚合通信模式也是共享数据，但是每个进程只复制部分数据，但是它们拥有同样大小的内存空间
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

	int root = 0;
	//如果是根进程，则初始化数组
	if (rank == root) {
		a[0] = 3;
		a[1] = 5;
		a[2] = 8;
		a[3] = 4;
	}

	//分散，并打印每个进程中的数组（这个数组应该和root上的不一样
	MPI_Scatter(a, 1, MPI_INT, b, 1, MPI_INT, root, MPI_COMM_WORLD);
	printf("rank %d b[0] = %d, b[1] = %d, b[2] = %d, b[3] = %d\n", rank, b[0], b[1], b[2], b[3]);

	MPI_Finalize();
	return 0;
}
//实验发现，scater是严格分配的，root数组不足会继续往后分配（其实是非法访问了），而过多的时候则不会分配给最后一个进程，因为分配是按照发送or接收的字节数目决定的

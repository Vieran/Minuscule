/*show non-blocking communication*/
//使用MPI_Iseng和MPI_Irecv实现非阻塞点对点通信（避免死锁）
//这两个函数在调用之后立即返回，而不确认消息的传递操作是否已经完成，因此需要指定何时必须完成这些操作，这里使用了MPI_Wait

/*
 * int MPI_Isend(void *message,
 * 				int count,
 * 				MPI_Datatype datatype,
 * 				int dest,
 * 				int tag,
 * 				MPI_Comm comm,
 * 				MPI_Request *send_request);
 * int MPI_Irecv(void *message,
 * 				int count,
 * 				MPI_Datatype datatype,
 * 				int source,
 * 				int tag,
 * 				MPI_Comm comm,
 * 				MPI_Request *recv_request);
 * int MPI_Wait(MPI_Request *request, MPI_Status *status);
 * int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
 * */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2) {
		printf("this program need 2 processes");
		MPI_Finalize();
		exit(0);
	}

	int a, b;
	int tag = 0; //随意选取一个tag值
	MPI_Status status;
	MPI_Request send_request, recv_request;
	if (rank == 0 ) {
		a = 3; //随机选取一个数字即可
		//此处的send和recv顺序无关紧要，因为是非阻塞的
		MPI_Isend(&a, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(&b, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &recv_request);

		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		printf("process %d received value %d\n", rank, b);
	} else {
		a = 6; //随机选取一个数字即可
		//此处的send和recv顺序无关紧要，因为是非阻塞的
		MPI_Isend(&a, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &send_request);
		MPI_Irecv(&b, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &recv_request);

		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		printf("process %d received value %d\n", rank, b);
	}

	MPI_Finalize();
	return 0;
}

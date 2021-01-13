#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
	MPI_Init(NULL, NULL);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int message_send = 3;
	MPI_Send(&message_send, 1, MPI_INT, world_rank, 0, MPI_COMM_WORLD);
	printf("send message!");
	int message_recv;
	MPI_Recv(&message_recv, 1, MPI_INT, world_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("recieve message!");
	MPI_Finalize();
}

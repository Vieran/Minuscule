/*this program show hello world in order*/
//it is important to specific root and workers
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(NULL, NULL);
	//MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//buffer for sending and receiving message
	int message[2];
	int dest, src;
	int tag = 0;
	MPI_Status status;

	if (size == 1) {
		printf("this program needs more than one process");
		MPI_Finalize();
		exit(0);
	}

	if (rank != 0) {
		message[0] = rank;
		message[1] = size;
		dest = 0; //send all message to rank 0
		MPI_Send(message, 2, MPI_INT, dest, tag, MPI_COMM_WORLD);
	} else {
		for(src = 1; src < size; src++){
			MPI_Recv(message, 2, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			printf("hello world from process %d of %d\n", message[0], message[1]);
	}
	}

	MPI_Finalize();
	return 0;
}

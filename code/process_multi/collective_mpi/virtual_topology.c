#include <mpi.h>
#include <stdio.h>
/*this program show how to split the MPI_COMM_WORLD into several communicators
 *by the cart topology, which map it into a ... easy understandable graph(?)
 *so that the process can communicate with neighbors easily
 *(do not need to count the neighbors' coordinate manually)
 */
#define SIZE 16
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

int main(int argc, char *argv[]) {
	int numtasks, rank, source, dest, tag = 1;
	int inbuf[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
	int nbrs[4];
	int dims[2] = {4, 4};
	int periods[2] = {0, 0};
	int reorder = 0;
	int coords[2];

	MPI_Request reqs[8];
	MPI_Status stats[8];
	MPI_Comm cartcomm;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	if (numtasks == SIZE) {
		//create catesian virtual topology
		MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
		//get rank
		MPI_Comm_rank(cartcomm, &rank);
		//get coordinates
		MPI_Cart_coords(cartcomm, rank, 2, coords);
		//get neighbor ranks
		MPI_Cart_shift(cartcomm, 0, 1, &nbrs[UP], &nbrs[DOWN]);
		MPI_Cart_shift(cartcomm, 1, 1, &nbrs[LEFT], &nbrs[RIGHT]);

		printf("rank=%d, coords=%d %d, neighbors(u,d,l,r)=%d %d %d %d\n", rank, coords[0], coords[1], nbrs[UP], nbrs[DOWN], nbrs[LEFT], nbrs[RIGHT]);


		//exchange data(rank) with 4 neibors
		int outbuf = rank;
		for (int i = 0; i < 4; i++) {
			dest = nbrs[i];
			source = nbrs[i];
			MPI_Isend(&outbuf, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &reqs[i]);
			MPI_Irecv(&inbuf[i], 1, MPI_INT, source, tag, MPI_COMM_WORLD, &reqs[i+4]);
		}

		MPI_Waitall(8, reqs, stats);
		printf("rank=%d, inbuf(u,d,l,r)=%d %d %d %d\n", rank, inbuf[UP], nbrs[DOWN], nbrs[LEFT], nbrs[RIGHT]);
	}
	else
		printf("Must specify %d processors. Terminating.\n", SIZE);
	
	MPI_Finalize();
}

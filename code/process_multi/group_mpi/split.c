// use MPI_Comm_split to divide a communicator into subcommunicators

/*
//根据color和key将通讯器拆分为一组新的通讯器
MPI_Comm_split(
	MPI_Comm comm, //将用作新通讯器的基础
	int color, //确定每个进程将属于哪个新的通讯器，传递相同值的所有进程都分配给同一通讯器
	int key, //确定每个新通讯器中的顺序（秩）（key最小值的进程将为0，下一个最小值将为1，依此类推； 如果存在平局，则在原始通讯器中秩较低的进程将是第一位
	MPI_Comm* newcomm) //将新的通讯器返回给调用者
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  // Get the rank and size in the original communicator
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int color = world_rank / 4; // 根据行号确定颜色

  //根据颜色拆分通讯器，然后调用（相同颜色根据原始的秩确定新的秩
  MPI_Comm row_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &row_comm);

  int row_rank, row_size;
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm, &row_size);

  printf("WORLD RANK/SIZE: %d/%d --- ROW RANK/SIZE: %d/%d\n",
    world_rank, world_size, row_rank, row_size);

  //释放通讯器也是很重要的
  MPI_Comm_free(&row_comm);

  MPI_Finalize();
}
//note: if the processes can not be devided evenly, it will left the overages to the last one group

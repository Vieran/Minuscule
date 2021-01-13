// use MPI_Comm_split to divide a communicator into subcommunicators

/*
//创建通讯器的副本（对于使用库执行诸如数学库的特殊函数的应用非常有用，避免用户代码和库代码互相干扰
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)

MPI_Comm_create(
	MPI_Comm comm,
	MPI_Group group,
    MPI_Comm* newcomm)

MPI_Comm_group(
	MPI_Comm comm,
	MPI_Group* group)

//此函数将使用一个 MPI_Group 对象并创建一个与组具有相同进程的新通讯器
MPI_Comm_create_group(
	MPI_Comm comm,
	MPI_Group group,
	int tag,
	MPI_Comm* newcomm)

//选择组中特定的秩并将其构建为新的组
MPI_Group_incl(
	MPI_Group group,
	int n,
	const int ranks[],
	MPI_Group* newgroup)

//组对象的工作方式与通讯器对象相同，不同之处在于不能使用组与其他秩进行通信（因为组没有附加上下文）；组特有而通讯器无法完成的功能是使用组在本地构建新的组(远程操作涉及与其他秩的通信，而本地操作则没有)
//创建新的通讯器是一项远程操作，因为所有进程都需要决定相同的上下文和组，而在本地创建组是因为它不用于通信，因此每个进程不需要具有相同的上下文，也就是可以随意操作一个组，而无需执行任何通信
//获取组的秩为MPI_Group_rank，获得组的大小为MPI_Group_size

//对两个组进行并操作（类比集合的并集
MPI_Group_union(
	MPI_Group group1,
	MPI_Group group2,
	MPI_Group* newgroup)

//对两个组进行交操作（类比集合的交集
MPI_Group_intersection(
	MPI_Group group1,
	MPI_Group group2,
	MPI_Group* newgroup)
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  //获取进程在原始通讯器的秩和原始通讯器的大小
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  //获取MPI_COMM_WORLD中的进程组
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  //specific thr prime ranks in world_group
  int n = 7;
  const int ranks[7] = {1, 2, 3, 5, 7, 11, 13};

  // Construct a group containing all of the prime ranks in world_group
  MPI_Group prime_group;
  MPI_Group_incl(world_group, 7, ranks, &prime_group);

  // Create a new communicator based on the group
  MPI_Comm prime_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, prime_group, 0, &prime_comm);

  int prime_rank = -1, prime_size = -1;
  // If this rank isn't in the new communicator, it will be MPI_COMM_NULL
  // Using MPI_COMM_NULL for MPI_Comm_rank or MPI_Comm_size is erroneous
  if (MPI_COMM_NULL != prime_comm) {
    MPI_Comm_rank(prime_comm, &prime_rank);
    MPI_Comm_size(prime_comm, &prime_size);
  }

  printf("WORLD RANK/SIZE: %d/%d --- PRIME RANK/SIZE: %d/%d\n",
    world_rank, world_size, prime_rank, prime_size);

  MPI_Group_free(&world_group);
  MPI_Group_free(&prime_group);

  if (MPI_COMM_NULL != prime_comm) {
    MPI_Comm_free(&prime_comm);
  }

  MPI_Finalize();
}
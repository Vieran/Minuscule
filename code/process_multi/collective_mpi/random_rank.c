// Runs the TMPI_Rank function with random input.
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

/*
// Header file for TMPI_Rank
#ifndef __PARALLEL_RANK_H
#define __PARALLEL_RANK_H 1

int TMPI_Rank(void *send_data, void *recv_data, MPI_Datatype datatype, MPI_Comm comm);

#endif
//MPI标准明确指出，用户不应以 MPI 起头命名自己的函数，如 MPI_<something>，以避免将用户函数与MPI标准本身的函数混淆
*/

//保存进程在通讯器中的次序（rank）和对应数字；该结构体用于数组排序，并同时完整保留所属进程信息
typedef struct {
  int comm_rank;
  union { //保存要排序的数字（可以是浮点数或整数，因此使用union关键字
    float f;
    int i;
  } number;
} CommRankNumber;

// Gathers numbers for TMPI_Rank to process zero. Allocates enough space given the MPI datatype and
// returns a void * buffer to process 0. It returns NULL to all other processes.
void *gather_numbers_to_root(void *number, MPI_Datatype datatype, MPI_Comm comm) {
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  // Allocate an array on the root process of a size depending on the MPI datatype being used.
  int datatype_size;
  MPI_Type_size(datatype, &datatype_size); //function that detect the size of the datatype
  void *gathered_numbers;
  if (comm_rank == 0) {
    gathered_numbers = malloc(datatype_size * comm_size);
  }

  // Gather all of the numbers on the root process
  MPI_Gather(number, 1, datatype, gathered_numbers, 1, datatype, 0, comm);

  return gathered_numbers;
}

// A comparison function for sorting float CommRankNumber values
int compare_float_comm_rank_number(const void *a, const void *b) {
  CommRankNumber *comm_rank_number_a = (CommRankNumber *)a;
  CommRankNumber *comm_rank_number_b = (CommRankNumber *)b;
  if (comm_rank_number_a->number.f < comm_rank_number_b->number.f) {
    return -1;
  } else if (comm_rank_number_a->number.f > comm_rank_number_b->number.f) {
    return 1;
  } else {
    return 0;
  }
}

// A comparison function for sorting int CommRankNumber values
int compare_int_comm_rank_number(const void *a, const void *b) {
  CommRankNumber *comm_rank_number_a = (CommRankNumber *)a;
  CommRankNumber *comm_rank_number_b = (CommRankNumber *)b;
  if (comm_rank_number_a->number.i < comm_rank_number_b->number.i) {
    return -1;
  } else if (comm_rank_number_a->number.i > comm_rank_number_b->number.i) {
    return 1;
  } else {
    return 0;
  }
}

// 此函数在根进程上对收集到的数字排序并返回一个数组，数组按进程在通讯器中的次序排序
// 注意：此函数只在根进程上运行
int *get_ranks(void *gathered_numbers, int gathered_number_count, MPI_Datatype datatype) {
  int datatype_size;
  MPI_Type_size(datatype, &datatype_size);

  // Convert the gathered number array to an array of CommRankNumbers. This allows us to
  // sort the numbers and also keep the information of the processes that own the numbers
  // intact.
  CommRankNumber *comm_rank_numbers = malloc(gathered_number_count * sizeof(CommRankNumber));
  int i;
  for (i = 0; i < gathered_number_count; i++) {
    comm_rank_numbers[i].comm_rank = i;
    memcpy(&(comm_rank_numbers[i].number), gathered_numbers + (i * datatype_size), datatype_size);
  }

  // Sort the comm rank numbers based on the datatype
  if (datatype == MPI_FLOAT) {
    qsort(comm_rank_numbers, gathered_number_count, sizeof(CommRankNumber), &compare_float_comm_rank_number);
  } else {
    qsort(comm_rank_numbers, gathered_number_count, sizeof(CommRankNumber), &compare_int_comm_rank_number);
  }

  //现在comm_rank_numbers是排好序的，下面生成一个数组，包含每个进程的排名，数组第i个元素是进程i的数字的排名
  int *ranks = (int *)malloc(sizeof(int) * gathered_number_count);
  for (i = 0; i < gathered_number_count; i++) {
    ranks[comm_rank_numbers[i].comm_rank] = i;
  }

  // Clean up and return the rank array
  free(comm_rank_numbers);
  return ranks;
}

// Gets the rank of the recv_data, which is of type datatype.
// The rank is returned in send_data and is of type datatype.
int TMPI_Rank(void *send_data, void *recv_data, MPI_Datatype datatype, MPI_Comm comm) {
  // Check base cases first - Only support MPI_INT and MPI_FLOAT for this function.
  if (datatype != MPI_INT && datatype != MPI_FLOAT) {
    return MPI_ERR_TYPE;
  }

  int comm_size, comm_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  // To calculate the rank, we must gather the numbers to one process, sort the numbers, and then
  // scatter the resulting rank values. Start by gathering the numbers on process 0 of comm.
  void *gathered_numbers = gather_numbers_to_root(send_data, datatype, comm);

  // Get the ranks of each process
  int *ranks = NULL;
  if (comm_rank == 0) {
    ranks = get_ranks(gathered_numbers, comm_size, datatype);
  }

  // Scatter the rank results
  MPI_Scatter(ranks, 1, MPI_INT, recv_data, 1, MPI_INT, 0, comm);

  // Do clean up
  if (comm_rank == 0) {
    free(gathered_numbers);
    free(ranks);
  }
}


int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Seed the random number generator to get different results each time
  srand(time(NULL) * world_rank);

  float rand_num = rand() / (float)RAND_MAX;
  int rank;
  TMPI_Rank(&rand_num, &rank, MPI_FLOAT, MPI_COMM_WORLD);
  printf("Rank for %f on process %d is %d\n", rand_num, world_rank, rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
// use MPI_Reduce to compute the average of an array of elements in parallel

/*
//将归约值发送给root进程
MPI_Reduce(
    void* send_data,
    void* recv_data, //recv_data 数组包含归约的结果，大小为sizeof（datatype）* count
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm communicator)

//MPI_Op常见的归约运算：MPI_MAX - 返回最大元素；MPI_MIN - 返回最小元素；MPI_SUM - 对元素求和；MPI_PROD - 将所有元素相乘；MPI_LAND - 对元素执行逻辑与运算；MPI_LOR - 对元素执行逻辑或运算；MPI_BAND - 对元素的各个位按位与执行；MPI_BOR - 对元素的位执行按位或运算；MPI_MAXLOC - 返回最大值和所在的进程的秩；MPI_MINLOC - 返回最小值和所在的进程的秩
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }

  int num_elements_per_proc = atoi(argv[1]);

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Create a random array of elements on all processes.
  srand(time(NULL)*world_rank);   // Seed the random number generator to get different results each time for each processor
  float *rand_nums = NULL;
  rand_nums = create_rand_nums(num_elements_per_proc);

  // Sum the numbers locally
  float local_sum = 0;
  int i;
  for (i = 0; i < num_elements_per_proc; i++) {
    local_sum += rand_nums[i];
  }

  // Print the random numbers on each process
  printf("Local sum for process %d - %f, avg = %f\n",
         world_rank, local_sum, local_sum / num_elements_per_proc);

  // Reduce all of the local sums into the global sum
  float global_sum;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Print the result
  if (world_rank == 0) {
    printf("Total sum = %f, avg = %f\n", global_sum,
           global_sum / (world_size * num_elements_per_proc));
  }

  // Clean up
  free(rand_nums);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
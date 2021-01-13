//this file packed the function MPI_Recv(), make it dynamically alloc place for the program

//there are still some problems in this function
//it cannot determine the datatype and alloc the received messages

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void MPI_Recv(
    void* data,
//    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm communicator,
    MPI_Status* status) {

    MPI_Status status;
    MPI_Probe(source, tag, communicator, &status);
    
    int number_amount;
    MPI_Get_count(&status, datatype, &number_amount);

    datatype *data = (datatype *)malloc(sizeof(datatype) * number_amount);
    MPI_Recv(data, number_amount, datatype, communicator, MPI_STATUS_IGNORE);
}
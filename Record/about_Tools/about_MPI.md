# Learning MPI

## What is it ?

> Message Passing Interface
>
> 进程间通讯的协议，支持点对点和广播（它是一种标准而不是一种语言，openMPI、MPICH等对这个协议进行了c/cpp语言的实现
>
> 适用于非共享内存编程



## MPI编程基础

[MPI Tutorial](https://mpitutorial.com/)*这个教程写得非常详细，例程也齐全*

[MPI]([Message Passing Interface (MPI) (llnl.gov)](https://computing.llnl.gov/tutorials/mpi/))*与第一个对比着看，虽然这个页面古老了一点，但是还是写得很详细的*



## 命令行使用

### 单节点

```bash
#运行程序a.x，使用y个进程
mpirun -np y a.x

#指定在节点casxxx上运行
mpirun -host casxxx -np 4 ./a.x

#更多用法请查看man手册
```



### 多节点

```bash
#查看当前节点的名称
hostname

#在指定节点上运行
mpirun -hosts casxxx,casyyy,caszzz -n 4 ./a.x

#运行程序a.x，使用y个进程，节点信息记录在hostfile/machinefile中（不同节点默认使用换行符分割
#写在同一个hostfile/machinefile里面的节点，必须在同一个salloc申请出来（如果使用salloc的话）
mpirun -np y -f hostfile ./a.x
mpirun -np y -machinefile mf ./a.x #intel更常用machinefile（可以指定placement，也就是每个节点进程数

#debug信息开到5以上看进程分布
mpirun -machinefile mf -np 4 -genv I_MPI_DEBUG=5 ./demo
#附加了这个debug信息之后，发现hostfile和machinefile跑出来的进程分布不一样

#这样的操作是每个节点上运行一份
#至于pmi是什么，自行Google（看起来像是和mpi差不多的东西，但是slurm多节点需要它
srun -N3 --nodelist=cas026,cas347,cas390 --mpi=pmi2 mpirun -n 4 -ppn 1 ./demo
```

[Controlling Process Placement with the Intel® MPI Library](https://software.intel.com/content/www/us/en/develop/articles/controlling-process-placement-with-the-intel-mpi-library.html)

[Job Schedulers Support (intel.com)](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-guide-linux/top/running-applications/job-schedulers-support.html)

[Displaying MPI Debug Information](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-guide-linux/top/analysis-and-tuning/displaying-mpi-debug-information.html)

[slurm中文手册](https://docs.slurm.cn/users/mpi-he-upc-yong-hu-zhi-nan)


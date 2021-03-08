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

**参考**

[stdout - Ordering Output in MPI - Stack Overflow](https://stackoverflow.com/questions/5305061/ordering-output-in-mpi)



## 基本运行方式

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

#debug信息开到5以查看进程分布（genv用于指定Intel中全局的环境变量
mpirun -machinefile mf -np 4 -genv I_MPI_DEBUG=5 ./demo
#附加了这个debug信息之后，发现hostfile和machinefile跑出来的进程分布不一样
#-genv和-env参数，可以man查看其区别

#这样的操作是每个节点上运行一份
#至于pmi是什么，自行Google（看起来像是和mpi差不多的东西，但是slurm多节点需要它
srun -N3 --nodelist=cas026,cas347,cas390 --mpi=pmi2 mpirun -n 4 -ppn 1 ./demo
```

[Controlling Process Placement with the Intel® MPI Library](https://software.intel.com/content/www/us/en/develop/articles/controlling-process-placement-with-the-intel-mpi-library.html)

[Job Schedulers Support (intel.com)](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-guide-linux/top/running-applications/job-schedulers-support.html)

[Displaying MPI Debug Information](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-guide-linux/top/analysis-and-tuning/displaying-mpi-debug-information.html)

[slurm中文手册](https://docs.slurm.cn/users/mpi-he-upc-yong-hu-zhi-nan)



## 进程绑定

```bash
#查看cpu物理核心与逻辑核心的关系
cpuinfo

#在每个节点的cpu0和cpu3上运行（mf里共8个节点
mpirun -machinefile mf -genv I_MPI_PIN_PROCESSOR_LIST=0,3 -genv I_MPI_DEBUG=5 -n 16 ./demo

#实现细粒度的进程绑定：先绑定到对应的物理核心（node），然后再绑定到逻辑核心（core）
#进程0在cas274的cpu10，进程1、2在cas275的cpu0、cpu1，进程3、4、5在cas276的cpu3、cpu4、cpu5——但是这样就多线程就没了
mpirun -genv I_MPI_DEBUG=5 -host cas274 -env I_MPI_PIN_PROCESSOR_LIST=10 -n 1 ./demo : \
						   -host cas275 -env I_MPI_PIN_PROCESSOR_LIST=0-1 -n 2 ./demo : \
						   -host cas276 -env I_MPI_PIN_PROCESSOR_LIST=3-10 -n 3 ./demo
#进程0在cas274，进程1、2在cas275，进程3、4、5在cas276——多线程依然存在
mpirun -genv I_MPI_DEBUG=5 -host cas274 -n 1 ./demo : \
						   -host cas275 -n 2 ./affi : \
						   -host cas276 -n 3 ./affi

#进程0、3、4、5在cas174，进程1、2在cas275——多线程依然存在
mpirun -genv I_MPI_DEBUG=5 -host cas274 -n 1 ./demo : \
						   -host cas275 -n 2 ./affi : \
						   -host cas274 -n 3 ./affi
```

[hpcwiki: binding/pinning](https://hpc-wiki.info/hpc/Binding/Pinning)

[openmpi: man page](https://www.open-mpi.org/doc/v3.0/man1/mpirun.1.php)

[Intel(R) MPI Library Developer Reference for Linux* OS: Process Pinning](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference/process-pinning.html)*总览，引出下面两条链接*

[Intel(R) MPI Library Developer Reference for Linux* OS: Interoperability with OpenMP* API](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference/process-pinning/interoperability-with-openmp.html)*缓存共享、socket和核心问题*

[Intel(R) MPI Library Developer Reference for Linux* OS: Environment Variables for Process Pinning](https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference/process-pinning/environment-variables-for-process-pinning.html)*给出了示例*

[Intel® MPI Library for Linux* OS Developer Reference 2017.pdf](https://scc.ustc.edu.cn/zlsc/tc4600/intel/2017.0.098/mpi/Developer_Reference.pdf)*资料虽然老了点，但是还能看*

[知乎：Linux内核101 NUMA架构](https://zhuanlan.zhihu.com/p/62795773)*读懂文档中的一些参数，会需要额外一些的知识*



## benchmark

*测量进程间通信的延时和带宽*

```bash
#测试节点间/节点内的通信延时
mpirun -np 2 -hosts cas218,cas256 ./osu_latency_mp
mpirun -np 2 -host cas218 ./osu_latency_mp

#测试节点间/节点内的通信带宽
mpirun -np 2 -hosts cas218,cas256 ./osu_bw
mpirun -np 2 -host cas218 ./osu_bw

#上述输出的数据的SIZE单位都是MB（？
```

[OSU InfiniBand Network Analysis and Monitoring Tool v0.9.6 User Guide](http://mvapich.cse.ohio-state.edu/userguide/osu-inam/#_overview_of_the_osu_inam_project)

[OSU Micro-Benchmarks 5.7](https://mvapich.cse.ohio-state.edu/benchmarks/)

[OMB (OSU Micro-Benchmarks) README](https://mvapich.cse.ohio-state.edu/static/media/mvapich/README-OMB.txt)
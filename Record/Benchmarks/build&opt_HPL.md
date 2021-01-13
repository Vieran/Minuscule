# HPL的运行和优化

*运行环境：pi2.0集群；gcc版本为9.3.0*



## 下载和解压HPL

*这里记录的是基本的操作步骤，详细的编译方案在后面*

```bash
#下载源码并解压
wget http://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar zxvf hpl-2.3.tar.gz

#进入hpl安装包
cd hpl-2.3

#剩下的步骤请根据编译方案具体操作

#注意：make之后，如果失败了，对应的clean命令再重新make（在Makefile里面可以找到对应的定义
make clean_arch_all arch=yyy #清理arch=yyy的编译文件
```



## 编译方案1

**openMPI-4.0.5和openBLAS-0.3.12**

#### 修改文件并编译

```bash
#复用文件
cp setup/Make.Linux_PII_CBLAS_gm ./Make.cblas_gm

#修改参数
#TOPdir参数改为hpl的路径
#LAdir参数改为openblas的安装路径
#LAlib参数改为$(LAdir)/lib/libopenblas.a（只有这一个库了
#HPL_OPTS添加参数-DHPL_PROGRESS_REPORT（允许细节输出
#最后记得将mpi/bin的路径输出到PATH再编译

#编译和运行
make arch=cblas_gm
cd bin/cblas_gm
vim HPL.dat #修改参数
mpirun -np 40 -x OMP_NUM_THREADS=1 ./xhpl | tee hpl.out
#-np是设置进程数目，必须大于等于P*Q
#-x是将将环境变量export，而且只能指定一个，这里是将线程数限定为1
```

#### 结果分析

1. 除开N、NB、P、Q之外，其他参数对结果的影响都不大
2. 最佳结果为：N=139776，NB=384，P=4，Q=10，PFACTS=1，NBMINS=2，NDIVS=2，RFACTS=0，BCASTS=0，DEPTHS=0，SWAP=2，L1=0，U=0，其他参数取默认值，取得Gflops=1.8747E+03



## 编译方案2

**Intel套件**

#### 修改文件并编译

```bash
#复用文件
cp setup/Make.Linux_Intel64 ./Make.intel

#修改参数
#TOPdir参数改为hpl的路径
#OMP_DEFS参数改为-qopenmp
#加载Intel套件（也可以自己安装一套
module load intel-parallel-studio/cluster.2020.1-intel-19.1.1

#编译和运行
make arch=intel
cd bin/intel
vim HPL.dat #修改参数
mpirun -np 40 ./xhpl | tee hpl.out
#-np是设置进程数目，必须大于等于P*Q
```

#### 结果分析

1. 除开N、NB、P、Q之外，其他参数对结果的影响都不大
2. 最佳结果为：N=139776，NB=384，P=4，Q=10，PFACTS=1，NBMINS=2，NDIVS=2，RFACTS=0，BCASTS=0，DEPTHS=0，SWAP=2，L1=0，U=0，其他参数取默认值，得到Gflops=2.0370E+03
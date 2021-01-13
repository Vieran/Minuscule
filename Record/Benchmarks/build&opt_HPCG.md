# HPCG的运行和优化

*运行环境：pi2.0集群；gcc版本为9.3.0*



## 下载和解压HPCG

```bash
#版本3.1.0
#到GitHub下载安装包并解压
wget https://github.com/hpcg-benchmark/hpcg/archive/HPCG-release-3-1-0.tar.gz
tar -zxvf HPCG-release-3-1-0.tar.gz -C ../benchmarks/
cd ../benchmarks/hpcg-HPCG-release-3-1-0

#创建编译用的文件夹
mkdir build
make #最后在bin文件夹下可以找到生成的文件

#运行
cd bin
vim hpcg.dat #修改文件内的参数
mpirun -np 8 xhpcg #这样会按照文件内的参数运行
mpirun -np 8 xhpcg 32 24 16 #指定nx=32，ny=24，nz=16，时间按照文件中的
mpirun -np 4 xhpcg --nx=16 --rt=1800 #指定nx=ny=nz=16，时间为1800s
```



## 编译方案1

**openMPI**

*openmpi版本为4.0.5，安装说明见build&opt_HPL.md*

#### 编译

```bash
#拷贝提供的复用文件，准备编译
cd setup/
cp Make.MPI_ICPC Make.openmpi

#进入build文件夹编译
cd build
../configure openmpi
make
```

#### 结果分析

修改x、y、z的值或者改变进程数目（np），得到的结果基本无差别，都是在25Gflops附近（这是怎么回事？？？）

| xyz  | num_of_nodes | memory_occupation | Gflops  |
| ---- | ------------ | ----------------- | ------- |
| 160  | 1            | 139/187           | 25.8886 |
| 168  | 1            | 159/187           | 25.6146 |
| 172  | 1            | -                 | aborted |
| 152  | 1            | 120/187           | 25.5871 |
| 144  | 1            | 103/187           | 25.6635 |



## 编译方案2

**Intel**

#### 编译

```bash
#拷贝提供的复用文件，准备编译
cd setup/
cp Make.MPIICPC_OMP Make.intel

#加载Intel套件（也可以使用自己安装的
module load intel-parallel-studio/cluster.2020.1-intel-19.1.1

#进入build文件夹编译
cd build
../configure intel
make
```

#### 结果分析

| xyz  | num_of_nodes | memory_occupation | Gflops  |
| ---- | ------------ | ----------------- | ------- |
| 160  | 3            | 139/187           | 76.7336 |
| 168  | 3            | 160/187           | 76.4352 |
| 172  | 3            | 183/187           | 75.8582 |
| 152  | 3            | 123/187           | 76.4156 |
| 144  | 3            | 104/187           | 76.1329 |
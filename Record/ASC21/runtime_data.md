# 程序运行时间记录

```bash
#将profile的标准输出和标准错误输出都重定向到文件xxx
nvprof ./demo &>xxx
#2>&1 意思是把“标准错误输出”重定向到“标准输出”
```



## 多节点数据

```bash
#3节点*40核心，random算例（16G），优化版本
#baseline：58.46s
mpirun -np 128 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #47.42s；这个或许是无效的，因为核心没那么多
mpirun -np 64 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #44.47s
mpirun -np 32 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #41.56s
mpirun -np 16 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #41.09s
mpirun -np 8 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #40.65s
mpirun -np 4 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #49.30s
mpirun -np 2 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #63.42s
mpirun -np 1 -machinefile mf -genv I_MPI_DEBUG=5 ./demo #59.81s

#8节点*40核心，交换包大小取决于进程数目，交换次数取决于门和target（若交换，必为n/2次）
#由于是random算例，无法统计交换次数（或者说，统计这个很麻烦，得写个脚本）；若需要统计，得造一个更加规则的算例再测
mpirun -machinefile mf -genv I_MPI_DEBUG=5 -n 1 ./demo
#1，无交换，59.67s
#2，交换大小为8G，67.48s
#4，交换大小为4G，45.78s
#8，交换大小为1G，28.23s
#16，交换大小为512M，19.61s
#32，交换大小为256M，18.85s
#64，交换大小为128M，16.37s
#128，交换大小为64M，15.72s
#256，交换大小为32M，15.29s
#512，交换大小为16M，16.60s
```

**一些总结**

1. machinefile分配的进程可能是不均匀的，需要自己查看具体到底是怎么分配的
2. 从3*40那组结果看，似乎看不出来什么
3. 从8*40那组结果看，在允许的情况下（内存够用），进程越多，似乎时间越短
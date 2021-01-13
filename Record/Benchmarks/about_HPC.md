# A Brief Introduction To HPC

## Before Start

### What is HPC ?

> High Performance Computing的缩写
>
> 指利用聚集起来的计算能力来处理标准工作站无法完成的数据密集型计算任务



### What is Linpack ?

> Linear system package的缩写
>
> 流行的用于**测试HPC浮点性能**的benchmark（基准）
>
> 通过对高性能计算机采用高斯消元法求解一元N次稠密线性代数方程组，评价计算机的浮点性能
>
> 包括Linpack100、Linpack1000和HPL这三类，而常用的是HPL（除此之外，HPCG也很常用，在后面都会介绍到）



### What is FLOPS ?

> floating-point operations per second的缩写，计算机每秒钟能完成的浮点运算的最大次数
>
> 这是是衡量计算机性能的一个重要指标（但是不是唯一的）
>
> **理论浮点峰值** = CPU主频 × CPU每个时钟周期执行浮点运算的次数 × 系统中CPU核心数目
>
> **实际浮点性能** = 计算量 / 计算时间 *计算量 = 2/3\*N^3 - 2\*N^2



## Introduction of HPL

### What is HPL ?

> High Performance Linpack的缩写
>
> 它封装了测试和定时的程序，用以量化获得的计算机计算的准确性以及计算解决方案所用的时间
>
> 编译HPL会生成HPL.dat和xhpl这两个文件，其中HPL.dat是参数文件，xhpl是可执行文件
>
> 通过运行可执行文件xhpl，会得到一些输出，其中就包括“计算机浮点运算的最大次数”，也就是FLOPS（输出实际上是用了Gflops作为单位，也就是10^9FLOPS)，这个结果越大，证明计算机的性能浮点运算性能越好



### What is the principal of HPL ?

> LU分解（原来是Ax=b，使用A=LU分解后是LUx=b，Ly=b，Ux=y得到x；U，上三角矩阵，通过高斯消元得到；L单位下三角矩阵，原矩阵对应位除以所在列的对应在U上的主元（对角线上那个数字））



### How to optimize ?

> 1. 调节HPL.dat中的参数
> 2. 编译中的优化
> 3. 阅读并优化HPL的源码
> 4. GPU（？



#### 调节HPL.dat中的参数

**文件中的参数很多，一些对结果的影响比较大，而另一些对结果的影响比较小。这里着重介绍几个对结果影响比较大的参数**

##### N

> 这个参数指定了测试矩阵的规模（阶数）
>
> 一般，用物理内存容量（单位：byte）的80%~85%来进行HPL的运算，剩余内存用于保证系统中的其他程序正常运行（注意：一个双精度数占8个字节），即：N x N x 8 = 系统总内存 x 80%

##### NB

> 这个参数指定了矩阵分块的规模（阶数），这样系数矩阵被分成NB x NB的循环块被分配到各个进程当中去处理
>
> NB的最优值主要通过实际测量来得到（在实验中，尽量选取2的整数幂的数字进行测量）

##### P和Q
> 这两个参数指定了二维网格的行数和列数，一般它们的选取遵循以下规则
>
> 1. P × Q = 系统CPU总核数 = 进程数
>
> 2. P ≤ Q
>
>    *一般来说，P的值尽量取得小一点，因为列向通信量（通信次数和通信数据量）要远大于横向通信，P最好选择2的幂*



#### 编译中的优化

> *主要是尝试了使用不同的库来编译，下面这两种编译方案的完整操作步骤记录在文件buld&opt_HPL.md中*
>
> 1. openMPI+openBLAS
> 2. Intel
>

*最佳结果：N=139776，NB=384，P=4，Q=10，PFACTS=1，NBMINS=2，NDIVS=2，RFACTS=0，BCASTS=0，DEPTHS=0，SWAP=2，L1=0，U=0，其他参数取默认值；在方案1中得到1.8747E+03，在方案2中得到2.0370E+03*

#### 阅读并优化HPL源码

> 暂时留白，待寒假期间补充（
>
> *一大堆文献等着阅读（OneDrive+EndNote）*



### Tips

> 1. 在Make.xxx文件里面开启细节输出progress_report再编译。这样可以在只运行一部分就可大致预估结果会如何，效率比较高
> 2. 对于HPL的其他参数，可以阅读HPL官网的介绍以及参考资料中的内容
> 3. 计算机浮点峰值的计算，也可以阅读参考资料中的内容



## Introduction of HPCG

### What is it ?

> High Performance Conjugate Gradients的缩写
>
> 旨在补充高性能 HPL 基准，HPCG更能够从计算、节约能源等上说明超算性能，相对于HPL而言，HPCG测试成绩往往较低
>
> 编译HPCG会生成hpcg.dat和xhpcg这两个文件，其中hpcg.dat是参数文件，xhpcg是可执行文件
>
> 通过运行可执行文件xhpcg，会得到两个输出文件，文件中同样给出了“计算机浮点运算的最大次数”，也就是FLOPS（输出实际上是用了Gflops作为单位，也就是10^9FLOPS)，这个结果越大，证明计算机的性能浮点运算性能越好



### How to optimize ?

> 1. 调节hpcg.dat中的参数
> 2. 编译中的优化
> 3. 阅读并优化HPCG源码
> 4. GPU（？



#### 调节hpcg.dat中的参数

**文件中的参数共4个，前面三个（x、y、z）指定计算的规模，最后一个指定计算的时间**

>  参数x、y、z都要可以被8整除
>
>  时间：官方要求必须不小于1800s，结果才有效



#### 编译中的优化

> *主要是尝试了使用不同的软件来编译，下面这两种编译方案的完整操作步骤记录在文件buld&opt_HPL.md中*
>
> 1. openmpi
> 2. Intel

*最佳结果：方案1中，x=y=z=144，time=60，Gflops=25.7886；方案2中*



#### 阅读并优化HPCG源码

> 暂时留白，待寒假补充
>
> *暂未开始寻找文献，但是应该懂得hpl就会对这个有更好的理解了*



## Reference

[HPL官网](http://www.netlib.org/benchmark/hpl/)

[HPCG官网](http://hpcg-benchmark.org/)

[GitHub上一个比较完整的关于HPL参数的设置介绍](https://github.com/kyrie2333/Linpack-HPL)

[某博客上的对于hpcg的介绍](https://enigmahuang.me/2017/12/27/HPCG_3_Notes/)

[计算CPU浮点运算峰值](https://www.jianshu.com/p/b9d7126b08cc)




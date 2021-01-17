# Working With Intel

*环境：pi2.0集群*

## Intel parallel studio

官网：[Intel® Parallel Studio XE](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html)

文档：/Reference/documents/intel-parallel-studio_2020.4.011

**什么是Intel parallel studio ？**

> Intel打造的一套开发工具，包括编译器（Intel编译器）、函数库（mkl等）以及一些数据分析器（vtune等）
>
> 这一套工具是经过优化的，性能比一般的编译器和函数库要好

**下载安装Intl parallel studio**

```bash
#到官网下载安装包并解压（需要注册，30天试用
tar -zxvf intel -C yyy

#然后到HPC Studio上进行安装
cd yyy
./install_GUI.sh

#然后只需要根据提示进行操作即可，最后的安装目录注意一下
#建议下载安装手册，阅读手册再安装
```



## Intel编译器优化---Linux

官方文档pdf：/Reference/documents/intel-compilers-opt-v19-1.pdf

官方文档doc：/Reference/documents/intel-compilers-opt-v19-1.doc

*详细的内容，建议直接看文档，pdf转换成doc了，可以拿去翻译*

[某博客：基于intel编译器的优化](https://blog.csdn.net/honey_yyang/article/details/7849013)

### 优化级别

| 参数    | 功能                                                         |
| ------- | ------------------------------------------------------------ |
| -O0     | 不优化                                                       |
| -O1/-Os | 优化尺寸（使得生成的代码尽可能少                             |
| -O2     | 最大化速度（默认的设置是这个；矢量化+文件内过程间优化        |
| -O3     | 同时启用-O1和-O2（优化循环；推荐用于执行许多浮点计算和处理循环的程序；不是所有的程序都适用 |

### 常见的一些优化参数

| 参数      | 功能                                                 |
| --------- | ---------------------------------------------------- |
| -qopenmp  | 对于openmp，生成多线程代码（仅仅对于Fortran管用      |
| -mkl=name | 链接到intel的mkl库                                   |
| -parallel | 自动检测并执行简单的结构化循环，并自动生成多线程代码 |



## MPI

```bash
#mpiexec、mpirun等的区别？
#多节点运行程序，hostfile默认使用换行符进行分割
#-f和-hostlist使用方法

mpiexec.hydra #比mpirun更加高效
```


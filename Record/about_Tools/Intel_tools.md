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



## VTune

**什么是VTune ？**

> 收集和分析系统行为的数据
>
> 帮助分析优化瓶颈，查看应用运行时候的资源占用情况（Linux中类似的工具还有perf、gprof
>
> 用户手册：[Intel® VTune™ Profiler User Guide](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html)
>
> 关于如何理解分析结果：[VTune的功能和用法介绍](https://blog.csdn.net/wy_stutdy/article/details/79106501)（在官方文档里面应该也有对应的内容

### Get Start

```bash
#开启vtune
source Intel-parallel-studio/vtune_profiler/env/vars.sh #加载环境变量的脚本（或者直接加载~/cyJ/script/intel-parallel-studio.sh，这样就整个Intel套件都可以使用了
vtune-gui #开启图形化界面（需要在hpc studio或者可以开启图形界面的应用打开
vtune #输出命令行参数帮助
#不会用的时候，开图形化界面，然后点击Help Tutor，跟着提示学一学
#直接使用configure analysis，会出现界面空白的问题（暂未解决，可以用命令行跑了再去打开结果进行分析

#初步尝试使用vtune分析一个程序的运行（机测1的amg题目；结果放在了result文件夹下
vtune -collect hotspots -result-dir result -quiet ./amg -problem 1 -P 1 1 1 -n 10 10 10 -printallstats

#获得简要的分析总结
vtune -help collect performance-snapshot #查看这条指令相关的手册
vtune -collect threading ./amg -problem 1 -P 1 1 1 -n 10 10 10 -printallstats #再次尝试了一次分析
```

分析结果都打印在屏幕上了，生成了一些文件，但是不知道那些文件怎么看？输出的分析结果如何去理解？

---官方文件里面都有详细的解释（如果是中文就更好了



## 其他

```bash
mpiexec.hydra #Intel的mpiexec可能比一般的软件包的mpirun更加高效
#Intel的mpi使用，详细见about_MPI.md
```


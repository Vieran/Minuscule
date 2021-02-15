# PROFILE

*简单介绍几种常见的profile工具及其使用方法*

## perf

*采样跟踪*

```bash
#几种可以查看的perf工具
man perf #这个手册最后会指出下面这些指令可以查看
man perf-top
man perf-stat
man perf-record
man perf-report
man perf-list

#查看当前系统支持的性能事件
perf list #列出来的事件在其他命令下可以通过-e跟踪，多个事件使用,分隔

#分析整体情况
perf stat ./demo

#进一步指定事件进行采样
perf record -g ./demo

#读取报告（还有很多其他的参数可以使用
perf report -i perf.data

#对系统性能进行健全性测试（然而这一大堆输出，看不懂是哪些参数
perf test

#实时显示系统/进程的性能统计信息（需要权限
perf top
```

[perf命令手册](http://linux.51yip.com/search/perf)

[perf使用指南](https://developer.aliyun.com/article/65255)



## gprof

*采样和插入代码，默认不支持多线程*

```bash
#简单使用，编译时候需要加-pg选项
gcc -pg xxx.c -o xxx.x
./xxx.x #程序运行结束后目录下生成了一个gmon.out文件

#查看分析结果
gprof demo gmon.out <选项>
```

[gprof的使用](https://www.cnblogs.com/youxin/p/7988479.html)



## vtune

*Intel-parallel-studio的工具*

```bash
#开启vtune
source ~/cyJ/WorkStation/Intel-parallel-studio/parallel_studio_xe_2020/bin/psxevars.sh
vtune-gui #开启图形化界面（需要在hpc studio或者可以开启图形界面的应用打开
vtune #输出命令行参数帮助

#初步尝试使用vtune分析一个程序的运行
vtune -collect hotspots -result-dir result -quiet ./amg -problem 1 -P 1 1 1 -n 10 10 10 -printallstats

#获得简要的分析总结
vtune -help collect performance-snapshot #查看这条指令相关的手册
```


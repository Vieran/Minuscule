# Profile Tools

*介绍一些常用的profile工具及其使用方法*

## VTune

> 收集和分析系统行为的数据
>
> 帮助分析优化瓶颈，查看应用运行时候的资源占用情况（Linux中类似的工具还有perf、gprof
>
> 用户手册：[Intel® VTune™ Profiler User Guide](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html)
>
> 关于如何理解分析结果：[VTune的功能和用法介绍](https://blog.csdn.net/wy_stutdy/article/details/79106501)（在官方文档里面应该也有对应的内容

```bash
#开启vtune
source Intel-parallel-studio/vtune_profiler/env/vars.sh #加载环境变量的脚本（或者直接加载~/cyJ/script/intel-parallel-studio.sh，这样就整个Intel套件都可以使用了
vtune-gui #开启图形化界面（需要在hpc studio或者可以开启图形界面的应用打开
vtune #输出命令行参数帮助
#不会用的时候，开图形化界面，然后点击Help Tutor，跟着提示学一学
#直接使用configure analysis，会出现界面空白的问题（暂未解决，可以用命令行跑了再去打开结果进行分析

#初步尝试使用vtune分析一个程序的运行（机测1的amg题目；结果放在了result文件夹下
vtune -collect hotspots -result-dir result -quiet ./amg -problem 1 -P 1 1 1 -n 10 10 10 -printallstats

#使用vtune获得全部分析
vtune -collect hpc-performance -result-dir result -quiet ./random.x

#获得简要的分析总结
vtune -help collect performance-snapshot #查看这条指令相关的手册
vtune -collect threading ./amg -problem 1 -P 1 1 1 -n 10 10 10 -printallstats #再次尝试了一次分析
```



## Perf

> Linux自带的性能分析工具

```bash
#必须root权限才能安装













```


# Vtune分析运行结果

## 执行vtune分析

*环境：Intel-parallel-studio-2020*

```bash
#进入到可执行文件所在的目录
cd /lustre/home/acct-hpc/asc/cyJ/asc21/QuEST-2.1.0/build_random

#加载vtune
source ~/cyJ/script/intel-parallel-studio_env.sh
vtune #检查vtune是否加载，查看可选的参数

#编译的时候加-g选项

#执行vtune分析hpc performance并将结果存储到result文件夹下
vtune -collect hpc-performance -result-dir result -quiet ./random.x

#打开vtune的图形化界面并在打开结果，找到上面生成的result文件夹下的后缀名为vtune的文件并打开
vtune-gui
```



## 查看分析结果

### random

```bash
vtune -collect hpc-performance -result-dir ~/cyJ/vtune_files/quest/random/base-intel-hs -quiet ./demo
```

random的瓶颈在哪里？



### GHZ_QFT

```bash
vtune -collect hpc-performance -result-dir ~/cyJ/vtune_files/quest/ghzqft/opt-intel-hs -quiet ./demo
```

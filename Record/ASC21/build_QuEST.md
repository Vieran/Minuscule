# get start

*对于QuEST的基本理解和初始的编译运行*



### 基本理解

> 全称：Quantum Exact Simulation Toolkit（量子精确仿真工具包）
>
> 性质：一套用cpp和c写的模仿量子行为的工具包（官方说法是高性能模拟器）
>
> 功能：可以模拟混合状态的退相干，应用具有任意数量的控制量子位的门（看起来是和量子物理行为相关的）



### 进一步理解

#### 入门知识阅读

阅读[写给开发者的量子计算入门教程](https://github.com/swardsman/learning-q-sharp)

##### 量子基础知识

1. 量子计算中信息存储和处理的基本单位是qubit（量子比特，Quantum Bit）；不同于传统的bit，qubit同时处于0和1的状态

2. 描述量子态的二维向量模长必须为1；一个量子的所有状态构成了一个量子空间，(1,0)和(0,1)是这个空间的一组基底，同时也对应经典计算的1和0；量子计算中使用Dirac符号来表示这一组基态矢量



##### 量子门

1. 量子门输入和输出的qubit数量相等
2. 每个量子门的操作对应一个矩阵（X、Y、Z、H、S、T），常见的这几种量子门的计算会经常出现
3. 一个量子门对应一个矩阵，这个矩阵必须是**幺正矩阵**（n维复方阵的共轭转置矩阵与其逆矩阵相等），因为量子态的演变必是幺正的



##### Dirac符号

*这一个章节的内容可以通过搜索引擎得到，或者直接阅读原文的html版本*

##### 其他的章节

*看起来是数学物理部分的内容，我想，应该不是先学这些，这部分可以在实践之后用于加强理解*

##### 量子计算开发

*内容主要是学习Q#语言开发，基本语法看起来和cpp差不多，主要内容是后面的qubit*



### 简单的下载和编译

```bash
#环境：pi2.0，节点cas452

#下载库和算例后解压（指定的是2.1.0版本，所以使用了scp将文件传过去
tar -zxvf QuEST-2.1.0.tar.gz
unzip ASC_QuEST赛题算例.zip

#创建文件夹进行编译
mkdir build
cd build
camke ..
#如果下载的是最新版本，需要指定c99
#cmake -DCMAKE_C_FLAGS="-std=c99" ..
make

#运行文件（然后输出环境介绍以及电路输出，注意到同一个文件运行多次的输出不完全相同
./demo
```



#### 测试算例

```bash
#将测试算例复制到QuEST目录下（此处question是存放算例的文件夹
cp ../qestion/mytimer.hpp ../qestion/GHZ_QFT.c ../qestion/random.c .

#此时因为测试算例里面的例子是需要使用c99的，所以必须指定c99
cd build
make clean #清理之前的编译内容
rm -rf * #删除cmake产生的文件

#在CMakelists文件使用编译选项，避免每次编译都要修改cmake的文件
#在原来的11和12行（大概这个位置？），修改为下面部分
##############################################以下为文件内容##################################################
#20201128T0933.add this for testing questions
#ifdef RANDOM
set(USER_SOURCE  "random.c"  CACHE STRING "Source to build with QuEST library")
set(OUTPUT_EXE   "random.x"  CACHE STRING "Executable to compile to")
#ifdef ghzqft
set(USER_SOURCE  "GHZ_QFT.c"  CACHE STRING "Source to build with QuEST library")
set(OUTPUT_EXE   "GHZ_QFT.x"  CACHE STRING "Executable to compile to")
#else
set(USER_SOURCE  "tutorial_example.c"  CACHE STRING "Source to build with QuEST library")
set(OUTPUT_EXE   "demo"  CACHE STRING "Executable to compile to")
#endif
############################################################################################################

#测试random.c
mkdir build_random
cd build_random
cmake -DCMAKE_C_FLAGS="-std=c99" -RANDOM ..
make
./random.x
#跑了三次，得到结果分别为99.829888 seconds、99.178968 seconds、99.027662 seconds
#检查probs.dat文件，对比官方给的文件，无差异，保证了正确性

#回到上层目录
cd ..

#测试GHZ_QFT.c
mkdir build_ghzqft
cd build_ghzqft
cmake -DCMAKE_C_FLAGS="-std=c99" -ghzqft ..
make
./GHZ_QFT.x
#跑了三次，得到结果分别为100.337823 seconds、98.235631 seconds、98.586789 seconds
##检查probs.dat文件，对比官方给的文件，无差异，保证了正确性
```

**小插曲**

> 在测试GHZ_QFT的时候，本来将宏命令设置为GHZ_QFT（和RANDOM统一）
>
> 但是”cmake -GHZ_QFT -DCMAKE_C_FLAGS="-std=c99" ..“执行它会报错“cannot create named generator HZ_QFT”，也就是说，它自动把G忽略了
>
> 经过查询-G和generator这两个关键字，得到了一些结果，确实-G就是一个关键字，而后面的HZ_QFT被当成了它要generate的对象了，所以就出现了奇怪的报错。但是这样的理解是存在问题的，因为一般的关键字后面都要有一个空格作为分隔符，然后才会写后续的参数，但是这里明明没有空格，难不成是因为cmake的语法是允许参数和选项之间不带空格的？



### 已优化的下载和编译

```bash
#下载并进入文件夹
git clone https://github.com/Lithiumcr/asc21-quest.git
cd asc21-quest

#切换到分支newhack
git checkout -b optimized origin/newhack
git branch -a #确认目前的分支是不是optimized分支

#加载Intel套件
module load intel-parallel-studio/cluster.2020.1-intel-19.1.1

#编译
CFLAGS=-g CC=icc ./configure random.c build_r #编译random.c生成的文件在build_r文件夹下
CFLAGS=-g CC=icc ./configure GHZ_QFT.c build_g #编译GHZ_QFT.c生成的文件在build_g文件夹下

#运行random算例（对于GHC_QFT.c，进入build_g即可，后面两个语句一样
cd build_r #这里是进入编译random.c生成的文件夹
./demo #运行可执行文件（然后会输出一个时间，即程序运行时间
./diff.sh #运行demo之后会生成两个文件，需要标准文件进行对比，这里直接运行这个脚本就可以了（输出语句，为identical即正确

#也可以把CMakeLists.txt和configure复制到初始版本那里，然后执行同样的命令进行编译（前提是那几个官方文件存在
```


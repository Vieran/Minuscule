# QuEST

## What is it

> 全称：Quantum Exact Simulation Toolkit（量子精确仿真工具包）
>
> 性质：一套用cpp和c写的模仿量子行为的工具包（官方说法是高性能模拟器）
>
> 初赛要求：给定两个算例（random.c和GHZ_QFT.c，都是30量子必特）要求使用QuEST的2.1.0版本编译并运行，在结果正确的情况下（给了两个检验输出的文件），运行时间越短越好
>
> 决赛预测：更大的量子比特位



## 工作进展

### 代码结构分析

> n量子比特的系统占用内存约为$2^{n+4}$Byte（算例的30量子比特占内存约为16G，MPI版本的占内存翻倍
>
> 代码的函数调用树以及对算例使用门的统计：[贾淞淋：QuEST门实现](https://sjtueducn-my.sharepoint.com/personal/keymorrislane_sjtu_edu_cn/Documents/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97%E7%AB%9E%E8%B5%9B/ASC21/%E5%91%A8%E6%8A%A5/20200121_%E8%B4%BE%E6%B7%9E%E6%B7%8B_QuEST%E7%AE%97%E4%BE%8B%E7%BB%9F%E8%AE%A1%E3%80%81%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%84%E3%80%81%E5%B9%B6%E8%A1%8C%E5%AE%9E%E7%8E%B0.pdf?CT=1611534377029&OR=ItemsView)

```c
//初赛两个算例的代码结构如下
//1.创建环境并定义量子系统的量子比特数量
QuESTEnv Env = createQuESTEnv();
Qureg q = createQureg(30, Env);
//2.使用各种量子门对前面定义的量子系统进行操作（运算）
//random算例的门类型较多且顺序随机，GHZ_QFT算例的门类型较少而且顺序规则（连着几个都是同一个门的操作
hadamard(q, 0);
controlledNot(q, 0, 1);
...;
//3.测量量子系统状态并终止环境
q_measure[i] = calcProbOfOutcome(q, i, 1);
Complex amp = getAmp(q, i);
destroyQureg(q, Env);
destroyQuESTEnv(Env);
```



## 运行方式

### 初始版本

```bash
#下载QuEST-2.1.0和初赛算例并解压（初赛算例，自行下载
wget https://github.com/QuEST-Kit/QuEST/archive/2.1.0.tar.gz
tar -zxvf QuEST-2.1.0.tar.gz
cd QuEST-2.1.0
#把初赛算例放在这个文件夹下即可

#测试random.c
mkdir build_random
cd build_random
#cpu版本
cmake -DCMAKE_C_FLAGS="-std=c99" -DUSER_SOURCE="random.c" ..
#gpu版本
cmake -DGPU_COMPUTE_CAPABILITY=70 -DGPUACCELERATED=1 -DUSER_SOURCE="random.c" ..
make
./random.x #输出即为程序的运行时间

#对于GHZ_QFT算例，测试方式同上，只需要将对应的.c文件更改未GHZ_QFT.c即可
#运行结束后，记得检查输出文件是否和官方给定的检测文件相同，以检验正确性
```



### 优化版本

```bash
#下载并进入文件夹
git clone https://github.com/Lithiumcr/asc21-quest.git
cd asc21-quest

#切换到分支newhack
git switch newhack

#cpu版本编译（加载Intel套件，如果需要指定编译选项，请到configure文件中找到cmake参数写入
CC=icc ./configure random.c build_r #编译random.c生成的文件在build_r文件夹下
CC=icc ./configure GHZ_QFT.c build_g #编译GHZ_QFT.c生成的文件在build_g文件夹下

#gpu版本编译（加载cuda，修改configure文件为nv_configure文件，修改如下
#-cmake .. -DUSER_SOURCE="$SOURCEFN" -DVERBOSE_CMAKE=ON -DDISTRIBUTED=1
#+cmake .. -DUSER_SOURCE="$SOURCEFN" -DGPU_COMPUTE_CAPABILITY=70 -DGPUACCELERATED=1
./nv_configure random.c nv_r
./nv_configure GHZ_QFT.c nv_r

#运行random算例（对于GHC_QFT.c，进入build_g即可，后面两个语句一样
cd build_r #这里是进入编译random.c生成的文件夹
./demo #运行可执行文件
./diff.sh #输出语句为identical即正确
```


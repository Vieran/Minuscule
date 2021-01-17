# Tips

### [PI 的使用](https://docs.hpc.sjtu.edu.cn/)

1. login之后申请节点再作业
2. 在进入节点前打开tmux（防止跑分的时候tmux占用资源
3. 作业结束后记得释放节点

```bash
#相关的操作命令
#显示当前用户可用节点（输出中各个参数的意义可以根据英文理解
scontrol show res

#显示当前用户可用节点的详细信息
sinfo

#根据自己的需求参数选定一个节点（满足条件的节点中随机分配），并且命名为xxx
salloc -N 1 --exclusive --reservation=asc --partition=cpu --job-name=xxx

#分配特定的节点xxx（两种方法，一种是全写，一种是缩写
salloc -N 1 --exclusive --reservation=asc --partition=cpu --nodelist=xxx
salloc -N 1 --exclusive --reservation=asc --partition=cpu -w xxx

#排除特定的节点
salloc -N 1 --exclusive --reservation=asc --partition=cpu --exclude=xxx

#查看节点xxx的状态
scontrol show node xxx

#查看每个节点具体在运行什么程序
scontrol show job
sjstat

#得到分配节点名称xxx后，ssh过去
ssh xxx

#在使用module load加载了很多环境变量之后，应该清理了加载其他的环境变量防止出错
module load xxx yyy #加载xxx和yyy模块
module purge #清空模块

#释放节点（查看当前节点名称->释放JOBID为yyy的节点
squeue
scancel yyy

#写脚本通过sbatch提交作业，详细的可以参考~/Test/Test1/brief.txt的内容
sbatch ./xxx.slurm

#SCP传输本地文件目录xxx到服务器上的yyy（服务器到本地只需要将两个参数反过来就可以
#注意：都是在本地的shell输入下列命令，这样才能找到本地文件！
scp -r xxx xyz@202.111.111.111:~/cyJ/work_station/yyy

#sbatch脚本将jobname附加到输出文件，且将输出文件放到指定的文件夹./outputfile/%j_%x.out（j是jobid，x是jobname
```



### 编译安装

```bash
#检查相关依赖，设置安装路径（不需要cc或者gcc
./configure --prefix=/path/to/destdir

#从Makefile中读取指令然后编译
make

#从Makefile中读取指令然后安装（可以指定路径
make install
make DESTDIR=/install/directory install

#静态库是编译的时候嵌入，编译一旦通过，即使.a文件不在了，也可以正常跑程序
#如果是动态库，编译的时候并不会把它放进文件内，每次运行的时候会再次寻找.so文件
```



### 查看服务器配置信息

```bash
#操作系统发行版本详细信息
lsb_release -a
#输出如下
LSB Version:	:core-4.1-amd64:core-4.1-noarch
Distributor ID:	CentOS
Description:	CentOS Linux release 7.7.1908 (Core)
Release:	7.7.1908
Codename:	Core

#查看cpu信息
lscpu

#查看cpu型号（开头那个数字： 逻辑cpu的个数 = 物理cpu × 每个cpu的核数
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
# 40  Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz

#查看物理cpu个数（输出一个数字
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

#查看每个物理cpu的core的个数（这个在申请节点之后查看，输出是一个数字
cat /proc/cpuinfo| grep "cpu cores"| uniq

#服务器型号（这个居然要sudo权限
dmidecode|grep "System Information" -A9|egrep  "Manufacturer|Product"

#硬盘分区
lsblk
fdisk -l
```

[CSDN上关于上述指令的博客](https://blog.csdn.net/u011636440/article/details/78611838)



### vim

```bash
#vim的全局搜索：在当前文件夹下查找所有的包含xxx的文件，并且设置可以跳转
:vim /xxx/** | copen

#不退出vim，直接跳转到命令行执行命令
:!export | grep PATH #相当于在命令行执行export | grep PATH

#命令模式下
u #撤销
CTRL+r #恢复撤销
x #删除当前光标所在位置的单个字符
r #按下r，然后输入新字符来替换光标所在处的单个字符（如果是R则相当于Windows下的insert，直到按esc退出
yw #复制一个单词
y$ #复制到行尾

#强大的g命令，全局的
:g

#查找和替换
:s/old/new/ #跳到old第一次出现的地方，并用new替换
:s/old/new/g #替换所有old
:n,ms/old/new/g #替换行号n和m之间所有old
:%s/old/new/g #替换整个文件中的所有old

#跳转
CTRL+w hjkl #窗格间跳转，hjkl分别是左上下右
CTRL+] #跳转到函数定义处（前提是生成了tags
CTRL+o #向后跳到后几次光标位置（跳到函数之后，再输入这个就可以跳回原处
CTRL+i #向前跳到前几次光标位置
```



### 基础工具相关操作

```bash
#查看当前目录下的文件树
tree -l 2 #展开层数为两层

#diff以逐行的方式比较文本文件的异同处；如果指定要比较目录，则会比较目录中相同文件名的文件，但不会比较其中子目录
diff f1 f2 -y -W 200 #-y按照以并列的方式显示异同之处，-W设置栏宽（和-y参数一起用）为后面的数字（这里是200）
diff d1 d2 #对比两个文件下的内容不同，输出是按照only in xxx：yyy的方式展示的
diff -aNru d1 d3 #-a，Treat all files as text and compare them line-by-line, even if they do not seem to be text；-N，In directory comparison, if a file is found in only one directory, treat it as present but empty in the other directory；-r，When comparing directories, recursively compare any subdirectories found；-u，Use the unified output format

#输出的内容中，|表示前后两个文件不同，<表示后面文件比前面少了一行，>表示后面文件比前面多了一行
#还有一个命令comm也可以实现比较，可以使用man手册查看

nmtui #网络配置

#输出重定向
python test.py > xxx #仅仅输出到文件xxx（覆盖
python test.py | tee xxx #同时输出到屏幕和文件xxx（覆盖
python test.py | tee -a xxx #同时输出到屏幕和文件xxx（追加

#反汇编工具
objdump -d xxx.o #将可执行文件xxx.o反汇编成为汇编语言

ulimit #man bash查看详情

#使用watch每隔10s查看squeue的执行状态
watch -n 10 squeue

gcc -w yyy.c -o yyy.x #不输出警告信息

update-alternatives #可以使用这个命令管理Linux上不同版本的软件，man可查看详情

#使用man可用查看c语言函数的用法（若没有MPI的函数，可以用apt搜索包然后安装上就可以了
man printf
apt search <package name>

#命令行中向上/下翻页
shift+page up/down
```

[求两个Linux文本文件的交集、差集、并集](https://www.cnblogs.com/thatsit/p/6657993.html)、[diff不同输出格式的区别](https://www.cnblogs.com/wangqiguo/p/5793448.html#_label4)



### 其他细节

1. 将操作步骤记录下来，方便查错和回溯

2. 建立设置环境变量的.sh文件，随时执行切换环境，养成管理环境变量的良好习惯

   ```bash
   #!/bin/sh
   APPS=${HOME}/cyJ/WorkStation
   MPI=${APPS}/openmpi
   BLAS=${APP}/openblas
   
   #环境变量以:分隔，放在前面的优先搜索，搜索到了就停止，所以一般新的放前面
   export PATH=${MPI}/bin:$PATH
   export PATH=${BLAS}/bin:$PATH
   export LIBRARY_PATH=${MPI}/lib:$LIBRARY_PATH
   export LD_LIBRARY_PATH=${MPI}/lib:$LD_LIBRARY_PATH
   export CPLUS_INCLUDE_PATH=${MPI}/include:$CPLUS_INCLUDE_PATH
   export C_INCLUDE_PATH=${MPI}/include:$C_INCLUDE_PATH
   export MANPATH=${MPI}/share:$MANPATH #配置man手册（man3是函数的定义和使用方法
   ```

3. 使用CTRL+r可以搜寻执行过的指令



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

#写脚本通过sbatch提交作业
sbatch ./xxx.slurm
#sbatch脚本将jobname附加到输出文件，且将输出文件放到指定的文件夹./outputfile/%j_%x.out（j是jobid，x是jobname

#SCP传输本地文件目录xxx到服务器上的yyy（服务器到本地只需要将两个参数反过来就可以
#注意：都是在本地的shell输入下列命令，这样才能找到本地文件！
scp -r xxx xyz@202.111.111.111:~/cyJ/work_station/yyy

#在桌面环境下，使用ssh -X可以进入可开启图形界面的命令行
ssh -X vol08
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
nvprof ./demo &>xxx #2>&1 意思是把“标准错误输出”重定向到“标准输出”

#反汇编工具
objdump -d xxx.o #将可执行文件xxx.o反汇编成为汇编语言

#运行mpi和openmp任务的时候，不限制内存大小
mpirun –hostfile <hosts_file> | -np <number_of_threads> bash -c "ulimit -s unlimited && our_binary_to_be_executed"
#或者把下列语句放入.bashrc（但是这个参数$PS1是什么？
if [ -z "$PS1" ]; then
       ulimit -s unlimited
fi

#使用watch每隔10s查看squeue的执行状态
watch -n 10 squeue

gcc -w yyy.c -o yyy.x #不输出警告信息

update-alternatives #可以使用这个命令管理Linux上不同版本的软件，man可查看详情

#使用man可用查看c语言函数的用法（若没有MPI的函数，可以用apt搜索包然后安装上就可以了
man printf
apt search <package name>

#命令行中向上/下翻页
shift+page up/down

#快速切换目录（目录栈
pushd
popd

#解压缩软件包
rpm2cpio xxx.rpm |cpio -idvm

#删除xxx环境变量
unset xxx

#打包全部文件，包括隐藏文件
tar zcvf asc21.tar.gz .[!.]* *

#配置免密登录（本地操作
ssh-keygen -R xxx.xxx.xxx #清除之前的key
ssh-keygen
ssh-copy-id user@xxx.xxx.xxx.xxx

#gpg检验工具的使用
gpg --verify gcc-9.3.0.tar.gz.sig gcc-9.3.0.tar.gz
gpg --keyserver keys.gnupg.net --recv-key A328C3A2C3C45C06
gpg --verify gcc-9.3.0.tar.gz.sig gcc-9.3.0.tar.gz

# strings，打印任意文件里面可输出的字符
strings /lib64/libgfortran.so.5.0.0 | grep GFORTRAN

# 递归查找某个文件
find . -name "xxx"

# man手册的使用
man -l xxx # 使用man打开本地的xxx文件而不是去路径上搜索
```

[求两个Linux文本文件的交集、差集、并集](https://www.cnblogs.com/thatsit/p/6657993.html)、[diff不同输出格式的区别](https://www.cnblogs.com/wangqiguo/p/5793448.html#_label4)



### 其他细节

1. 将操作步骤记录下来，方便查错和回溯

2. sshfs挂载远程的Linux文件系统目录，直接在本地改代码

3. 建立设置环境变量的.sh文件，随时执行切换环境，养成管理环境变量的良好习惯

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

4. 使用CTRL+r可以搜寻执行过的指令

5. Windows和Linux协作：GitHub/sftp。只在本地改代码，仓库在本地，然后push到远端运行；或者只在本地改代码改代码，仓库在远端，设置git bash忽略LF和CRLF区别；或者直接在远端写代码。

   ```bash
   git config --global core.autocrlf input
   git config --global core.safecrlf true
   ```

   

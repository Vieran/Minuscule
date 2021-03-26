# Learning GDB

*Linux命令行中的GDB使用*

[Debugging with GDB](https://www.sourceware.org/gdb/current/onlinedocs/gdb.html)

## 常用命令

```bash
#调试程序编译时候应该加上-g选项
#输出信息过多的时候，gdb会暂停输出并打印一些提示文字，禁止这个可以使用
set pagination off

#加载调试的程序xxx（或者可以直接命令行下执行gdb xxx
file xxx

#运行（run）程序（有断点就在断点处停留
r [参数]
start #也可以用这个命令开始运行程序

#继续（continue）执行程序，直到下一个断点/程序结束
c

#设置断点（break）
b [行号/函数名/*代码地址]
b 6 #在第6行设置断点
b func_1 #在函数func_1处设置断点
b *0xFFF111 #在地址0xFFF111处设置断点

#删除（delete）断点
d [编号] #编号从1开始递增
d 3 #删除断点3
d #删除所有断点

#执行
stepi [行数] #执行几条汇编指令，默认是1行；如果有函数会跳入函数（step into，写为step的时候对应几行源码
nexti #执行一条汇编指令；会把函数一并执行完（step over，写为next的时候对应一行源码
finish #回到当前运行的函数
until [行号] #跳至运行到对应的行号处

#查汇编（display assembly）
disas [函数名/地址] #反汇编“函数名/地址附近”的函数，默认是当前函数
disas func_3 #反汇编函数func_3
disas 0xFFF111,0xFFF999 #反汇编0xFFF111,0xFFF999范围内的代码
disas 0xFFF111 #反汇编0xFFF111附近的函数

#查源码
list

#查数据
p [格式] [寄存器/数据] #以某进制（默认十进制）输出（print）寄存器/数据的值
p /x ($rax+8) #以十六进制输出寄存器%rax的内容加上8
p /t 0x1111 #以二进制输出0x1111的值
p *(long*) ($rsp+8) #输出位于$rsp+8处的长整数
p &var #输出var的内存地址（引用
p *var #输出存在于地址var的数值（解引用
x /[num][format][width] <地址> #从地址开始，检查num个字节的内存，使用format打印出来，把内存当成width的值
x /2g 0xFFFF1111 #检查从地址0xFFFF1111开始的双字（8byte）
x /20b func_3 #检查函数func_3的前20个字节

#信息（information）
i [reg,frame,locals,args] #显示有关当前寄存器/栈帧/本地变量/环境变量有关信息（也可以使用info
i frame #有关当前栈帧信息
info registers #所有寄存器的值

#退出
q #退出（quit）gdb
kill #停止程序
```



## TUI快捷键

```bash
#从裸奔到gdb tui
CTRL+x+a
#直接开启tui调试程序xxx
gdb -tui xxx

#在不同窗格之间跳转（源码窗格是编号1
CTRL+x o #先按前面的，然后单独按o会可以在几个窗格之间跳转（其实o可以替换为对应的窗格编号，就像下面那些命令

#恢复窗格的format（有时候可能会变形
CTRL+l

#开个窗格显示全部的汇编指令
CTRL+x 2
#在上面的基础上，开个窗格显示当前所有的寄存器变量（这里是general类型的
CTRL+x 2
#再在上面的基础上，指定显示特定的变量
tui reg [变量类型] #先不输入变量类型，会出现命令行提示所有可用的类型

#查看先前执行的命令
CTRL+p #如果在gdb命令的窗格下，可以直接上下键就跳转命令

#tui的layout指令
layout [next|prev|src|asm|split|regs]

#甚至，你可以在gdb中使用python，就像在shell中使用一样
python print(gdb.breakpoints().[0].location) #打印第一个断点的位置信息
python gdb.Breakpoint('7') #在第7行设置断点（这个命令看起来并不好用，行号会出问题
```



## core dump

> 当程序运行的过程中异常终止或崩溃，操作系统会将程序当时的内存状态记录下来，保存在一个文件中，这种行为就叫做Core Dump（核心转储），也就是保存程序崩溃的时候的内存快照

```bash
#（在命令行下输入）开启core dump，这个只对当前shell有效（也就是临时的，想要永久生效得设置对应的文件
ulimit -s unlimited
ulimit -c unlimited #不限制core dump文件的大小（如果要限制，unlimited改为对应的文件大小，单位是kb
ulimit -a #查看所有的ulimit变量值

#GPU下调试（OptiX/RTCore需要编译时候加-lineinfo，其他的加-g或者-G
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
(cuda-gdb) target core core.cpu core.cuda #同时看cpu和gpu的core文件
(cuda-gdb) target cudacore core.cuda.localhost.1234 #仅仅加载gpu的core文件

#修改/proc/sys/kernel/core_uses_pid文件可以让生成core文件名自动加上pid号
echo 1 > /proc/sys/kernel/core_uses_pid

#修改/proc/sys/kernel/core_pattern来控制生成core文件保存的位置以及文件名格式（默认保存在可执行文件所在的目录下
#将core文件保存在/tmp/corefile目录下，文件名格式为 core-命令名-pid-时间戳
echo "/tmp/corefile-%e-%p-%t" > /proc/sys/kernel/core_pattern

############################################################################################################
#使用gdb查看程序xxx执行后的core dump文件（文件名为yyy）信息
gdb xxx yyy
#接下来的操作就像是在调试的时候一样（but，记住这是一个dead样本

#回溯（backtrace）整个调用栈，对于栈中的所有帧来说，每一行对应于一个帧，回溯过程中使用从CTRL+c来停止回溯
bt [n] #n指定回溯innermost的n个帧
bt [-n] #-n指定回溯outermost的n个帧

#选择一个帧（frame）
f n #选择编号为n的帧
up n #向上移动n个帧（默认n=1），若n>0，则向outermost方向移动
down n #向下移动n个帧（默认n=1），若n<0，则向innermost方向移动

#查看帧的信息
info f #打印当前帧的详细信息
info args #打印当前帧的变量信息
layout reg #输出当前帧的寄存器信息（使用info也可，但不是tui
layout asm #输出汇编（tui
```



## 反向调试

```bash
#查看程序反向运行到哪里
bt #backtrace

#设置函数运行方向
set exec-direction [forward | reverse]

#开启记录，以便后面回溯
record

#根据记录反向运行
reverse-continue #反向运行到断点/函数开始
reverse-step #到上一行源代码
reverse-next #到上一行源代码，但是不进入函数
reverse-stepi #到上一条机器指令
reverse-nexti #到上一条机器指令；如果这条指令用来返回一个函数调用，则整个函数将会被反向执行

#结束反向运行，回到程序运行处
reverse-finish

#结束记录
record stop
```



## 并行

### 多进程

```bash
#默认追踪根进程，可以设置多进程同时调试
follow-fork-mode [parent/child] #调试父/子进程
detach-on-fork [on/off] #是否同时调试父、子进程（没有被跟进的进程block在fork位置

#显示所有被调试的进程（*表示正在调试的进程
info inferiors

#切换到某个进程去调试
inferior [进程编号]

#脱离进程
detach inferior [进程编号] #脱离该进程，该进程自由运行结束
kill inferior [进程编号] #杀死某进程，但是进程还在，可以用run命令执行它
remove-inferior [进程编号] #删除进程，删除前必须kill/detach
set schedule-multiple [on/off] #off表示只有当前的inferior会被执行；on表示所有执行状态的inferior都会被执行
set print interior-events [on/off] #打开/关闭inferior状态的提示信息
```



### 多线程

```bash
#显示所有线程的编号
info threads
#切换到某个线程
thread [线程编号]

#设置线程执行与否
show scheduler-locking #查看当前锁定的线程的运行模式
set scheduler-locking [on/off] #on是只有当前被调试程序会执行，off是所有线程都执行（默认值）
set non-stop [on/off] #调试一个线程时，其他线程是否运行

#设置线程执行特定的指令
thread apply [线程编号] command #线程编号之间以空格隔开
thread apply all command #所有线程都执行指令
```



## 附录

[GDB core dump例程](https://www.cse.unsw.edu.au/~learn/debugging/modules/gdb_coredumps/)

[使用gdb的100个小技巧](https://wizardforcel.gitbooks.io/100-gdb-tips/content/index.html)

[十五分钟教会gdb](https://www.bilibili.com/video/BV1KW411r7BR?from=search&seid=67360422147624704)

[cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html)

[GDB调试之栈帧、汇编](https://ivanzz1001.github.io/records/post/cplusplus/2018/11/08/cpluscplus-gdbusage_part4)

*多线程和多进程调试的内容比较复杂一点，后续用到的时候再根据需要去查*
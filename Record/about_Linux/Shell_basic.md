# Terminal&Shell Basic

*学习终端的基本知识和shell基本用法*

## 终端

*与Linux系统交互的直接接口---shell命令行界面(CLI，command line interface)*

**设置终端的背景色**

| 选项           | 参数                                           | 描述                                        |
| -------------- | ---------------------------------------------- | ------------------------------------------- |
| -background    | black/red/green/yellow/blue/magenta/cyan/white | 终端背景色改为指定颜色                      |
| -foreground    | black/red/green/yellow/blue/magenta/cyan/white | 终端字体色改为指定颜色                      |
| -inversescreen | on/off                                         | 交换背景色和字体色                          |
| -reset         | 无                                             | 终端外观恢复成默认设置并清屏                |
| -store         | 无                                             | 终端当前的字体色和背景色设置成reset选项的值 |

*注意：xterm是Linux中第一个可用的终端仿真器。它能够仿真旧式终端硬件，如VT和Tektronix终端。目前xterm还是用得比较多，查看参数请在命令行输入xterm -ti vt100*



## SHELL

> **shell的类型**
>
> 在/etc/passwd文件中，用户ID记录的第7个字段中列出了默认的shell程序
>
> 默认的交互shell一般是bash；还有一个默认系统shell是/bin/sh，它作为默认的系统shell，用于需要在启动时使用的系统shell脚本
>



## 基本bash shell命令

### 查看手册

```bash
#使用man来查看xxx的使用手册
man xxx

#使用关键字yyy搜索手册（忘记了命令名
man -k yyy

#还可以通过info命令查看xxx的手册
info xxx

#一般的命令带有help选项，比如git
git --help
```

**手册页有对应的内容区域，每个内容区域都分配了一个数字（区域号），从1开始，一直到9**

| 区域号 | 所涵盖的内容             |
| ------ | ------------------------ |
| 1      | 可执行程序或shell命令    |
| 2      | 系统调用                 |
| 3      | 库调用                   |
| 4      | 特殊文件                 |
| 5      | 文件格式与约定           |
| 6      | 游戏                     |
| 7      | 概览、约定及杂项         |
| 8      | 超级用户和系统管理员命令 |
| 9      | 内核例程                 |

*man工具通常提供的是命令所对应的最低编号的内容，如果要查看的命令有多个手册页都有记录，可以通过man section topic来指定看哪一个部分的，比如man 1 intro就是查看intro在区域1的内容*



### 文件基本操作

**ls、cp、mv、rm的参数**

```bash
#显示文件xxx的访问时间（默认是显示更改时间
ls -l --time=atime xxx

#显示当前目录下所有文件的inode（不同的文件的inode是不一样的
ls -i *

#递归显示当前目录
ls -R

#复制/移动文件xxx到yyy的时候加-i参数，提醒避免覆盖
cp -i xxx yyy

#mv移动文件不会改变时间戳和inode编号

#删除xxx的时候加参数-i，避免误删（会提醒是不是真的要删除
rm -i xxx
```



**链接文件**

> 符号链接：指向存放在虚拟目录结构中某个地方的另一个文件；这两个通过符号链接在一起的文件，彼此的内容并不相同
>
> 硬链接：独立的虚拟文件，其包含了原始文件的信息及位置；它们从根本上而言是同一个文件（通过查看inode可以得知）；引用硬链接文件等同于引用了源文件
>
> 注意：只能对处于同一存储媒体的文件创建硬链接，要想在不同存储媒体的文件之间创建链接， 只能使用符号链接；为了避免错误，尽量不要复制链接（再创建源文件的一个链接来使用

```bash
#创建yyy的符号链接xxx（这里xxx指向yyy，前提是原始文件yyy必须存在
ln -s yyy xxx

#创建yyy的硬链接xxx
ln yyy xxx
```



**处理目录**

```bash
#递归创建目录和子目录加入-p选项
mkdir -p New_Dir/Sub_Dir/Under_Dir

#删除目录的基本命令是rmdir，但是这只能删除空目录
#rm可以使用-r选项递归删除，这时候也可以删除目录下的内容以及目录本身
```



**查看文件内容**

```bash
#查看文件xxx的类型
file xxx

#cat查看整个文件
#n显示行号，-b只在有文本的地方加行号，-T隐藏制表符（用^I替换制表符）

#more分页显示文件内容，只能向后翻页
#less也可以分页显示文件内容，而且支持的操作比more更多（可以用man查看用法

#tail/head查看文件xxx后/前几行（-n指定行数，缺省是10
tail -n 3 xxx
tail -3 xxx
```



## 理解shell

*善用子shell会很有帮助，但是生成子shell的速度慢、成本高*

```bash
#在一行中指定要依次运行的一系列命令，只需要在命令之间加入;即可
pwd ; ls ; cd /etc ; pwd ; cd ; pwd ; ls
(pwd ; ls ; cd /etc ; pwd ; cd ; pwd ; ls) #构成进程列表（命令分组），生成了一个子shell来执行对应的命令
{pwd ; ls ; cd /etc ; pwd ; cd ; pwd ; ls} #构成进程列表，但是不会生成子shell来执行对应的命令

#使用&将进程列表置入后台
#当命令被置入后台，在shell CLI提示符返回之前，会出现两条信息。第一条信息是显示在方括号中的后台作业（background job）号; 第二条是后台作业的进程ID（可以用ps或者jobs命令查看
sleep 3& #将此命令放入后台执行

#使用coproc命令进行协程处理
coproc xxx { sleep 10; } #设置sleep 10为协程xxx，注意格式（必须确保在第一个花括号和命令名之间有一个空格、命令以分号结尾另外、分号和闭花括号有一个空格
```



**内建/外部命令**

> **外部命令**：有时候也被称为文件系统命令，是存在于bash shell之外的程序，它们并不是shell程序的一部分。外部命令程序通常位于/bin、/usr/bin、/sbin或/usr/sbin中。当外部命令执行时，会创建出一个子进程，这种操作被称为衍生（forking）
>
> **内建命令**：和shell编译成了一体，作为shell工具的组成部分存在，不需要借助外部程序文件来运行

```bash
#type命令
type xxx #可以查看xxx是否为内建命令
type -a xxx #要查看命令xxx的不同实现
#对于多种实现的命令，如果想要使用其外部命令实现，直接指明对应的文件即可，可以输入/bin/pwd执行外部的pwd命令

#查看最近用过的命令列表
history
history -a #在退出shell会话之前强制将命令历史记录写入.bash_history文件
!998 #唤起history中的第998条命令并执行
#小细节：命令历史记录被保存在隐藏文件.bash_history中，它位于用户的主目录中。这里要注意的是，bash命令的历史记录是先存放在内存中，当shell退出时才被写入到历史文件中

#命令别名
alias -p #查看当前可用的别名
alias xxx="yyy -z" #将xxx取为yyy -z的别名（这种做法使得xxx只在当前shell有效
```

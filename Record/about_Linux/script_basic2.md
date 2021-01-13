# Bash Shell Script Basic2

*学习bash脚本进阶*

## 处理用户输入

### 命令行参数

```bash
#bash shell会将特殊变量位置参数（positional parameter）分配给输入到命令行中的所有参数
#位置参数变量是标准的数字：$0是程序名，$1是第 一个参数，$2是第二个参数，依次类推，直到第九个参数$9

./xxx.sh "a a"  b c #命令行中输入下述语句，传递给xxx.sh脚本的参数为a a、b、c（这里""仅仅表示数据的起始位置
$1 #脚本中这个变量代表第一个参数a a

#使用basename剥离脚本变量名（否则会显示命令行下输入的路径
name=$(basename $0) #这样得到的name就仅仅是脚本的名称

#使用命令行参数变量的时候，必须先检查该参数是否存在
if [ -n "$1" ] #检查第一个参数是否存在的if语句

#特殊的参数变量
$# #这个特殊的变量表示脚本运行时携带的命令行参数的个数
${!#} #表示最后一个参数（不能在括号中使用$，必须换成!
$* #将命令行上提供的所有参数当作一个单词保存在这个参数里（默认空格分割
$@ #将命令行上提供的所有参数当作同一字符串中的多个独立的单词保存保存在这个参数里（默认空格分割

#使用shift命令时，默认情况下它会将每个参数变量向左移动一个位置（$1被删除，而$0不变
while [ -n "$1" ]; do
	echo "Parameter #$count = $1" #仅仅使用$1遍历整个命令行变量参数
	count=$[ $count + 1 ]
	shift
done
#注意：使用shift命令的时候，如果某个参数被移出，它的值就被丢弃了，无法再恢复（需谨慎
```



### 命令行选项

#### 基本方法

```bash
#基本方法：用shift的方法移动参数
#用处理命令行参数的方法处理命令行选项
./xxx.sh -a -b -c #在脚本中写case语句处理-a、-b、-c即可

#分离命令行选项和命令行参数参数
./xxx.sh -a -b -- para1 para2 para3 #用--将命令行选项和参数分开（--后面的是命令行参数

#处理带值的选项
#举例：脚本内容
while [ -n "$1" ]; do
	case "$1" in
		-a) echo "Found the -a option";;
		-b) param="$2"
			echo "Found the -b option, with parameter value $param"
			shift ;;
		-c) echo "Found the -c option";;
		--) shift
			break ;;
		*) echo "$1 is not an option";;
	esac
	shift
done
#调用脚本的命令行语句
./xxx.sh -a -b para1 -c
```

#### getopt命令

```bash
#使用getopt命令（接受一系列任意形式的命令行选项和参数，并自动将它们转换成适当的格式
getopt [optstring] [parameters] #基本使用格式（其中optstring定义了命令行有效的选项字母以及哪些选项需要参数

#在命令行中执行下面语句可以看到具体的输出，理解getopt是怎么解析参数的
getopt ab:cd -a -b test1 -cd test2 test3 #定义了四个有效选项abcd，其中:在b后面表示b需要参数
#输出：-a -b test1 -c -d -- test2 test3
getopt ab:cd -a -b test1 -cde test2 test3 #其中包含了一个无效参数，输出中会给出警告信息
getopt -q ab:cd -a -b test1 -cde test2 test3 #加了-q参数，忽略警告信息

#脚本中使用
set -- $(getopt -q ab:cd "$@") #set选项--将命令行参数替换成后面的参数（也就是用getopt解析的结果替换了原来的命令行参数
while [ -n "$1" ]; do
	case "$1" in
		-a) echo "Found the -a option";;
		-b) param="$2"
			echo "Found the -b option, with parameter value $param"
			shift ;;
		-c) echo "Found the -c option";;
		--) shift
			break ;;
		*) echo "$1 is not an option";;
	esac
	shift
done
count=1 #将剩下的参数输出
for param in "$@"; do
	echo "Parameter #$count: $param"
	count=$[ $count + 1 ]
done

./xxx.sh -ac #这样调用脚本，它依然可以正常工作

#但是getopt无法正常解析带""的选项（因为它只将空格当成分隔符
./xxx.sh -cd "1 2" 3 #这种情况，会将1 2解析成为两个参数
```

#### getopts命令

```bash
#getopts是bash shell的内建命令
getopts [optstring] [variable] #基本使用格式（其中optstring定义了命令行有效的选项字母以及哪些选项需要参数

#与getopt不同，getopts每次调用只处理一个参数，处理完所有的参数后，它会退出并返回一个大于0的退出状态码
#getopts命令会用到两个环境变量：OPTARG（如果该参数有选项，则保存选项参数）、OPTIND（保存参数列表中getopts正在处理的参数位置）
getopts :ab:cd -a -b test1 -cde test2 test3 #在optstring前加:，忽略警告信息

#举例
while getopts :ab:c opt; do
	case"$opt" in #getopts命令解析命令行选项时会移除开头的-，所以不需要在case中加-
		a) echo "Found the -a option" ;;
		b) echo "Found the -b option, with value $OPTARG";;
		c) echo "Found the -c option" ;;
		*) echo "Unknown option: $opt";;
	esac
done
shift $[ $OPTIND - 1 ] #处理完最后一个选项时，用shift和OPTIND移去那个选项
count=1 #将剩下的参数输出
for param in "$@"; do
	echo "Parameter #$count: $param"
	count=$[ $count + 1 ]
done

#getopts几个优点：可以识别""；将选项字母和参数值放在一起使用，而不用加空格；将未定义的选项输出为?
```

**一些具有特定含义的选项**

```bash
-a #显示所有对象
-q #以安静模式运行
-s #以安静模式运行
-i #忽略文本大小写
-r #递归处理目录和文件
```



### 交互式输入

```bash
#read命令从标准输入（键盘）接受输入并存储到变量中，使用重定向的时候可以读取文件中的一行数据
read age #将变量存储在age中
read -p "Please enter your age: " age #输出语句，读取输入并将其存储到age中
read -p "Enter your name: " #将变量存储在特殊的环境变量REPLY中

#指定超时的时间（计时器过期后，返回一个非零退出状态码
if read -t 5 -p "Please enter your name: " name; then
	echo "Hello $name, welcome to my script" 
else
	echo echo "Sorry, too slow! "
fi

#指定读取字符数目
read -n1 -p "Do you want to continue[Y/N] " answer #-n指定接受字符数目为1
case $answer in
	Y | y) echo "fine, continue on…";;
	N | n) echo OK, goodbye
		exit;;
esac

#以隐藏的方式读取（没有回显
read -s -p "please input you password: " passwd

#从文件中读取一行数据
count=1
cat xxx.txt | while read line; do #每次读取一行文本，没有内容之后返回非零退出状态码
	echo "Line $count: $line"
	count=$[ $count + 1]
done
echo "Finished processing the file"
```



## 呈现数据

### 文件描述符

> Linux系统将每个对象当作文件处理，包括输入和输出进程，并且用文件描述符（file descriptor）来标识每个文件对象
>
> 文件描述符是一个非负整数，可以唯一标识会话中打开的文件
>
> 每个进程一次最多可以有九个文件描述符，但是bash shell出于特殊目的只保留了3个

| 文件描述符 | 缩写   | 描述     | 用法                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| 0          | STDIN  | 标准输入 | 使用<可以将数据输入到任何从STDIN接受数据的shell命令          |
| 1          | STDOUT | 标准输出 | \>重写，\>>追加                                              |
| 2          | STDERR | 标准错误 | 默认和STDOUT指向同样的地方，但STDERR不会随着STDOUT的重定向而发生改变 |

```bash
#只重定向STDERR的内容到文件xxx
ls -a 2>xxx

#分别重定向错误到文件xxx和非错误数据到文件yyy
ls -a 2>xxx 1>yyy

#将STDOUT和STDERR都重定向到文件xxx，并且由于bash shell赋予了错误信息更高的优先级，所以会在文件前面显示
ls -a &>xxx

#列出打开的文件描述符（lsof的默认输出描述请自行看书：pdf的337/621，表15-2
lsof #管理员直接可以运行这个
/usr/sbin/lsof #普通用户必须指定全路径才能运行（防止信息泄露
/usr/sbin/lsof -a -p $$ -d0,1,2,3,6,7 #p指定进程ID，$$表示当前PID；d指定显示的文件描述符编号；a表示对选项执行and操作
```

### 脚本中的重定向

```bash
#临时输出重定向
#脚本xxx.sh中写入
echo "error" >&2 #把这一则消息重定向到STDERR
echo "normal" #不作处理，则默认是STDOUT
#执行xxx.sh，并将错误输出到yyy文件
./xxx.sh 2>yyy #这样error则输出到了yyy文件，而normal则输出到屏幕上

#永久输出重定向
exec 1>xxx #exec命令启动一个新shell，并将脚本中发给STDOUT的所有输出会被重定向到文件xxx
exec 2>yyy #将所有的STDERR内容重定向到文件yyy
#在脚本中可以交错使用永久重定向，而且依然可以使用临时重定向来改变定向的位置

#使用输入重定向
exec 0<xxx #将文件xxx作为输入重定向到STDIN
#例子：从文件xxx中读取信息
exec 0<xxx
count=1
while read line; do
	echo "Line #$count: $line"
	count=$[ $count + 1 ]
done

#创建自己的重定向
exec 3>xxx #将文件描述符3重定向到文件xxx，也就是3指向了文件xxx
exec 6<&0 #将文件描述符0重定向到6，也就是6指向了标准键盘输入
#例子
exec 3>xxx
echo "This should display on the monitor" #输出到STDOUT
echo "and this should be stored in the file" >&3 #这一句话会被输出到文件xxx中
echo "Then this should be back on the monitor" #输出到STDOUT

#将已经被重定向的标准输入/输出/错误重新定向到原来的位置：重定向前先将其保存在自己创建的重定向中，需要换回来的时候再改回来
exec 3>&1 #将文件描述符3定向到文件描述符1，即3指向显示器
exec 1>xxx #将文件描述符1定向到文件xxx，即1指向文件xxx
echo "This should store in the output file" #输出到文件xxx
exec 1>&3 #将文件描述符1重定向到3，即1重新指向显示器
echo "Now things should be back to normal" #输出到显示器
#######################################################################################################
echo 6<&0 #将文件描述符0定向到6，即6指向标准键盘输入
exec 0<yyy #将文件描述符0定向到文件yyy，即标准输入是文件yyy
#执行从文件yyy读取信息的操作
exec 0<&6 #将STDIN恢复到原来的位置

#创建读写文件描述符（但是这种操作很危险，容易出错
exec 3<>xxx #将文件描述符3分配给文件xxx，具有读和写的功能
#注意：文件描述符本质是一个指针，当脚本向文件中写入数据时，它会从文件指针所处的位置开始

#关闭文件描述符
exec 3>&- #将文件描述符3重定向到特殊符号-，即可关闭文件描述符3

#阻止输出
exec 2 > /dev/null #将标准错误重定向到null文件，这样shell输出的数据都会被丢掉而不被保存
cat /dev/null > xxx #常用这种方法来将文件xxx清空，而不用删除再重建
```

### 创建临时文件

> Linux使用/tmp目录来存放不需要永久保留的文件
>
> 大多数Linux发行版配置了系统在启动时自动删除/tmp目录的所有文件。

```bash
mktemp #在/tmp中创建临时文件，文件名随机生成
mktemp -t test.XXXXXX #t指定在/tmp中创建临时文件，文件名模板为test.*

#在当前目录下创建临时文件（文件的读和写权限将被分配给文件的创建者，即属主
mktemp test.XXXXXX #指定文件名模板为test.*（后面的6个x是为了保证文件唯一）

#创建临时目录
mktemp -d dir.XXXXXX #d指定创建临时目录，目录名称模板为dir.*

#在脚本中，创建并使用临时文件
tempdir=$(mktemp -d testdir.XXXXXX)
cd $tempdir
testfile1=$(mktemp test.XXXXXX)
testfile2=$(mktemp test.XXXXXX)
exec 3>$tempfile1
echo "This is the first tempfile" >&3 #重定向方法1
echo "This is the second tempfile" | tee -a $testfile2 #重定向方法2
exec 3>&-
cat $tempfile
rm -f $tempfile 2> /dev/null
```



## 控制脚本

> 各种系统信号见sys_control.md
>
> shell中运行的每个进程称为作业，每个作业会被分配唯一的作业号

### 处理信号

```bash
#中断进程（SIGINT）：CTRL+c
#暂停进程（SIGTSTP）：CTRL+z
ps -l #查看各个进程，其中S列中会将被暂停的进程显示为T
exit #终止已被暂停的作业（或者ps查看了其作业ID之后，使用kill终止

#捕获信号
trap commands signals #命令使用的基本格式

#示例1：捕获CTRL+c信号（无法通过CTRL+c终止这个脚本进程）
trap "echo 'sorry, CTRL+c has been trapped'"
count=1
while [ $count -le 10 ]; do
	echo "Loop #$count"
	sleep 3
	count=$[ $count + 1 ]
done
#在脚本执行的过程中，所有的CTRL #+c都会被捕捉而不被处理，而且每次都会回复echo里面那句话

#示例2：捕获EXIT信号（脚本退出时输出执行一些操作
trap "echo Goodbye..." EXIT #脚本退出时（正常退出/被强制终止），输出Goodbye...
count=1
while [ $count -le 10 ]; do
	echo "Loop #$count"
	sleep 3
	count=$[ $count + 1 ]
done

#删除捕获
trap -- SIGINT #删除对SIGINT信号的捕获
```

### 后台运行

```bash
#在命令后加&即可将命令作为系统中的一个独立的后台进程运行
./xxx.sh & #这个命令将脚本xxx.sh放到后台运行；开始时返回作业号和进程ID；结束时也会返回信息
#然而，它仍然会使用终端显示器来显示STDOUT和STDERR消息，最好将脚本中的输出信息进行重定向；而且当shell退出时，进程回被强制终止

#在非控制台下运行脚本（阻断所有发送给脚本的SIGHUP信号，让脚本一直以后台模式运行到结束，即使shell退出
nohup ./xxx.sh &
#由于nohup命令解除终端与进程的关联，进程不再同STDOUT和STDERR联系，STDOUT和STDERR的消息被重定向到名为nohup.out的文件中
#如果nohup运行了多个命令，输出会被追加到nohup.out这样会导致文件信息混乱，需要谨慎使用
```

### 作业控制

```bash
#查看作业
jobs -l #l查看作业的PID
#带+的作业会被当做默认作业，如果未指定任何作业号，该作业会被当成作业控制命令的操作对象；当前默认作业完成处理后，带-的作业成为下一个默认作业；任何时候，都只有一个带+和一个带-的作业

#重启已停止的作业
bg 3 #以后台模式重启作业号为3的作业
fg 3 #以前台模式重启作业号为3的作业
```

### 调整谦让度

> 在多任务操作系统中，内核负责将CPU时间分配给系统上运行的每个进程
>
> 调度优先级（scheduling priority）是内核分配给进程的CPU时间（相对于其他进程）
>
> 调度优先级是个整数值，从20（最高优先级）到+19（最低优先级）
>
> 在Linux系统中，由shell启动的所有进程的调度优先级均默认为0

```bash
#设置命令启动时的调度优先级
nice -n 10 ./xxx.sh & #设置脚本xxx.sh以优先级为10运行
nice -10 ./xxx.sh & #同上
ps -o pid,ni,cmd #查看当前进程的优先级
#nice命令不允许普通用户提高进程的优先级（也就是不能设置n小于0

#修改已运行命令的优先级
renice -n 10 -p 4321 #修改PID为4321的进程的优先级为10
#注意：只能对属于你的进程执行renice；只能通过renice降低进程的优先级；root用户可以通过renice来任意调整进程的优先级
```



## 定时运行作业

### at命令

> at命令允许指定Linux系统何时运行脚本，并将作业提交到队列中，指定shell在该时运行该作业
>
> at的守护进程atd以后台模式运行，该进程会检查系统上的一个特殊目录（通常位于/var/spool/at）来获取用at命令提交的作业。默认情况下，atd守护进程每60s检查一下这个目录。有作业时，atd守护进程会检查作业设置运行的时间，如果时间跟当前时间匹配，atd守护进程就会运行此作业；如果时间已经过了，会在第二天同样时间执行
>
> 针对不同优先级，存在26种不同的作业队列，作业队列通常用小写字母a~z 和大写字母A~Z来指代；作业队列的字母排序越高，作业运行的优先级就越低（更高的nice值），默认作业被提交到a作业队列
>
> 注意：显示器并不会关联到该作业，Linux系统会将提交该作业的用户的电子邮件地址作为STDOUT和STDERR；任何发到STDOUT或STDERR的输出都会通过邮件系统发送给该用户。所以最好在脚本中对STDOUT和STDERR进行重定向

```bash
#指定时间执行作业
at [-f filename] time #基本命令格式（可以用-q选项指定不同的优先级，即队列
#time的格式
#标准的小时和分钟格式：10:15
#AM/PM指示符：10:15 PM
#特定可命名时间：now、noon、midnight、teatime（4 PM）
#标准日期格式：MMDDYY、MM/DD/YY或DD.MM.YY
#文本日期：Jul 4或Dec 25，加不加年份均可
#指定时间增量：当前时间+25 min，10:15+7天

at -f xxx.sh now #现在执行脚本xxx.sh
at -M -f xxx.sh now #如果不想在at命令中使用邮件或重定向，可以加上-M选项来屏蔽作业产生的输出信息

atq #列出等待的作业
atrm 11 #删除作业号为11的作业
```

### cron命令

> Linux系统使用cron程序来安排要定期执行的作业
>
> anacron程序：cron程序的问题是它假定Linux系统是7×24小时运行的，然而这未必成立。当系统关机在开机之后，cron不会去运行错过的作业，而如果anacron知道某个作业错过了执行时间，它会尽快运行该作业（当系统再次开机时，原定在关机期间运行的作业会自动运行）；anacron程序只会处理位于cron目录的程序
>
> *详细的anacron内容需要的时候再man查询*

```bash
#cron时间表（指定作业何时运行
min hour dayofmonth month dayofweek command #基本命令格式
15 10 * * * ./xxx.sh #在每天的15:10执行脚本xxx.sh
15 16 * * 1 ./xxx.sh #在每周一4:15PM执行脚本xxx.sh（0是周日，6是周六
00 12 1 * * ./xxx.shd #在每个月的第一天中午12点执行脚本xxx.sh
00 12 * * * if [`date +%d -d tomorrow` = 01 ] ; then ; command #在每个月的最后一天中午12点执行命令

#构建cron时间表
crontab -e #为cron时间表添加条目
crontab -l #列出已有的cron时间表

#浏览预置的cron目录（hourly、daily、monthly、weekly
ls /etc/cron.*ly
#如果脚本需要每天运行一次，只要将脚本复制到daily目录，cron就会每天执行它
```


# Bash Shell Script Advance

*学习bash脚本高阶*

## 函数

### 基础

> 函数必须先声明再调用
>
> 函数可以多次重复写，以调用前的最后一个执行
>
> 默认情况下，函数的退出状态码是函数中最后一条命令返回的退出状态码。在函数执行结束后，可以用标准变量$?来确定函数的退出状态码

```bash
#创建函数方式1
function name(){
	commands
}

#创建函数方式2
name(){
	commands
}
```

### 返回与传参

```bash
#return命令指定一个整数值作为函数的退出状态码（必须是0~255，本质上是返回执行的最后一条命令的退出状态码
function dbl {
	read -p "Enter a value: " value
	echo "doubling the value"
	return $[ $value * 2 ] #return的值即为状态退出码
}
dbl
echo "The new value is $?" #$?返回执行最后一条命令的退出码

#使用函数输出
function dbl {
	read -p "Enter a value: " value
	echo $[ $value * 2 ]
}
result=$(dbl) #将函数dbl的输出（STDOUT）赋值给result
echo "The new value is $result"

#在脚本中指定函数将参数和函数放在同一行
func1 $value1 10
#在函数中使用参数，使用的形式类似命令行参数（即$1、$2等），注意这里的参数不是直接的命令行参数，而是脚本中显式地传给函数的参数
function func7 {
	echo $[ $1 * $2 ]
}
if [ $# -eq 2 ]; then
	value=$(func7 $1 $2)
	echo "The result is $value"
else
	echo "Usage: badtest1 a b"
fi
```

### 变量和数组

```bash
#bash中的局部变量和全局变量类似于c；但是，与c不同的是，bash的函数中默认是全局变量，如果想要指定是局部变量，加local关键字
local result=1

#传递数组
myarray=(1 2 3 4 5)
func ${myarray[*]} #如果仅仅传递$myarray的话，就相当于只传递了数组的第一个元素

#例子（返回数组类似，需要的时候自行查书理解P383/621
function addarray {
	local sum=0
	local newarray
	newarray=($(echo "$@"))
	for value in ${newarray[*]}; do
		sum=$[ $sum + $value ]
	done
	echo $sum
}
myarray=(1 2 3 4 5)
echo "The original array is: ${myarray[*]}"
arg1=$(echo ${myarray[*]})
result=$(addarray $arg1)
echo "The result is $result"

#source和.命令都可以加载库文件
#关于递归调用和函数库的使用，请自行参考书的内容P386/621
```



## 图形界面

```bash
#创建菜单前首先清空界面
clear

#要在echo命令中包含非可打印的字符，必须用-e选项
echo -e "\t1. Display disk space"
echo -en "\t\tEnter option: " #n表示不换行
read -n 1 option

#更多的、真正图形化的设计，请自行参考书第18章节内容
```



## sed

> 流编辑器，在编辑器处理数据之前基于预先提供的一组规则来编辑数据流
>
> 操作：一次从输入中读取一行数据；根据所提供的编辑器命令匹配数据；按照命令修改流中的数据；将新的数据输出到STDOUT
>
> *并没有对源文件真正进行修改，只是指定了输出了限制*

```bash
#命令格式
sed options script file

sed -e 's/brown/green/; s/dog/cat/' xxx.txt #替代文件xxx.txt中对应的内容，-e表示多个目录
#如果不想用；分割，可以使用交互式的，先输入sed -e '然后一个一回车，最后一行输入命令和' 再输入文件名即可

sed -f script data.txt #使用文件script中的命令来处理data.txt

#s替换（substitute，其实这里的替换和vim的习惯差不多
s/pattern/replacement/flags #基本格式（flag可为行号，g（全部），p（打印原先行），w 文件名（将替换的结果写到文件中）
echo "This is a test" | sed 's/test/big test/'
sed -n 's/test/trial/p' data.txt #-n选项将禁止sed编辑器输出，-p替换标记会输出修改过的，结果就是只有修改过的被输出
sed 's!/bin/bash!/bin/csh!' /etc/passwd #使用!作为分隔符，这样更方便地处理一些需要/的地方（不然要用\转义
sed '3,${s/fox/elephant/}' data.txt #$是指最后一行，这里指定了第3到最后一行的替换

#d/D删除（delete
sed '2,6d' data.txt #删除到6行
sed '/number 1/d' data.txt #匹配number 1，然后删掉匹配出来的行
#D只删除多行模式空间中的第一行（下述的N和P同理

#n/N下一行（next
sed '/header/{n;d}' data.txt #找到header匹配的行，跳到下一行，删除该行（这个命令是重复执行的，直到文件结束
sed '/first/{N;s/\n/ / }' data.txt #找到first匹配的行，将下一行添加到该行后，然后将换行符改为空格

#p/P打印（print
sed -n 'N ; /System/P' data.txt #打印匹配到System的模式空间中的第一行

#其他：!排除该行；h/H/g/G/x操作保持空间；b分支；t测试；&代表替换命令中的匹配的模式；()替换子模式；
```



## gawk

> 提供了一种编程语言而不只是编辑器命令
>
> 操作：定义变量来保存数据；使用算术和字符串操作符来处理数据；使用结构化编程概念（比如if-then语句和循环）为数据处理增加处理逻辑；通过提取数据文件中的数据元素，将其重新排列或格式化，生成格式化报告

```bash
#命令格式
gawk options program file
#CTRL+d结束gawk的交互式程序

gawk '{print "hello world!"}' #把命令放到花括号中，而gawk命令行假定脚本是单个文本字符串，所以还需要''
echo "My name is Rich" | gawk '{$4="Christine"; print $0}' #多条命令用;分割

#$0代表一个文本行，$1~n代表文本行中的第n个数据字段
gawk '{print $1}' data.txt #这个命令会打印文件data.txt中的每一行的第一个单词
gawk -F: xxx /etc/passwd #使用F指定分割符为:，默认以任意空白字符分割；读取文件xxx中的命令并执行与/etc/passwd

#关键字BEGIN/END可以在处理数据前/后进行一些操作
gawk 'BEGIN {print "hello world"} {prin $0} END {print "goodbye"}' data.txt
#或者，看一个gawk的脚本
BEGIN {print "The latest list of users and shells";print " UserID       \t       Shell";print "-------- \t -------";FS=":"}
{print $1 "         \t " $7}
END {print "This concludes the listing"}
```

*sed和gawk的内容较为丰富，且贴近实际使用，此处不作全部记录，请自行查阅书第19、21、22章的内容，或者使用man手册*



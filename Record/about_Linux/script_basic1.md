# Bash Shell Script Basic1

*学习bash脚本基础*

## 创建脚本

```bash
#!/bin/bash
#上面这一行不是一般的注释，它指定了执行脚本的shell，除它之外，其他的以#开头均为注释
echo "hello world"

#在命令行中执行下述指令
chmod u+x xxx #给脚本xxx添加执行权限
export .:$PATH #将当前路径添加到可执行文件的路径，这样就可以直接输入脚本名称执行脚本
xxx #执行脚本
```



## 基本用法

```bash
#把文本字符串和命令输出显示在同一行（添加-n参数
echo -n "the time and date are:"
date

#特殊符号需要转义（$符号是表示引用的变量的内容
$PATH #表示可执行文件的路径
\$PATH #仅仅表示字符串$PATH

#从命令输出中提取信息，并将其赋给变量（使用``或者$()都可以
testing=`date +%y%m%d` #格式化date的输出并赋值给testing
today=$(date +%y%m%d) #其中%y%m%d告诉date命令将日期显示为年月日的组合（这是date命令的用法，可以通过man查到
#反引号的作用就是将反引号内的Linux命令先执行，然后将执行结果赋予变量（可以直接在命令行模式下敲一个`然后回车试试
#注意：命令替换符会创建一个子shell来运行对应的命令；使用内建的shell命令不会创建子shell

#输入、输出和重定向
who > xxx #将who命令的输出写入文件xxx（不存在则创建，存在则重写
who >> xxx #将who命令的输出追加到文件xxx（文件必须存在
wc < xxx #将文件xxx重定向到命令wc（输出行、词、字节数
wc << yyy #内联重定向（在命令行中输入需要重定向的内容，使用yyy作为输入停止的标志

#使用管道符进行重定向
[command1] | [command2] #将command1的输出直接重定向到command2
ls -R | sort | more #递归地列出本文件夹下的所有文件，用sort命令将其排列，并使用more分页查看
ls -R | sort > files.txt #将结果输出到files.txt中

#shell中运行的每个命令都使用退出状态码（exit status）告诉shell它已经运行完毕，这个值在命令结束运行时由命令传给shell
#退出状态码的取值的0~255的整数（成功结束的状态码是0；其他的自行搜索引擎查询，此处不列举了
echo $? #$?这个变量专门用于保存上一个已执行命令的退出状态码
exit 4 #在脚本中写了这一行表示返回退出状态码4（就是可以自定义退出状态码，但是注意数值限定在0~255，超出了会对256取模

#注意：bash中的'定义字符串所见即所得，即将内容原样输出；而"会把里面的变量、命令等解析出来（使用echo '$var'和echo "$var"
```

*疑问：命令替换会创建一个子shell来运行对应的命令。子shell（subshell）是由运行该脚本的shell 所创建出来的一个独立的子shell（child shell）。正因如此，由该子shell所执行命令是无法使用脚本中所创建的变量的。？？？*



## 执行数学运算

```bash
#使用expr命令（Bourne shell古老的方法
expr $a \* $b #将变量a的值和变量b的值相乘

#使用方括号（bash改进的方法
var=$[$a * $b] #将变量a的值和变量b的值相乘

#使用内建的bash计算器（bash默认只支持整数运算，进行浮点运算需要用用bc；zsh支持完整的浮点运算
bc -q #-q不显示欢迎信息；设置scale（默认为0）的值来改变小数点的位数；在计算器里也可以定义变量

#脚本中使用计算器
variable=$(echo "options; expression" | bc) #使用管道符的方法
variable=$(bc << EOF
options
statements
expressions
EOF
)  #内联输入重定向的方法（逻辑更清晰

var=$(echo "scale=4; 3.44 / 5" | bc) #设置4位小数输出，计算3.44/5的结果并赋值给var
#计算a1+b1
var1=10.46
var2=43.67
var3=33.2
var4=71
var5=$(bc << EOF
scale = 4
a1 = ($var1 * $var2)
b1 = ($var3 * $var4)
a1 + b1
EOF
)
```



## 字段分隔符

```bash
#bash shell使用内部字段分隔符（internal field separator）作为分隔符，这是一个环境变量IFS决定的
#默认情况下的字段分隔符为：空格、制表符、换行符
IFS=$'\n':;" #更改字段分隔符为换行符、冒号、分号、双引号
IFS=: #更改IFS为冒号

#比较推荐使用下述方法改变IFS的值，这样较为安全
IFS.OLD=$IFS
IFS=$'\n' #更改IFS
IFS=IFS.OLD #更换回原来的IFS

#当分隔符为空格的时候，将存在空格的字段用""引起来即可被shell正常识别
```



## 结构化命令

*structured command，允许控制语句的执行顺序*

> 这些结构化命令会被当作一整个语句块，需要重定向只需要在最后的语句结束标志那里操作即可

### if语句

```bash
#if-then语句的基本格式
if command1 #当且仅当command1的退出状态码是0，执行后面的语句
then
	command2
	... #这里表示其他的语句
fi

if command1; then #这个格式更好看一点
	command2
	...
fi
#下面的then同样可以放在if那一行，不赘述

#if-then-else语句
if command1 #当且仅当command1的退出状态码是0，执行then语句块的内容
then
	command2
	... #这里表示其他的语句
else #当if语句中的命令返回非零退出状态码时，执行else语句块的内容
	commands #也可以是多条语句
fi

#嵌套if语句是可以执行的，但是逻辑不够清晰，推荐使用elif
if command1 #当且仅当command1的退出状态码是0，执行then语句块的内容
then
	commands
elif command2 #当且仅当command2的退出状态码是0，执行then语句块的内容
then
	commands
else #这一层是嵌套在elif上的
	commands
fi
#嵌套可以有多层，此处不赘述（不建议嵌套太多

#if语句只能判断退出状态码是否为0，而test语句可以检测其他的状态码
test [conditions]
if test conditions; then #test在if语句中的使用
	commands
else
	commands
fi

#或者使用下面这种方式代替test
if [ conditions ]; then #第一个方括号之后和第二个方括号之前必须加上一个空格
	commands
else
	commands
fi

#复合条件测试（and和or
[ condition1 ] && [ condition2 ]
[ condition1 ] || [ condition2 ]

#上述的conditions可以为：数值比较、字符串比较、文件比较
```



#### 数值比较

```bash
#以下n1和n2仅限整数
n1 -eq n2 #n1 == n2
n1 -ge n2 #n1 >= n2
n1 -gt n2 #n1 > n2
n1 -le n2 #n1 <= n2
n1 -lt n2 #n1 < n2
n1 -ne n2 #n1 != n2

#示例
num1=100
num2=10
if [ $num1 -eq $num2 ]; then
	echo "num1 is equal to num2"
elif [ $num1 -gt $num2 ]; then
	echo "num1 is greater than num2"
fi  
```



#### 字符串比较

```bash
#str1和str2是字符串（注意比较符号的转义；其中str1和str2必须是变量，而不能直接是字符串
str1 = str2 #str1与str2是否完全相同
str1 != str2 #str1与str2是否不同
str1 \< str2 #str1是否比str2小
str1 \> str2 #str1是否比str2大
-n str1 #检查str1的长度是否非0
-z str1 #检查str1的长度是否为0（没有被定义的字符串的长度也是0

#注意：字符串比较的大于和小于顺序和sort命令所采用的不同
#比较测试中使用的是标准的ASCII顺序；sort命令使用的是系统的本地化语言设置中定义的排序顺序，对于英语，本地化设置指定了在排序顺序中小写字母出现在大写字母前

if [ "$var1" = "$var2" ] #防止因为var1或者var2为空导致报错，所以加""
```



#### 文件比较

```bash
-e file #检查file是否存在
-d file #检查file是否存在并是一个目录
-f file #检查file是否存在并是一个文件
-s file #检查file是否存在并非空
-r file #检查file是否存在并可读
-w file #检查file是否存在并可写
-x file #检查file是否存在并可执行
-O file #检查file是否存在并属于当前用户
-G file #检查file是否存在并且默认组与当前用户相同
file1 -nt file2 #检查file1是否比file2新
file1 -ot file2 #检查file1是否比file2旧

#需要创建/读写文件时，先检查是否存在总是好事情（先查目录再查文件
```



#### 高级用法

```bash
#数学表达式的双括号（而且不需要将双括号中表达式里的大于/小号转义
(( expression )) #expression可以是任意的数字赋值/比较表达式
#支持++、--、!、&&、||、~（位求反）、**（幂运算）、<<（左位移）、>>（右位移）、&（位布尔和）、|（位布尔或）

#高级字符串处理功能的双方括号
[[ expression ]] #使用模式匹配（正则表达式）比较字符串

#例子
$USER=root
if [[ $USER == r* ]]; then
	echo "Hello $USER"
fi
```



### case语句

```bash
#基本格式
case variable in
pattern1)
	commands1;;
pattern2)
	commands2;;
...
*)
	commands;; #这个就是除了给出的pattern之外的其他可能性，相当于default
esac
```



### 循环

```bash
#控制循环的语句
break #跳出当前循环
break n #跳出n层循环，默认是1

continue #跳过本次循环，直接进入下一次循环
continue n #跳过n层循环，直接进入第n层循环的下一次循环

#循环也可以嵌套，此处不再赘述
```



#### for循环

```bash
#bash shell的for循环的基本格式
for var in list
do
	commands
done

for var in list; do
	commands
done

#如果遇到单引号内的内容被定义为单独的数据值，可以有两种解决办法：使用转义字符来将单引号转义；使用双引号来定义用到单引号的值
#I don't know if this'll work--->I don\'t know if "this'll" work

#在list中，for循环假定每个值都是用空格分割的
#Nevada New Hampshire New Mexico New York--->Nevada "New Hampshire" "New Mexico" "New York"

#例子
file="state"
for state in $(cat $file); do
	echo "visit $state"
done


#c语言的for循环的基本格式（使用需谨慎
for (( variable assignment ; condition ; iteration process )) #变量可以定义多个
do
	commands
done
#注意这个与bash shell的for不同的地方：变量赋值可以有空格；条件中的变量不以美元符开头；迭代过程的算式未用expr命令格式
```



#### while循环

```bash
#基本格式
while test command #当且仅当退出状态码为0，执行循环内的语句
do
	other commands
done
#注意：test command可以为多个指令，只有最后一个指令的退出状态码决定什么时候结束循环

#例子
var1=10
while echo $var1
	[ $var1 -ge 0 ] #这一句决定什么时候结束循环
do
	echo "This is inside the loop" var1=$[ $var1 - 1 ]
done
```



#### until语句

```bash
#基本格式
until test command #当且仅当测试命令的退出状态码不为0时，执行循环中列出的命令
do
	other commands
done
#注意：test command可以为多个指令，只有最后一个指令的退出状态码决定什么时候结束循环

#输出结果重定向到文件xxx（的所有的结构化命令都可以这么操作
until [ $var -eq 0]; do
	echo $var
done > xxx #管道符也可以这样直接用
```



## 实例

#### 查找可执行文件

```bash
#!/bin/bash
IFS=:
for folder in $PATH; do
	echo "$folder:"
	for file in $folder/*; do
		if [ -x $file ]; then
			echo "	$file"
		fi
	done
done
```

#### 创建多个用户账户

```bash
#!/bin/bash
input="user.csv" #这是一种文件格式，内容为userid，user_name
while IFS=',' read -r userid name #read命令会自动读取文件的下一行内容，当返回FALSE时（读取完整个文件），退出循环
do
	echo "adding $userid"
	useradd -c "$name" -m $userid
done < "$input" #$input变量指向数据文件，并且该变量被作为while命令的重定向数据
```


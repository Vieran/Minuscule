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

### 变量

```bash
#bash中的局部变量和全局变量类似于c；但是，与c不同的是，bash的函数中默认是全局变量，如果想要指定是局部变量，加local关键字
local result=1

#传递数组
myarray=(1 2 3 4 5)
func ${myarray[*]} #如果仅仅传递$myarray的话，就相当于只传递了数组的第一个元素
```


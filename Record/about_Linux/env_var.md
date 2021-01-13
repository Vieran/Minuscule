# Environment Variable

*学习Linux的环境变量*

[A Complete Guide to the Bash Environment Variables ](https://www.shell-tips.com/bash/environment-variables/)

## 什么是环境变量

> 存储有关shell会话和工作环境的信息
>
> 全局变量：对于创建它们的shell和所有生成的子shell都是可见的（但是对于父shell是不可见的
>
> 局部变量：只对创建它们的shell可见

```bash
#查看环境变量的几种命令
printenv HOME #不带参数的时候输出全局变量
env #不带参数的时候输出全局变量
echo $HOME #在变量名前加上$可以显示变量当前的值，而且能够让变量作为命令行参数
set #显示为某个特定进程设置的所有环境变量，包括全局、局部、用户自定义变量
```



## 对变量几种的操作

```bash
#所有系统环境变量都使用大写字母，这是bash shell的惯例；所以用户自定义变量应该使用小写字母，避免重名
#变量名、等号和值之间没有空格！
#如果要用到变量，使用$；如果要操作变量，不使用$
my_var="hello world" #设置一个局部环境变量
export my_var #将my_var设置为全局变量
unset my_var #删除环境变量
```



## 其他

```bash
#PATH用于搜索可执行文件
PATH=$PATH:. #将当前路径添加到PATH中（这里只是局部变量，export之后才变成全局变量

#数组变量（给某个环境变量设置多个值，不太常用
my_var=(one two three four) #把值放在括号里，使用空格分开
echo ${my_var[2]} #输出数组的下标为2的值
echo ${my_var[*]} #输出整个数组变量
unset my_var[1] #将my_var[1]置为空
unset my_var #删除整个数组变量
```


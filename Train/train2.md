# 培训2

2020.11.27 晚上 斗鱼直播间



## 环境变量

```bash
#通过export改变环境变量（仅仅对当前的shell管用
export HOME=/ #将home改成了根目录

#常见的几种环境变量
PATH #完整的可执行文件的搜索路径
CPATH #当指定-I参数的时候，会到这个路径下搜库索文件
LIBRARY_PATH #用于gcc的静态库链接（在设置环境变量的时候会用到
LD_LIBRARY_PATH #用于gcc的动态库链接（在设置环境变量的时候会用到

#环境变量的初始脚本是.bashrc，在用户登陆的时候执行，但是请不要在这个文件夹里设置很heavy的环境变量（尽量不要改动，除非你知道自己在做什么
```



## 管理环境

1. 创建一个诸如packages问价夹来管理所有下载的安装包

2. 创建一个诸如apps的文件夹来存放安装的软件

   *注意软件的命名方式，需要包含软件包版本、编译器版本等关键信息*

3. 创建一个类似于benchmarks的文件夹，用于管理测试集群的软件包，随时测试集群是否正常工作

4. 在编译的时候使用一个干净的build文件夹，进去文件中执行configure，这样出错的时候可以直接删除那些出错的文件，不会污染源文件夹的环境

5. 在编写可执行文件的时候，尽量使用统一的后缀，便于识别（比如.x后缀



## 其他

```bash
#执行命令的时候，最好加上2>&1选项
#Linux中0表示stdin标准输入，1表示stdout标准输出，2表示stderr标准错误，而>表示重定向，这里写的&符号可以减少打开文件的次数且不会导致2的输出把1的覆盖掉（详细了解请自行搜索
./configure | tee xxx.log 2>&1

#在编译的时候使用宏命令
#test.c文件的main函数如下
int main()
{
#ifdef DHELLO
	print("xxx");
#else
	print("yyy");
#endif
	return 0;
}
#在编译的时候，下列语句就使用了DHELLO的宏命令，执行test.x的时候会输出xxx
gcc -o test.x -DHELLO ./test.c

#使用-lha参数查看文件的具体信息（可以查看链接
ls -lha

#编译的时候，使用-j选项可以选择job数目（可能会快一些
make -j #没错，就是没有带数字（其实可以带数字
```

1. 在很多时候，sharp符号（#）表示the number of

2. 修改一些关键文件的时候，要么备份，要么选择注释而不是删除

   *注释的时候打一下备注，诸如时间20201128T0101以及修改原因等*

3. 查看makefile，可以看到make clean之类是语句是如何定义的（原来makefile定义了这么多东西

4. 查找东西是按照path从前到后依次搜索的（搜索到了就停止），为了避免版本冲突，一般新的版本放前面

5. source某个文件和直接执行某个文件的区别（在当前shell有效，开启一个子shell执行文件
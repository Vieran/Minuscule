# Learning Make

### What is make/makefile ?

> 代码的编译（生成中间代码文件）和连接（将中间代码文件合成可执行文件）
>
> make只是一个指令，makefile告诉了make该干什么



### make语法

```bash
#指定按照文件xxx进行构建
make -f xxx
make --file=xxx
```



### makefile语法

*[一个链接，详细地解释了make的基本用法](https://gist.github.com/isaacs/62a2d1825d04437c6f08)*

#### 基本语法

```makefile
#target是目标文件，可以是.o文件，也可以是可执行文件（必须
#prerequisites是生成target所需要的文件/目标
#command是任意shell指令，不和target在同一行的时候必须要tab开头，和target在同一行可以用;作为分隔
<target>: <prerequisites...>
	<commands>
#正常情况下，make会打印每一条command，然后再执行，这被称为echoing（即使是注释内容也会被打印
#使用@可以关闭echoing；由于在构建过程中，需要了解当前在执行哪条命令，所以通常只在注释和纯显示的echo命令前面加上@

#make会比较targets文件和prerequisites文件的修改日期，如果prerequisites文件的日期要比targets文件的日期要新，或者target不存在的话，那么make就会执行后续定义的命令，否则不会执行后续定义的命令

#例子：文件中写了这些，然后执行make等价于执行make tutorial（结果是输出echo的那句话
tutorial:
	@# todo: have this actually run some kind of tutorial wizard?
	@echo "Please read the 'Makefile' file to go through this tutorial"

# $$  A literal $ character inside of the rules section
#     More dollar signs equals more cash money equals dollar sign.
```



#### 定义变量

```makefile
#make首先使用makefile语法解析makefile的每一行，然后结果传递到shell，所有这里需要两个$$来调用shell变量
#对比下列两个写法，第一个没有保存变量foo，因为每一个command都是独立的，除非使用;表示连续，并且注意换行符\
var-lost:
	export foo=bar
	echo "foo=[$$foo]"

var-kept:
	export foo=bar; \
	echo "foo=[$$foo]"

#内置变量（跨平台兼容性
$(CC) #指向当前使用的编译器
$(MAKE) #指向当前使用的make工具

#自定义变量后续使用的时候要放在$()中
cc = gcc
prom = xxx
source = main.c fetch.c stack.c
$(prom):$(source)
	$(cc) -o $(prom) $(source)
	
#自动变量（值与当前的规则有关
#$@ 指代当前构建的target（@像a，argument
#指令创建a和b文件
a b:
	touch $@
# $@  The file that is being made right now by this rule (aka the "target")
#     You can remember this because it's like the "$@" list in a
#     shell script.  @ is like a letter "a" for "arguments.
#     When you type "make foo", then "foo" is the argument.

#$< 指代第一个前置条件
#将b文件复制生成a文件
a: b c
	cp $< $@
# $<  The input file (that is, the first prerequisite in the list)
#     You can remember this becasue the < is like a file input
#     pipe in bash.  `head <foo.txt` is using the contents of
#     foo.txt as the input.  Also the < points INto the $
	
#$? 指代比目标更新的所有前置条件，以空格分隔
#p2的时间戳比t新，则$?指代p2
t: p1 p2
# $?  All the input files that are newer than the target
#     It's like a question. "Wait, why are you doing this?  What
#     files changed to make this necessary?"

#$^ 指代所有前置条件，以空格分隔
#这里$^指代p1 p2
t: p1 p2
# $^  This is the list of ALL input files, not just the first one.
#     You can remember it because it's like $<, but turned up a notch.
#     If a file shows up more than once in the input list for some reason,
#     it's still only going to show one time in $^.

#$* 指代匹配符%匹配的部分
#这里文件为f1.txt，则$*就表示f1
%.txt
# $*  The "stem" part that matched in the rule definition's % bit
#     You can remember this because in make rules, % is like * on
#     the shell, so $* is telling you what matched the pattern.

#$(@D) 和 $(@F) 分别指向$@的目录名和文件名
#$(<D) 和 $(<F) 分别指向$<的目录名和文件名
#$@是src/input.c，则$(@D)为src，$(@F)为input.c
# You can also use the special syntax $(@D) and $(@F) to refer to
# just the dir and file portions of $@, respectively.  $(<D) and
# $(<F) work the same way on the $< variable.  You can do the D/F
# trick on any variable that looks like a filename.

#关于变量赋值的扩展（动态/静态
VARIABLE = value #在执行时扩展，允许递归扩展
VARIABLE := value #在定义时扩展
VARIABLE ?= value #只有在该变量为空时才设置值
VARIABLE += value #将值追加到变量的尾端

#一个例子
dest/%.txt: src/%.txt
    @[ -d dest ] || mkdir dest
    cp $< $@
#上面代码将src目录下的txt文件，拷贝到dest目录下
#首先判断dest目录是否存在，如果不存在就新建
#$<指代前置文件（src/%.txt），$@指代目标文件（dest/%.txt）
```



#### 判断和循环

```makefile
#makefile使用bash指令完成判断和循环
#一个判断的例子
ifeq ($(CC),gcc)
  libs=$(libs_for_gcc)
else
  libs=$(normal_libs)
endif
#上面代码判断当前编译器是否为gcc，然后指定不同的库文件

#一个循环的例子
LIST = one two three
all:
    for i in $(LIST); do \
        echo $$i; \
    done
#运行结果（每个之间是换行不是空格）：one two three
```



#### 函数

```makefile
#makefile函数的使用的两种格式
$(function arguments)
${function arguments}

#内置函数
#shell 函数用于执行shell指令
srcfiles := $(shell echo src/{00..99}.txt) #这里输出src/{00..99}.txt

#wildcard 函数用来在Makefile中，替换Bash的通配符
srcfiles := $(wildcard src/*.txt)

#subst 函数用来文本替换
#格式：$(subst from,to,text)
$(subst ee,EE,feet on the street) #将feet on the street替换成fEEt on the strEEt

#patsubst 函数用于模式匹配的替换
#格式：$(patsubst pattern,replacement,text)
$(patsubst %.c,%.o,x.c.c bar.c) #将x.c.c bar.c换成x.c.o bar.o

#替换后缀名函数的写法是：变量名 + 冒号 + 后缀名替换规则（实际上是patsubst的简写形式
min: $(OUTPUT:.js=.min.js) #这里将变量OUTPUT中的后缀名.js全部替换成.min.js 

#一个例子
comma:= ,
empty:=
# space变量用两个空变量作为标识符，当中是一个空格
space:= $(empty) $(empty)
foo:= a b c
bar:= $(subst $(space),$(comma),$(foo))
# bar is now 'a,b,c'
```



### 完整的实例

```makefile
cc = gcc
prom = calc
obj = $(src:%.c=%.o) 

#找出当前目录下所有的.c和.h文件
deps = $(shell find ./ -name "*.h")
src = $(shell find ./ -name "*.c")
 
$(prom): $(obj)
    $(cc) -o $(prom) $(obj)
 
%.o: %.c $(deps)
    $(cc) -c $< -o $@
 
clean:
    rm -rf $(obj) $(prom)
```


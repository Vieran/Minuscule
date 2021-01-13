# Learning CMake

### What is cmake ?

> 平台无关、书写简单
>
> 作用是生成makefile



### CMake语法

```bash
#执行命令生成makefile
cmake

#使用make进行编译
make
```



### CMakeList.txt

*命令不区分大小写*

#### 基本内容

```cmake
#在顶层CMakeList.txt
#指定cmake版本
cmake_minimum_required(VERSION 3.10)

#指定项目名称为xxx
project(xxx)

#指定构建目标app的源文件（将yyy文件编译成为zzz的可执行文件
add_executable(zzz yyy)
```



#### 添加库文件

```cmake
#在顶层CMakeList.txt
#在顶层添加add_subdirectory的调用
include_directories("${PROJECT_SOURCE_DIR}/src")

#添加src子目录
add_subdirectory(src)

#添加src目录下的库文件libra（指明main函数需要链接这个库
target_link_libraries(src libra)

#在src目录下的CMakeList.txt
aux_source_directory(. DIR_LIB_SRCS) #查找当前目录下的所有源文件，并将名称保存到DIR_LIB_SRCS变量
add_library(libra ${DIR_LIB_SRCS}) #生成链接库libra（这样顶层文件才可以调用
```



#### 自定义编译选项

```cmake
#加入一个配置头文件head.h，这个文件由CMake从head.h.in生成，通过这样的机制将可以通过预定义一些参数和变量来控制代码的生成
configure_file(
"${PROJECT_SOURCE_DIR}/head.h.in"
"${PROJECT_BINARY_DIR}/head.h"
)

# 是否使用自己的libra库
option(USE_MYLIBRA "Use provided libra" ON)

#是否加入libra库
if(USE_MYLIBRA)
	include_directories("${PROJECT_SOURCE_DIR}/math")
	add_subdirectory(src)
	set(EXTRA_LIBS ${EXTRA_LIBS} libra)
endif (USE_MYLIBRA)

#在main函数中添加下列命令，让其根据USE_MYLIBRA的预定义值来决定是否调用LIBRA库（这里是cpp代码
#ifdef USE_MYLIBRA
#include "src/head.h"
#endif

#上面引入了head.h文件，但是我们不直接编写这个文件，为了方便从cmakelists.txt导入配置，写一个head.h.in文件（这样cmake会自动根据配置文件生成head.h
#cmakedefine USE_MYLIBRA
```



#### 更多内容

> [CMake的官方文档](https://cmake.org/cmake/help/latest/genindex.html)
>
> [CSDN上一个较为详细的教程](https://blog.csdn.net/zhuiyunzhugang/article/details/88142908)
>
> [某个博客（思路和别的不一样）](https://aiden-dong.github.io/2019/07/20/CMake%E6%95%99%E7%A8%8B%E4%B9%8BCMake%E4%BB%8E%E5%85%A5%E9%97%A8%E5%88%B0%E5%BA%94%E7%94%A8/)
>
> [cmake用法及常用命令总结](https://www.cnblogs.com/ZY-Dream/p/11232779.html)


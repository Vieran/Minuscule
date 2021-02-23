# 代码的奇技淫巧

*主要介绍一些代码优化中的技巧*

## 宏定义

**本质：代码替换**

### 基础语法

```c
/*不带参数的宏定义*/
#define <identifier> <list_to_replace>
//注意使用括号，因为宏定义是直接替换的
#define N (1+2)

//多行使用\分隔
#define USA "the United \
States of \
America"

/*带参数的宏定义*/
#define <identifier>(para1, para2, ...) <list_to_replace>
//标识符和参数之间不能存在空格
#define defineHalfBlockSize(g, halfBlock) \
    const int64_t halfBlock = (1LL << g->targetQubit) - 1
//参数为空的宏
#define getchar() getc(stdin)

/*删除宏定义*/
#undef <identifier>
```

### 例子

```c
//循环展开：减少分支，利用流水线并行，编译器自动向量化
#define LOOP_UNROLL_DOUBLE(action, actionx2, width) do { \
	unsigned long __width = (unsigned long)(width); \
	unsigned long __increment = __width >> 2; \
	for (; __increment > 0; __increment--) { \
		actionx2; \
		actionx2; \
	} \
	switch (__width & 3) { \
		case 1: action; break; \
		case 2: actionx2; break; \
	} \
} while (0)
```

[知乎：c语言有什么奇技淫巧](https://www.zhihu.com/question/27417946/answer/1253126563)



## 进程绑定

**MPI进程绑定到指定的核上，以减少通信的开销**

```
//
```

[OpenMPI doc](https://www.open-mpi.org/doc/v3.1/man1/mpirun.1.php#sect3)

[HPC Wiki: Bingding/Pinning](https://hpc-wiki.info/hpc/Binding/Pinning)

[StackOverflow: Processor/socket affinity in OpenMPI](https://stackoverflow.com/questions/17604867/processor-socket-affinity-in-openmpi)




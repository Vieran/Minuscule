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

**tips for reading code: `代码流程`+`数据排布`**



## 进程绑定

**MPI进程绑定到指定的核上，以减少通信的开销**

```bash
# 到MPI那份文档去看
```

[OpenMPI doc](https://www.open-mpi.org/doc/v3.1/man1/mpirun.1.php#sect3)

[HPC Wiki: Bingding/Pinning](https://hpc-wiki.info/hpc/Binding/Pinning)

[StackOverflow: Processor/socket affinity in OpenMPI](https://stackoverflow.com/questions/17604867/processor-socket-affinity-in-openmpi)



## 并行tricks

1. **并行数据结构**：很多代码中使用了map、queue、stack等数据结构，但是基础的数据结构大部分是非线程安全的，而大量数据情况下，会耗费大量时间，而一些已经实现好了的线程安全的数据结构是很好的替代品

   > [有靠谱的并发哈希表C/C++实现吗？ - 左睿的回答 - 知乎](https://www.zhihu.com/question/46156495/answer/583947454)

2. **并行解压/压缩**：大量数据的读写经常需要解压数据和压缩数据，而并行地进行会能够得到很好的加速（pugz）

3. **使用现成的加速库**：对于数值运算（矩阵乘之类的常见运算）已经存在很多性能很好的库，直接使用比手写一个来的快多了

4. **内存映射(MMIO)/写入tmp目录**：内存映射让进程可以像访问内存一样对普通文件操作，减少读写文件的IO耗时；程序需要读写中间文件的时候，可以将文件放在/dev/shm目录下（内存映射目录），可以获得像内存映射一样高效的IO

5. **并行MPI通信**：MPI可以多线程通信（MPI_Init_thread、MPI_THREAD_MULTIPLE）

6. **向量化**：向量化，可以做的不仅仅是简单的加减乘除


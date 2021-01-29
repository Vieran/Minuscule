# Basic GPU Learning

*从零开始学习GPU基础*

- [x] GPU和CPU的区别
- [x] GPU如何执行运算工作（CUDA程序运行的大致流程
- [x] CUDA编程基础
- [ ] GPU结构

[CUDA TOOLKIT DOC](https://docs.nvidia.com/cuda/index.html)

## 几个名词

> **C**entral **P**rocessing **U**nit简称CPU，中央处理器，负责解释计算机指令以及处理计算机软件中的数据
>
> **G**raphics **P**rocessing **U**nit简称GPU，图形处理器，具有运算和数据处理的能力（可以理解为地位和CPU对等）
>
> **C**ompute **U**nified **D**evice **A**rchitecture简称[CUDA](https://baike.baidu.com/item/CUDA)，统一计算架构，是由NVIDIA推出的一种集成技术（或者说是一种编程模型，给各种语言提供了编程的接口，使之可以构建基于GPU计算的应用程序
>
> **H**eterogeneous **C**omputing简称HC，异构计算，使用不同类型[指令集](https://zh.wikipedia.org/wiki/指令集)和体系架构的计算单元组成系统的计算方式（GPU并非独立运行的计算平台，需要与CPU协同工作，可以看成是CPU的协处理器，因此所说的GPU并行计算，其实是指的基于CPU+GPU的异构计算架构
>
> 1. GPU的执行速度远低于CPU，但是GPU的优势是[显存带宽](https://baike.baidu.com/item/%E6%98%BE%E5%AD%98%E5%B8%A6%E5%AE%BD)高（多个执行单元
> 2. GPU适合执行并行计算

[知乎：cpu和gpu的区别是什么](https://www.zhihu.com/question/19903344)*这里面提到了CPU和GPU分别适合做什么样的工作*



## CUDA编程概念

> 1. 异构模型，需要CPU和GPU协同工作
>
> 2. CUDA本质是多线程（每个线程会分配唯一的线程ID，这个ID值可以通过核函数的内置变量`threadIdx`来获得）
>
> 3. CUDA程序中包含host程序和device程序（`host`指代CPU及其内存，`device`指代GPU及其内存），分别在CPU和GPU上运行
>
> 4. GPU的一个核心组件是`SM(Streaming Multiprocessor)`，SM的基本执行单元是`线程束(warps)`；线程束包含很多个线程（白皮书），这些线程执行相同的指令（`SMIT(Single Instruction Multiple Thread)`），但是它们拥有单独的指令地址计数器和寄存器状态，也有自己独立的执行路径，所以尽管线程束中的线程同时从同一程序地址执行，但是可能具有不同的行为
> 5. 每个线程块必须在一个SM上执行（单个SM的资源有限，这导致线程块中的线程数是有限制的，白皮书可查），线程块在SM上进一步被划分为多个线程束
> 6. 每个线程都调用kernel（核）函数，`<<<grid, block>>>`指定线程数量为grid * block（其实就是grid个block，block个线程数），逻辑上这些线程是并行的，但是在物理层受到了SM的限制



## CUDA编程实例

*这里介绍的是CUDA C/CPP*

> 1. 声明核函数：\__global__
> 2. 调用核函数：kernel_fun<<< grid, block >>>(prams...)

```c
#看一小段代码理解一些cuda编程
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

//cuda6.0引入统一内存（Unified Memory）避免再host和device上进行数据的深拷贝
//CUDA中使用cudaMallocManaged函数分配托管内存：
//cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flag=0);

__global__ void add(float* x, float* y, float* z, int n) {
	//access the global index
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//threads per grid
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		z[i] = x[i] + y[i];
}

int main() {
	int N = 1 << 20;
	int nBytes = N * sizeof(float);

	//alloc memory for host
	float *x, *y, *z;
	cudaMallocManaged((void**)&x, nBytes);
	cudaMallocManaged((void**)&y, nBytes);
	cudaMallocManaged((void**)&z, nBytes);

	//initial the data
	for (int i = 0; i < N; i++) {
		x[i] = 10.0;
		y[i] = 20.0;
	}

	//setting for kernel running
	dim3 blockSzie(256);
	dim3 gridSize((N + blockSzie.x - 1) / blockSzie.x);
	//execute kernel
	add <<<gridSize, blockSzie >>> (x, y, z, N);

	//同步device，确保结果能够正确访问
	cudaDeviceSynchronize();

	//check the result
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = fmaxf(maxError, fabs(z[i] - 30.0));
	printf("maxError: %f\n", maxError);

	//free the memory of device
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	return 0;
}
```

[知乎：CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)*讲得非常通俗易懂*

[CUDA编程之快速入门](https://www.cnblogs.com/skyfsm/p/9673960.html)

[CUDA tutorial](https://cuda-tutorial.readthedocs.io/en/latest/)



## GPU结构

1. GPU是为追求高带宽（并行）而使用了集成显存，因而显存较小



## 其他

```bash
#查看显卡型号（两种方式
nvcc --version
nvidia-smi

#.cu文件编译和运行（ptx指令是cuda程序编译出来的汇编码
nvcc a.cu -o a.x
./a.x

#命令行下进行profile，并把结果输出到文件xxx
nvprof ./a.x &>xxx
nvprof -o xxx.nvvp ./demo #这也输出的文件在图形界面可以看

#图形界面进行profile（两种工具，后者是新的，更推荐使用
nvvp
nsight
```

[知乎：关于CUDA的一些概念](https://zhuanlan.zhihu.com/p/91334380)

[是时候用nsight分析优化工具了](https://cloud.tencent.com/developer/article/1468566)

[NVVP toolkit doc](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
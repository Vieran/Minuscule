# Learning openMP

## What is it ?

> open multi-processing
>
> 便携、可扩展，使得在C++和Fortran中的并行编程更加简单和灵活
>
> 适用于多核节点和芯片，NUMA系统，GPU以及连接到CPU的其他设备的算法
>
> 用于共享内存并行系统的多处理器程序设计的指导性的编译处理方案



## C++

**fork/join并行执行模式**:并行执行的程序要全部结束后才会运行后面非并行部分的代码 

#### 基本知识

```cpp
//包含头文件omp.h
#include <omp.h>

//所有并行块由#pragma omp开头的编译制导语句来开始，将{}中的代码分配到每个线程中去执行
#pragma omp parallel [for | sections] [子句[子句]…]
{
    //指定需要并行计算的代码（当只有一个语句的时候，可以直接写而不需要大括号
}

//将for循环分配到每个线程去执行
#pragma omp parallel for

//并行部分的代码一次只能由一个线程执行，相当于取消了并行化
#pragma omp critical

//同步并行线程，让线程等待，直到所有的线程都执行到该行
#pragma omp barrier

//将并行块内部的代码划分给线程组中的各个线程
//一般在内部嵌套几个独立的section语句，每个分块都会并行执行
//可以使用nowait来停止等待
#pragma omp [parallel] sections [子句]
{
	#pragma omp section
	{
		//section的代码
	}
    //上述的section可以有多个，此处不再赘述
}

//线程数量相关
omp_set_num_threads(x);  //函数来手动设置线程数为x
omp_get_thread_num();  //来获取当前线程的编号
omp_get_num_threads();  //来获取线程总数
```


```c++
//例子
#include <stdio.h>
#include <omp.h>
int main(int argc, char* argv[])
{
	//将程序并行化了，输出的i并不是从0到9顺序排列的
    #pragma omp parallel for
	for (int i = 0; i < 10; i++)
		printf("i = %d\n", i);
/*
	//上面这句话也可以这么写，但是奇怪的事情是，{不能和parallel同一行
    #pragma omp parallel
    {
	for (int i = 0; i < 10; i++)
		printf("i = %d\n", i);
	}
*/
    
    //写了并行之后，同时输出六个线程；不写就只输出一个
    #pragma omp parallel num_threads(6)
    printf("Thread: %d\n", omp_get_thread_num());
    
    //section模块的使用
    #pragma omp parallel sections
	{
		#pragma omp section 
		printf("Section 1 ThreadId = %d\n", omp_get_thread_num());
		#pragma omp section
		printf("Section 2 ThreadId = %d\n", omp_get_thread_num());
		#pragma omp section
		printf("Section 3 ThreadId = %d\n", omp_get_thread_num());
		#pragma omp section
		printf("Section 4 ThreadId = %d\n", omp_get_thread_num());
	}
	return 0;
    
	return 0;
}
```



#### private

```c++
//private可以将变量声明为线程私有
//声明称线程私有变量以后，每个线程都有一个该变量的副本，线程之间不会互相影响，其他线程无法访问其他线程的副本；原变量在并行部分不起任何作用，也不会受到并行部分内部操作的影响
int i;
#pragma omp parallel for private(i)
for (i = 0; i < 10; i++)
{
	printf("i = %d\n", i);
}
printf("outside i = %d\n", i);

//firstprivate可以使得线程私有变量继承原来变量的值
int t = 4;
#pragma omp parallel for firstprivate(t)
for (int i = 0; i < 5; i++)
{
	t += i;
	printf("t = %d\n", t);
}
printf("outside t = %d\n", t);

//lastprivate可以在退出并行部分时将计算结果赋值回原变量
/*
在循环迭代中，是最后一次迭代的值赋值给原变量；
在section结构中，那么是程序语法上的最后一个section语句赋值给原变量；
在类(class)变量作为lastprivate的参数时，我们需要一个缺省构造函数，除非该变量也作为firstprivate子句的参数；此外还需要一个拷贝赋值操作符
*/
//将上面firstprivate换成lastprivate即可

//threadprivate可以将一个变量复制一个私有的拷贝给各个线程，即各个线程具有各自私有的全局对象
//格式为#pragma omp threadprivate(list)
#include <stdio.h>
#include <omp.h>
int g = 0;
#pragma omp threadprivate(g)
int main(int argc, char* argv[])
{
	int t = 20, i;
	#pragma omp parallel
	{
		g = omp_get_thread_num();
	}
	#pragma omp parallel
	{
        //输出大概就是thread id 和 g 是一一对应的
		printf("thread id: %d g: %d\n", omp_get_thread_num(), g);
	}
/*
	//上面这段代码中，可以将两个parallel部分合并成为一个，也就是
	#pragma omp parallel
	{
		g = omp_get_thread_num();
    	printf("thread id: %d g: %d\n", omp_get_thread_num(), g);
	}
	//所以这里可以合理猜想，一个函数（这里是main）中只能有一种线程分割吗？（也就是这里的g和thread id一一对应的原因？
*/
	return 0;
}
```



#### shared

```cpp
//shared可以将一个变量声明成共享变量，并且在多个线程内共享
//需要注意的是，在并行部分进行写操作时，要求共享变量进行保护，否则不要随便使用共享变量，尽量将共享变量转换为私有变量使用
int t = 20;
#pragma omp parallel for shared(t)
for (int i = 0; i < 10; i++)
{
	if (i % 2 == 0)
		t++;
	printf("i = %d, t = %d\n", i, t);
}
```



#### reduction

```cpp
//reduction可以对一个或者多个参数指定一个操作符，然后每一个线程都会创建这个参数的私有拷贝，在并行区域结束后，迭代运行指定的运算符，并更新原参数的值
//私有拷贝变量的初始值依赖于redtution的运算类型
//使用方式：reduction(operator:list)
int i, sum = 10;
#pragma omp parallel for reduction(+: sum)
for (i = 0; i < 10; i++)
{
	sum += i;
	printf("%d\n", sum);
}
printf("sum = %ld\n", sum);
```



#### copyin

```cpp
//copyin可以将主线程中变量的值拷贝到各个线程的私有变量中，使之初始化的值与主线程中的一致
//其参数必须要被声明称threadprivate，对于类的话则并且带有明确的拷贝赋值操作符
int g = 0;
#pragma omp threadprivate(g) 
int main(int argc, char* argv[])
{
	int i;
	#pragma omp parallel for
	for (i = 0; i < 4; i++)
	{
		g = omp_get_thread_num();
		printf("thread %d, g = %d\n", omp_get_thread_num(), g);
	}
	printf("global g: %d\n", g);
    
	#pragma omp parallel for copyin(g)
	for (i = 0; i < 4; i++)
		printf("thread %d, g = %d\n", omp_get_thread_num(), g);
	return 0;
}
```



#### 静态调度

```cpp
//当parallel for没有带schedule时，大部分情况下系统都会默认采用static调度方式
//假设有n次循环迭代，t个线程，那么每个线程大约分到n/t次迭代
//这种调度方式会将循环迭代均匀的分布给各个线程，各个线程迭代次数可能相差1次
//使用方式：schedule(method)
int i;
#pragma omp parallel for schedule(static)
for (i = 0; i < 100; i++)
{
	printf("i = %d, thread %d\n", i, omp_get_thread_num());
}

//在静态调度的时候，可以通过指定size参数来分配一个线程的最小迭代次数
//指定size之后，每个线程最多可能相差size次迭代，[0,size-1]的迭代是在第一个线程上运行，依次类推
//使用方式：schedule(static, size)
int i;
#pragma omp parallel for schedule(static, 3)
for (i = 0; i < 10; i++)
{
	printf("i = %d, thread %d\n", i, omp_get_thread_num());
}
```



#### 动态调度

```cpp
//动态分配是将迭代动态分配到各个线程，依赖于运行状态来确定
int i;
#pragma omp parallel for schedule(dynamic)
for (i = 0; i < 10; i++)
{
	printf("i = %d, thread %d\n", i, omp_get_thread_num());
}
return 0;
```



#### 线程相关的信息

```cpp
//返回调用函数时可用的处理器数目
//int omp_get_num_procs(void);

//返回当前并行区域中的活动线程个数，如果在并行区域外部调用，返回1
//int omp_get_num_threads(void);
printf("%d\n", omp_get_num_threads());  //这句打印1
#pragma omp parallel  
{
    printf("%d\n", omp_get_num_threads());
}

//返回当前的线程号（从0开始
//int omp_get_thread_num(void);

//设置进入并行区域时，将要创建的线程个数
//int omp_set_num_threads(int);
omp_set_num_threads(4);
#pragma omp parallel  
{
	printf("%d of %d threads\n", omp_get_thread_num(), omp_get_num_threads());
}

//可以判断当前是否处于并行状态
//int omp_in_parallel();

//获取最大的线程数量（是确定的值，与其在并行区域内外没有关系，但是使用omp_set_num_threads会造成影响
//int omp_get_max_threads();

//设置是否允许在运行时动态调整并行区域的线程数（参数为0则禁止，为1则系统自动调整以最佳利用资源；默认禁止
//void omp_set_dynamic(int);

//返回当前程序是否允许在运行时动态调整并行区域的线程数（返回0说明禁止，1说明允许
//int omp_get_dynamic();
```



#### 互斥锁

```cpp
//Openmp中有提供一系列函数来进行锁的操作，一般来说常用的函数的下面4个
void omp_init_lock(omp_lock);  //初始化互斥锁
void omp_destroy_lock(omp_lock);  //销毁互斥锁
void omp_set_lock(omp_lock);  //获得互斥锁
bool omp_test_lock(omp_lock);  //获得互斥锁（非阻塞版本
void omp_unset_lock(omp_lock);  //释放互斥锁

//一个完整的例子
#include <stdio.h>
#include <omp.h>
static omp_lock_t lock;
int main(int argc, char* argv[])
{
    int i;
	omp_init_lock(&lock); 
	#pragma omp parallel for   
	for (i = 0; i < 5; ++i)
	{
		omp_set_lock(&lock);
		printf("%d+\n", omp_get_thread_num());
		printf("%d-\n", omp_get_thread_num());
		omp_unset_lock(&lock); 
	}
	omp_destroy_lock(&lock);
	return 0;
}
//确保在上锁的时候，其他线程也不能执行（也就是，必须一个线程执行完，另外一个才可以执行

//获得互斥锁的非阻塞版本
#include <stdio.h>
#include <omp.h>
static omp_lock_t lock;
int main(int argc, char* argv[])
{
    int i;
	omp_init_lock(&lock); 
	#pragma omp parallel for   
	for (i = 0; i < 5; ++i)
	{
		if (omp_test_lock(&lock))
		{
			printf("%d+\n", omp_get_thread_num());
			printf("%d-\n", omp_get_thread_num());
			omp_unset_lock(&lock);
		}
		else
		{
			printf("fail to get lock\n");
		}
	}
	omp_destroy_lock(&lock);
	return 0;
}
//可能会出现一个线程获得互斥锁失败的情况
```



#### 小疑问以及思考

```cpp
//这是标答
	#pragma omp parallel 
	{
	    int i, j;
		#pragma omp for  //这里是说接下来有一个并行的for循环
		for (i = 0; i < 5; i++)
			printf("i = %d\n", i);
		#pragma omp for  //这里也是说接下来有一个并行的for循环
		for (j = 0; j < 5; j++)
			printf("j = %d\n", j);
	}

//这是我作了修改之后的（就是想要实践一下是不是只有在大括号内的才是并行？
	#pragma omp parallel 
	{
	    int i, j;
		#pragma omp parallel  //这里说明大括号里面是一个并行的模块（这里是for循环，它是第二层并行了
		{
		for (i = 0; i < 5; i++)
			printf("i = %d\n", i);
		}
        //这里是第一层并行
		for (j = 0; j < 5; j++)
			printf("j = %d\n", j);
	}
```



## 编译运行

```bash
#需要加-fopenmp才使得真正并行
mpicc -fopenmp yyy.c -o yyy.x

export OMP_NUM_THREADS=20 #设置默认的线程数目（如果不写这个应该是跑满全部核心 
```


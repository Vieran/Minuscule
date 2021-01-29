# 理论性能分析

论文：Roofline: An Insightful Visual Performance Model for Multicore Architectures

## 前提

计算理论的性能峰值$$Attainable\ GFlops/sec = min 
\left\{\begin{matrix}
Peak Memory Operational
Bandwidth \times Intensity{\normalsize } \\\mathrm{} Peak Floating-Point
Performance
\end{matrix}\right.$$

## roofline优化

### 计算瓶颈

> 1. **Improve instruction-level parallelism (ILP) and apply SIMD **
>
>    提高指令级的并行，使用向量化（SIMD）
>
> 2. **Balance floating-point operation mix**
>
>    均衡浮点混合计算，最好是一乘一加搭配

### 内存瓶颈

> 1. **Restructure loops for unit stride accesses**
>
>    ？
>
> 2. **Ensure memory affinity**
>
>    将数据和负责该数据的线程分配给同一个memory-processor，这样就不需要跨芯片（chip）访问内存
>
> 3. **Use software prefetching**
>
>    使操作尽可能在内存中进行；在某些计算机上，软件预取比单独的硬件预取提供更多的带宽

*论文中的figure2展示了这些优化对性能的影响，并且文章详细说明了什么情况下应该进行哪些优化而哪些优化可以跳过*



## 3Cs优化

*3Cs模型：compulsory, capacity, and conflict misses*

*操作强度：operational intensity；此段均是在操作强度不变的假设下进行讨论的*

> 1. 论文举例说明了几种改善3Cs的方法（但是我看不懂
> 2. Generally, we advise improving operational intensity of the kernel before implementing other optimizations. 



## 其他

*这部分是我没有仔细阅读的内容（因为看不懂or暂时没有必要关心）*

1. 几种常见的处理器的特点以及它们的roofline
2. 关于roofline模型的谬论



[知乎：Roofline Model与深度学习模型的性能分析](https://zhuanlan.zhihu.com/p/34204282)
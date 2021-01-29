# 向量化

*简单介绍计算机的计算向量化*

## 什么是向量化？

> Vectorization is the more limited process of converting a computer program from a scalar implementation, which processes a single pair of operands at a time, to a vector implementation which processes one operation on multiple pairs of operands at once.																																															--wikipedia
>
> 简而言之，就是一条指令操作多个数据（SIMD），可以认为向量化是指令级的数据并行



## 如何实现向量化？

> 主流的处理器都支持向量指令集（SIMD，Single Instruction Multiple Data），比如X86处理器的SSE/AVX指令集，ARM的NEON指令集，NVIDIA的CUDA和开发的OpenCL标准则通过层次优化的编程模型（SMIT，Single Instruction Multiple Threads）来使一份代码既支持多核并行又支持向量化
>
> 其中AVX指令集可以通过官网查询[Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=6)



## AVX指令编程

```cpp
//换算关系：1 byte = 8 bit；double 64bit，long 64bit，float 32bit，int 32bit，short 16bit，char 8bit

//AVX中常见的向量前带__，中间写着这个向量的的大小（单位是bit），最后写着向量的数据类型
__m512d;   //包含8个double类型的512bit的向量（double类型占64bit）
__m128;  //包含4个float类型的128bit的向量（默认是float类型，占32bit）
__m256i;  //包含数个整-型的256bit的向量（整型可以是(unsigned )char、short、int、long

//常见的函数格式：返回向量类型、函数名、输入数据类型
_mm<bit_width>_<name>_<data_type>;
//<data_type>里面的东西比较多，可以自行查看参考的资料

//使用AVX512指令集的时候，编译选项需要添加-AVX512或者-AVX2（可以通过官网查询找到

//例子
#include <immintrin.h>
__m512d b000 = _mm512_load_pd((*(*(b+k)+j)+i + 8)); //解引用取出b[k][j][i]
b000 = _mm512_mul_pd(_c0_, a000);  //将_c0_和a000相加并赋值给b000
b000 = _mm512_fmadd_pd(_c1_, a00_1, b000);  //将_c1_和a00_0相乘，加上b000并赋值给b000
_mm512_storeu_pd((*(*(b+k)+j)+i), b000);  //将结果保存回到b[k][j][i]中

//如果使用Intel编译器编译，需要使用mpiicc（类似于gcc
//注意事项：循环内展开的层数需要注意，在循环计数器的变化那里需要座对应的修改
```

[知乎上对AVX编程的简单介绍](https://zhuanlan.zhihu.com/p/94649418)


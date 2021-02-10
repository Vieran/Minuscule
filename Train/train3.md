# 培训3

2020.12.16 下午 NIC-401



### LU分解

具体是那部分的计算最耗时间---矩阵乘法（混合乘加运算）



### pivot（首元）的选取---partial pivot

保证不会出现分母为0的情况---将最大的元素放在矩阵第一列（确保除法进行后<1）



NB是矩阵分块的大小；更新列的L，更新行的U，更新剩下的大矩阵；主要的时间消耗在最后的大矩阵



### 参考资料

[algorithm of hpl](http://www.netlib.org/benchmark/hpl/algorithm.html)

《matrix》3-1.2.4.6



### vtune的使用

1. 代码逻辑：主线---vtune的start address（某本书《加密与解密》
2. zoom in and filter in
3. top-down----source function stack，展现代码调用函数的层次
4. 循环展开和循环融合P199（考察的主要内容）---匹配cache，空间局部性
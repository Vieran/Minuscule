# 稀疏矩阵基础

*学习稀疏矩阵的存储和运算优化*

## 存储

*稀疏矩阵的存储又称矩阵压缩*

1. 三元组（coordinate format）表示法：仅存储非零元素、其行下标和列下标

2. 行逻辑链接（compressed sparse row）的顺序表：相比三元组，增加了一个数组以存储矩阵`每行第一个非零元素在一维数组中的位置`

   *提高了提取数据时遍历数组的效率；还有一种是列逻辑链接（compressed sparse column）*

3. 十字链接表：链表+数组存储矩阵

   *便于对矩阵进行插入和删除*



## 矩阵乘优化

### OSKI

[稀疏矩阵乘法优化之OSKI](https://zhuanlan.zhihu.com/p/342711915)

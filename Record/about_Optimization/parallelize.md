# Parallelize

*简单介绍并行算法*



## 七类数值方法

*七个小矮人*

> 1. 密集线性代数
> 2. 系数线性代数
> 3. 谱方法
> 4. N体方法
> 5. 结构化网格
> 6. 非结构化网格
> 7. 蒙特卡洛法
>
> **新添**：图遍历、有限状态机、组合逻辑和统计机器学习



## 常见并行算法

| 通用类并行算法                      | 例子                 | 通用类并行算法              | 例子                             |
| ----------------------------------- | -------------------- | --------------------------- | -------------------------------- |
| 分支-聚合（fork-jion）              | OpenMP并行for循环    | 任务数据流（task dataflow） | 广度优先搜索                     |
| 分治法（divide & conquer）          | 快速傅里叶变换、快排 | 置换（permutation）         | 佳能算法、快速傅里叶变换         |
| 管理者-工作者模式（manager-worker） | 简单的自适应网格细化 | 光环交换（holo exchange）   | 有限差分、有限元偏微分方程求解器 |
| 易并行（embarrassingly parallel）   | 蒙特卡洛             |                             |                                  |



## 并行的工具

1. MPI（多进程通信
2. OpenMP（多线程
3. CUDA（GPU多线程



## 经典算法并行化

- [ ] 动态规划（背包问题
- [ ] 快速排序
- [ ] 最短路算法并行（dijkstra
- [ ] 搜索算法（DFS
- [ ] 旅行商问题

在探索上述算法的过程中，学习了其他一些内容

1. 随机行走算法（寻找全局最优解

   [某博客：介绍一个全局最优化的方法：随机游走算法](https://www.cnblogs.com/lyrichu/p/7209529.html)
   
2. 遗传算法（寻找全局最优解

   [简书：超详细的遗传算法解析](https://www.jianshu.com/p/ae5157c26af9)
   
   [知乎：十分钟搞懂遗传算法（含源码）](https://zhuanlan.zhihu.com/p/33042667)
   
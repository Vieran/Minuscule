# 培训1

2020.11.11 下午 网络信息中心401

### 什么是高性能计算

> 广义：高度自动化方法，通过远快于传统的方法解决科学问题
>
> 狭义：汇聚计算资源，以远高于一般计算设备的速度解决科学问题
>
> 衡量计算机性能：FLOPS（floating-point operations per second，每秒浮点运算次数
>
> 实现快速通信：IB（Infini Band，无限宽带
>
> 节点（计算单元）、交换机、节点之间的连接

*超算的特殊性：专业性、单一性、高性能*

*要求：了解服务器底层结构，需要搭建服务器，类比是赛车*



### 计算机体系结构

*any problem in computer science can solved by another layer of indirection*

> **What to learn ?**
>
> 操作系统基本指令、多核开发、*nix内核、集群管理……
>
> 回到计算机的本源（如果最后要做高性能计算的话



### 三大超算赛事

1. ASC（ASC Student Supercomputer Challenge，ASC超算竞赛

   初步准备比较复杂

2. ISC（International Supercomputing Conference Student Cluster Competition，国际大学生超算竞赛

   初步准备比较简单，非常灵活

3. SC（Supercomputing Conference Student Cluster Competition

   初步准备比较简单，超算领域的顶会，对撰写论文要求比较高；挑战性最高

*比赛的要求有共同点*



### 推荐书目

> Linux命令行与shell脚本编程大全
>
> 程序员的自我修养（厕所读物
>
> Linux的鸟哥私房菜（四个虚拟机搭建集群
>
> 大话计算机（科普读物

> high performance computing
>
> introduction to high performance computing for scientists and engineers
>
> matrix computations 3rd edition（应用数学、工程计算--->并行计算
>
> 两个网站---寻找问题的答案，了解Linux系统



### 建议

> 先“动手做”，再“认真学”
>
> 多看文档、多动手写文档
>
> 敲命令的之前，再三检查，想清楚为什么这么做
>
> 写文档的时候，将目标列出来，然后一步一步向ta逼近



### 培训计划

1. HPC基本结构（阅读pi集群文档
2. Linux使用与配置（翻墙+本地安装集群
3. 并行计算软件的编译与运行（Mini-app of ECP Project
4. 性能测评（hpl，hpcg，graph500等等，了解他们的意义
5. 性能分析与评估（文献、书籍阅读，使用VTune分析你之前运行过的程序的性能
6. 代码阅读（阅读分析hpl、hpcg的代码结构，并写出文档
7. 并行开发与优化（并行矩阵乘法、2维热传导问题、优化MiniFE、AMG-1.2
8. 微架构性能分析
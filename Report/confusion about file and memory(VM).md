# CONFUSION & SOLUTION

**file and memory(VM)**



1. **文件是什么？**

   > 文件分为普通文件和匿名文件（定义源自于CSAPP
   >
   > 普通文件：普通的磁盘文件
   >
   > 匿名文件：`内核创建`，包含的全部是二进制零（这个二进制零是什么东西？
   >
   > *也就是，不是实际存在于磁盘上的文件都是匿名文件？包括“所有的程序”（非狭义的程序，而是指所有的可执行的东西）运行时的栈、堆等等？*

2. **虚拟内存**

   > 操作系统提供给用户的`抽象`，使得用户以`一致的方式`“操纵”（不是直接操作，而是通过系统调用来使用）物理内存空间
   >
   > 某种意义上，它实现了“用磁盘空间来`扩展物理内存`"
   >
   > *这里面涉及了一个用户态和内核态的概念，就是用户不是直接操作物理地址的，它需要通过操作系统的底层调用才能实现对一个物理内存上的东西的访问？比如，当访问一个c程序创建的动态（或者静态）数组，实际上系统中发生了什么？*

3. **分页(page)**

   > 分页是操作系统为了`加速`查找和索引内存的一种方式
   >
   > 一个大的内存区域（不管是虚内存还是物理内存），直接进行索引是很麻烦的（$4G = 4 \times 2^{40}$个不同的索引），所以采用分页的方式来进行管理是高效的（$4G = 4KB \times 1K \times 1K \times 1K$，单个页面只需要$4KB = 2^{10}$个不同索引，而取消剩余的$2^{30}$不同索引的代价只是多了三次对页层级的寻址）

4. **文件和虚拟内存**

   > 文件描述符是一个指针，它指向此时所在的文件的位置，而内存空间（诸如堆）在底层也是通过指针来操作的（？）
   >
   > Linux系统里面有一种称为`mmap`的技术，可以用于实现底层的malloc（或者new），即申请堆空间，而mmap返回的是指针，这使得可以像读文件一样去读取内存（或者说读文件本来就是读取内存）。那么问题来了，既然它们都是通过指针来操作的，到底文件和一般的内存有什么区别？
   >
   > **我想：**
   >
   > 这或许就是”`一切皆文件`“。
   >
   > 一般的内存，应该就是开始所讲的两种文件里面的`匿名文件`，它不存在于磁盘上而只在”程序“（广义的程序）运行时存在，它可以是我们所指的”堆“、”栈“（比如程序运行时要使用的”instruction“被载入到了内存中，除非写汇编直接运行，否则这个位置我们是不知道的，我记得8086汇编里面的内存是可以直接读写的，随意独写内存可能会把程序的”instruction“覆盖掉，导致出现问题），我们无法直接访问，但是可以通过一些方式（比如在c语言里面调用函数）得到它的句柄（比如c语言里面的指针），以实现对它们的访问。所以我认为（也正如jsl学长所说），文件是操作系统对内存的一种抽象（或许这句话有失偏颇，但是大概就是这个意思），没有一个实际的东西叫做”文件“，而内存是实实在在的，内存里面写着数据，如果那些数据要很久很久地存在（也就是不仅仅在运行时候存在），那么，我们就给它一个名字，姑且叫做`一般文件`罢（`匿名文件`应该是仅在”短暂“存在的）。

5. **内核态和用户态**

   > 操作系统对底层的资源（比如物理内存）的调用作了隐藏，在操作系统上运行“程序”，是无法直接越过它去访问底层的资源的（即使你写的是汇编也是要经过它的“审批”），它隐藏了的内容，可以理解为“内核态”，是用户看不见的
   >
   > 它对底层的抽象使得“程序”（这里就不称为软件了，直接称为程序，广义的程序）开发更加容易（不需要关心底层的东西，只需要调用操作系统给的接口）
   >
   > *然而有趣的是，不一定需要“操作系统”也可以，比如8086CPU可以直接编译运行汇编——这是因为它的指令集和微架构匹配，也就是芯片并不是只含有一个CPU*
   >
   > 一个危险的想法：在没有操作系统的CPU上（这样就不会被操作系统强制限制），用汇编写一个程序，而且让它在运行的时候随意地修改代码区段，这样就会出错~~（那真是件很有趣的事情~~
   
6. **剩下的疑问**

   > 匿名文件的`二进制零`是什么玩意？
   >
   
7. **What to do ?**

   > ========= So, welcome to  `Computer Organization and Design: The Hardware/Software Interface` ============
   >
   > ========= Come and write an operating system of your self. ================================================
   >
   > `RISC(Reduced Instruction Set Computer)`，即精简指令集，目前主流的处理器都采用这类指令集，代表包括ARM、MIPS、RISC-V等
   >
   > `CISV(Complex Instruction Set Computer)`，即复杂指令集，目前主流的仅Intel的部分产品采用此种指令集，代表包括x86-64和IA32（x86架构的32位版本）



参考资料

[从内核文件系统看文件读写过程](https://www.cnblogs.com/huxiao-tee/p/4657851.html)

[认真分析mmap：是什么 为什么 怎么用](https://www.cnblogs.com/huxiao-tee/p/4660352.html)

[知乎：Cache 和 Buffer 都是缓存，主要区别是什么？](https://www.zhihu.com/question/26190832)

[知乎：为什么汇编语言不能越过操作系统操控硬件？](https://www.zhihu.com/question/43575404)

[知乎：关于CPU、指令集、架构、芯片的一些科普](https://zhuanlan.zhihu.com/p/19893066)

[关于RISC-V和开源处理器的一些解读](http://crva.ict.ac.cn/?page_id=540)
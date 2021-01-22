# System Control Command



### 管理进程

```bash
#查看此刻系统上运行的进程（可控参数很多，只需要记住常用的
ps -f #参数-f表示全格式（显示的信息更加详细了
ps --forest #以树状结构显示进程

#动态查看系统进程
top

#kill向PID为xxx的进程发出信号（前提是必须是进程的属主或者是root
kill xxx #默认发出TERM信号
kill -s HUP xxx #使用-s指定其他信号

#killall向进程名为xxx的进程发出信号
killall http* #支持通配符，便于终止多个进程
```



**Linux进程信号**

| 信号 | 名称 | 描述                         |
| ---- | ---- | ---------------------------- |
| 1    | HUP  | 挂起                         |
| 2    | INT  | 中断                         |
| 3    | QUIT | 结束运行                     |
| 9    | KILL | 无条件终止                   |
| 11   | SEGV | 段错误                       |
| 15   | TERM | 尽可能终止                   |
| 17   | STOP | 无条件停止运行，但不终止     |
| 18   | TSTP | 停止或暂停，但继续在后台运行 |
| 19   | CONT | 在STOP或TSTP之后恢复执行     |



### 监测磁盘空间

*目前大部分Linux发行版都能自动挂/卸载特定类型的可移动存储媒体，如不支持自动挂/卸载可移动存储媒体，就必须手动完成*

```bash
#查看当前系统上挂载的设备列表
mount #mount可以有很多参数（具体可以man手册或者搜索引擎

#手动挂载媒体设备
mount -t type device directory #基本命令格式
mount -t vfat /dev/sdb1 /media/disk #手动将文件类型为vfat的U盘/dev/sdb1挂载到/media/disk

#手动卸载媒体设备
umount [directory | device ] #基本命令格式
umount /dev/sdb1 #卸载挂载了的/dev/sdb1（设备名称
umount /media/disk #卸载挂载了的/media/disk（挂载的设备的文件名

#如果在卸载过程中显示还有进程在使用该设备，可以用lsof查看
lsof [directory | device ] #基本命令格式
lsof /luster #查看挂载了的/luster被哪些进程使用

#查看所有已挂载磁盘的使用情况
df -h #使用-h参数输出的文件

#显示某个特定目录的磁盘使用情况（-c显示所有已列出文件总的大小，-s显示每个输出参数的总计。-h按用户易读的格式输出大小
du
```



### 处理数据文件

```bash
#sort排序数据（默认将所有按照字符处理
#-g按照浮点数（支持科学记数法）排列数字；-n按照数值排列；-t指定一个用来区分键位置的字符；--key=POS1[,POS2]（短写为-k）排序从POS1位置开始，如果指定了POS2的话，到POS2位置结束；-r反序排序（升序变成降序）
sort -t ':' -k 3 -n /etc/passwd #对密码文件/etc/passwd根据用户ID进行数值排序，并且使用:来分隔字段
du -sh * | sort -nr #查看当前目录下的磁盘使用情况，并且按照降序输出结果

#grep搜索数据（在输入或指定的文件中查找包含匹配指定模式的字符的行
#-v反向搜索（输出不匹配该模式的行）；-n显示匹配模式的行所在的行号；-c输出有多少行含有匹配的模式；-e指定多个匹配模式
grep [options] pattern [file] #基本命令格式（建议查看man手册，grep非常强大
grep three file1 #在file1中搜索匹配three的文本
grep -e t -e f file1 #在file1中搜索含有字符t或字符f的所有行

#gzip压缩；gzcat查看压缩过的文本文件的内容；gunzip解压文件

#tar归档数据
tar function [options] object1 object2 ...
tar -zxvf filename.tgz #解压.tgz文件
tar -cvf test.tar test/ test2/ #创建了名为test.tar的归档文件，含有test和test2目录内容
tar -tf test.tar #列出test.tar的内容，但是不提取（查看压缩包内容
```



**tar命令的功能**

| 功能 | 长名称        | 描述                                                         |
| ---- | ------------- | ------------------------------------------------------------ |
| -A   | --concatenate | 将一个已有tar归档文件追加到另一个已有tar归档文件             |
| -c   | --create      | 创建一个新的tar归档文件                                      |
| -d   | --diff        | 检查归档文件和文件系统的不同之处                             |
|      | --delete      | 从已有tar归档文件中删除                                      |
| -r   | --append      | 追加文件到已有tar归档文件末尾                                |
| -t   | --list        | 列出已有tar归档文件的内容                                    |
| -u   | --update      | 将比tar归档文件中已有的同名文件新的文件追加到该tar归档文件中 |
| -x   | --extract     | 从已有tar归档文件中提取文件                                  |

**tar命令选项**

| 选项    | 描述                              |
| ------- | --------------------------------- |
| -C dir  | 切换到指定目录                    |
| -f file | 输出结果到文件或设备file          |
| -j      | 将输出重定向给bzip2命令来压缩内容 |
| -p      | 保留所有文件权限                  |
| -v      | 在处理文件时显示文件              |
| -z      | 将输出重定向给gzip命令来压缩内容  |



### ulimit指令

| 参数                  | 作用                            |
| --------------------- | ------------------------------- |
| -c <core文件大小上限> | core dump的文件大小，单位为区块 |
| -f <文件大小>         | 建立的最大文件，单位为区块      |
| -m <内存大小>         | 指定可使用内存的上限，单位为KB  |
| -s <堆叠大小>         | 堆的上限，单位为KB              |
| -v <虚拟内存大小>     | 可使用的虚拟内存上限，单位为KB  |


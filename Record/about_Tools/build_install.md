# 编译安装软件/库

*记录一些常见的软件/库的编译安装*

### openMPI

```bash
#版本4.0.5
#进入openmpi安装包的目录
cd ~/cyJ/work_station/openmpi-4.0.5

#创建编译文件夹
mkdir build
cd build

#编译安装到指定路径
../configure --prefix=~/cyJ/WorkStation/openmpi-4.0.5 #这里会要求写绝对路径
make all install
```



### openBLAS

```bash
#版本0.3.12
#进入openblas安装包的目录
cd openblas-0.3.12

#指定编译路径
make FC=gfortran -j #指定FC（在自己安装的gcc的时候，必须写这个，不然会报错
make -j #不指定FC（使用module load的时候系统帮忙自动设定了

#编译安装
make PREFIX=~/cyJ/WorkStation/openblas-0.3.12 install
```



### GCC

```bash
#版本9.3.0
#到GitHub找到mirror后下载/找一个镜像源下载，然后解压

#进入解压后的文件夹并安装相关的依赖
cd gcc-9.3.0
contrib/download_prerequisites #如果显示依赖无法下载（大概率是墙的问题），就把这个脚本里面的base_url改成一个镜像即可

#编译安装
mkdir build
cd build
../configure --prefix=/lustre/home/acct-hpc/asc/cyJ/WorkStation/gcc-10.2.0 --disable-multilib #需要绝对路径
make -j #加这个参数可以开启并行编译（更快一些
make install
```



### spack

```bash
#下载spack安装包
cd /home/vieran/workstation
git clone https://github.com/spack/spack.git

#设置与spack相关的环境变量，以便可以在命令行中直接使用spack
cd spack
. share/spack/setup-env.sh

#使用spack安装软件后需要激活对应的环境变量才能使用该软件
spack env activate myenv -v --with-view
```


# LLVM入门

## 编译安装

```bash
# 下载、解压
mkdir build
cd build
cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" -DCMAKE_INSTALL_PREFIX=$HOME/apps/llvm-12.0.0 -DCMAKE_BUILD_TYPE=Release ../llvm
cmake --build . -- check-all make -j20
```



## 基础使用

```bash
# 指定gcc的路径
clang++ -g --gcc-toolchain=$HOME/apps/gcc-9.3.0-cas -O3 toy.cpp `llvm-config --cxxflags`
```


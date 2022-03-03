# Customize Your Vim

*选择neovim作为默认编译器（因为它支持的自动补全插件更nice），并对其进行配置*

## 安装neovim

```bash
#直接下载binary包，避免编译报错
wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz

#在环境变量脚本中设置
NVIM=${HOME}/cyJ/WorkStation/nvim-linux64
export XDG_CONFIG_HOME=${NVIM} #指定neovim的初始化脚本
export PATH=${NVIM}/bin:$PATH
alias vim=neovim #用neovim替代vim

#创建neovim默认的配置文件夹和文件
mkdir nvim #在neovim安装文件夹下创建文件夹
cd nvim
vim init.vim #创建配置文件

#颜色配置
#在.bashrc里面加入
export TERM=screen-256color
#在init.nvim加入
if &term == 'screen'
	set t_Co=256
endif
```



## 配置neovim

[VimAwesome](https://vimawesome.com/)

### vim-plug

```bash
#在neovim安装目录下的nvim目录下创建autoload文件夹，将指定的文件下载到此文件夹下（这个得挑时间，有时候dns被污染了就下载不了
mkdir autoload
cd autoload
wget https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

#修改neovim的配置文件
vim init.vim
#添加下列语句，插件相关语句放在start和end中间
call plug#begin()
call plug#end()

#使用方法
:PlugInstall [name ...] [#threads] #安装插件
:PlugUpdate [name ...] [#threads] #升级或者安装插件
:PlugClean[!] #删除没有列出来的插件
:PlugUpgrade #升级vim-plug本身
:PlugStatus #查看插件状态
:PlugDiff #检查升级前后的变化
:PlugSnapshot[!] [output path] #创建脚本来保存当前的插件的snapshot
```



### coc.nvim

```bash
#自动补全插件
#下载nodejs的binary包（方便，不需要安装权限），放到WorkStation目录下，并在环境变量文件中输出其bin的位置
NODEJS=${APP}/node-v14.15.3-linux-x64
export PATH=${NODEJS}/bin:$PATH

#在vim-plug中加入下列语句，然后:PlugInstall
Plug 'neoclide/coc.nvim', {'branch': 'release'}

#安装对c语言家族的支持（在neovim中执行
:CocInstall coc-clangd

#编译安装llvm
#下载并解压，进入目录
mkdir build
cd build
module load gcc/9.3.0-gcc-4.8.5 #要求gcc版本大于5.1.0
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$HOME/cyJ/WorkStation/llvm-11.0.0 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;libunwind;lldb;compiler-rt;lld;polly;deuginfo-tests" --enable-optimized ../llvm
make -j #使用release是为了防止因为占用内存过大而被kill（这里只安装了clang
make install
#在脚本中将llvm输出到到环境变量（看起来仅仅安装clang是不够的，因为插件提示没有language server，故使用系统的llvm了

#if you can sudo, run the following commands
apt-get install nodejs clang clangd npm #be carefull that clang and clangd should be the same version

#clangd经常崩溃，而且无法补全全部函数（比如路径上的mpi函数），故换用ccls
git clone --depth=1 --recursive https://github.com/MaskRay/ccls
cd ccls
cmake -H. -BRelease -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/clang+llvm-xxx -DCMAKE_INSTALL_PREFIX=$HOME/cyJ/WorkStation/ccls -DCMAKE_CXX_COMPILER=g++ #最后这个g++得指定，需要支持c++17的
cmake --build Release --target install #安装到前面设置的路径下，这样统一一点
#最后将ccls所在的路径export到PATH中，按照官方给的文档进行配置

#与airline相关的配置，自行再查看Readme.md进行设置
```



### NERDTree

```bash
#显示文件目录插件
#在init.vim中加入
Plug 'preservim/nerdtree'

#相关配置和使用
:NERDTree #打开nerdtree（然后输入?可以直接查看帮助手册
:help NERDTree #查看帮助手册

#在init.vim中加入下列语句，则可以使用快捷键CTRL+n打开nerdtree
nnoremap <C-n> :NERDTree<CR>
```



### vim-airline

```bash
#状态栏插件
#在init.vim中加入
Plug 'vim-airline/vim-airline'

#这个还可以安装主题插件，但是这里就不安装了；其他使用，可以自行官方参考文档
#在init.vim中加入下列语句开启coc.nvim的支持
let g:airline#extensions#coc#enabled = 1
```



### fzf

```bash
#模糊搜索工具
#在init.vim中加入
Plug 'junegunn/fzf'

#使用时候提示下载则下载，然后会在neovim安装目录的plugged文件夹下找到全部的文件
#fzf是portable的，复制一份到WorkStation里面，在脚本里面export路径，即可在一般命令行下使用
:FZF #用于查找当前目录下的所有文件
:q #退出fzf

#更多使用方法查看其README.md和README-VIM.md
```

[fzf常用方法](https://zhuanlan.zhihu.com/p/41859976)

[vim中fzf常用方法](https://vimjc.com/vim-fzf-plugin.html#3-Vim-fzf%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)



### ack.vim

```bash
#在vim中进行使用ack进行全局搜索（比vim原生的全局搜索快，ack本身是一个软件，这个插件是vim里面的ack前端
#在init.vim中加入
Plug 'mileszs/ack.vim'

#使用方法
:Ack xxx * #全局搜索xxx

#在init.vim中加入下列语句，不自动跳转到第一个搜索结果，使用快捷键搜索
cnoreabbrev Ack Ack!
nnoremap <Leader>a :Ack!<Space> #leader键就是\（在enter上方那个按键
```



### ale

```bash
#语法检查工具
#在init.vim中加入
Plug 'dense-analysis/ale'

#查看使用手册
:help ale

#配合coc.nvim使用
:CocConfig #在vim中执行此命令，然后打开coc的设置文件
#在文件中写入（如果ale和airline的显示没有配合好，那么将无法达到效果，所以这里先设置未false，待设置好了再改回来
{
	"diagnostic.displayByAle": true
}

#在init.vim中加入下列语句
let g:ale_disable_lsp = 1

#如果打开文件的时候显示“xxx.h no such file”，则需要到/usr/include下找一找有没有对应的文件
#如果找到是在文件夹里面，则创建一个软链接即可
ln -s /directory/xxx.h xxx.h

#更多的其他设置自行查看官方文档
```



### tagbar

```bash
#显示和跳转到函数定义的插件
#首先需要安装universal-ctags
#下载安装包/克隆GitHub仓库，并解压缩，进入到解压缩后的目录，指定安装位置并安装
./autogen.sh
./configure --prefix=$HOME/cyJ/WorkStation/universal-ctags CC="gcc -std=c99"
#在这个c99这里坑了很久，仔细阅读readme和docs/autotools.rst，在CFLAGS指定c99并未起作用，受到alias gcc='gcc -std=c99'（这个不起作用）启发就这么写了
make -j
make install

#在当前目录下递归地为所有文件创建tags（会在当前目录下生成一个名为tags的文件
ctags -R

#在init.vim中加入
Plug 'majutsushi/tagbar'
nnoremap <silent><F10> :TagbarToggle<CR> #F10显示定义
nnoremap <silent><F11> <C-]> #使用F11替代CTRL+]跳转到函数定义
nnoremap <silent><F11> <C-o> #使用F12替代CTRL+o返回上一次光标处

#查看使用帮助手册（暂时还没有去学很多这个相关的使用
:help tagbar
```



## Vim的使用

[vim help v8.2](http://vimcdoc.sourceforge.net/doc/help.html)

```bash
#vim的全局搜索：在当前文件夹下查找所有的包含xxx的文件，并且设置可以跳转
:vim /xxx/** | copen

#不退出vim，直接跳转到命令行执行命令
:!export | grep PATH #相当于在命令行执行export | grep PATH

#命令模式下
u #撤销
CTRL+r #恢复撤销
x #删除当前光标所在位置的单个字符
r #按下r，然后输入新字符来替换光标所在处的单个字符（如果是R则相当于Windows下的insert，直到按esc退出
yw #复制一个单词
y$ #复制到行尾
w #以单词为间隔向右跳转
b #以单词为间隔向左跳转
CTRL+v #整块选中
G #跳转到文件最后一行
0 #跳转到本行的开头
$ #跳转到本行的结尾
CTRL+f #向前翻页（forward
CTRL+u #向后翻页（upword
ggVGyy #复制整个文件
yaw #复制一个单词（这个a可以换做i
vawp #选中一个单词，粘贴替换（这个a可以换做i
"xy3y #选中某一个（x）粘贴板，然后复制（这里是三行）
"xp #选中某一个（x）粘贴板，然后粘贴其中内容
:reg #查看所有粘贴板内容
* #光标位于某个单词时按下*，高亮整个文件所有相同单词，相当于在全文搜索该单词
:e #刷新文件

#强大的g命令，全局的
:g

#查找和替换
:s/old/new/ #跳到old第一次出现的地方，并用new替换
:s/old/new/g #替换所有old
:n,ms/old/new/g #替换行号n和m之间所有old
:%s/old/new/g #替换整个文件中的所有old
f<字母> #在当前行向右查找该字母第一次出现的地方，在按;查找当前行下一次出现的地方
F<字母> #在当前行向左查找该字母第一次出现的地方，在按;查找当前行下一次出现的地方

#跳转
CTRL+] #跳转到函数定义处（前提是生成了tags；如果定义了快捷键，可以直接用快捷键跳转，比如F11
CTRL+o #向后跳到后几次光标位置（跳到函数之后，再输入这个就可以跳回原处
CTRL+i #向前跳到前几次光标位置

#分屏
#打开文件的时候使用-On（竖直）或者-on（水平）分屏，n是窗口数量
vim -O file1 file2
CTRL+w whjkl #窗格间跳转，hjkl分别是左上下右，w是轮流切换
:vsplit filename #竖直分屏打开文件，可以简写vsp（水平分屏是split，简写sp
CTRL+w v #竖直分屏打开当前文件（水平分屏是s
CTRL+w =+- #=设置所有屏幕相同高度，+增加当前屏幕高度，-减小当前屏幕高度
CTRL+w c #关闭当前屏幕
map si :set splitright<CR>:vsplit<CR> #快捷键si竖直向右分屏
map sn :set nosplitright<CR>:vsplit<CR> #快捷键sn竖直向左分屏

#查阅函数/变量的作用
:h <函数名/变量名>

#对于某个已编辑但是想要管理员权限才能保存的内容
:w !sudo tee % #本质是把当前内容存进这个文件
:w <不需要管理员权限的目录> #然后再在命令行用sudo替换文件

#切换为粘贴模式（不自动缩进
:set paste
:set nopaste #恢复正常模式（输入时候正常缩进

# vimdiff对比两个文件
vimdiff -d file1 file2 #左右分屏
vimdiff -o file1 file2 #上下分屏
#在file1某行输入dp（diff put），将file1该行推送到file2；在file1某行输入do（diff obtain），将file2该行拉取到file1
```

[vim粘贴板的使用](https://www.cnblogs.com/huahuayu/p/12235242.html)

[vim命令大全](https://zhuanlan.zhihu.com/p/51440836)


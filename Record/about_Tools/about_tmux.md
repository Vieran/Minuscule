# Learning tmux

### 简介

> 终端复用器、挂在systemd下，即使当前的shell关闭，也不会停止作业
>
> 可以分屏操作，每一个窗格相当于一个新的shell
>
> CTRL+b之后的按键是对tmux本身进行操作
>
> [tmux的GitHub仓库](https://github.com/tmux/tmux)、[窗口指令简单整理](https://www.cnblogs.com/lizhang4/p/7325086.html)、[较为详细的使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)、[Tmux使用手册](http://louiszhai.github.io/2017/09/30/tmux/)



### 简单操作

```bash
#使用CTRL+b之后
#[：进入copy mode（按ctrl+s搜索字符，按n再次搜索同样的字符，按shift+n进行反向搜索，按两次esc退出copy mode
#d：分离当前会话
#s：列出所有会话
#c：创建新的窗口（CTRL+b n进行切换
#$：重命名当前会话
#%：左右分屏
#"：上下分屏
#上下左右键：在窗格之间跳转（一直按住CTRL+b+上下左右键，可以调整窗格大小

#创建一个名字为xxx的tmux
tmux new -As xxx #不存在则新建，存在则直接连
tmux new -s xxx

#显示当前存在的tmux会话列表
tmux ls

#连接名称为xxx/编号为x的会话
tmux a -t xxx
tmux a -t x

#关闭当前tmux窗格使用CTRL+b+x
#关闭tmux使用CTRL+b+&
#暂时退出tmux使用CTRL+b+d或者使用下述命令
tmux detach

#设置scrollback行数为x
tmux scrollback x

#杀死名称为xxx/编号为x的会话
tmux kill-session -t xxx
tmux kill-session -t x

#切换到名称为xxx/编号为x的会话
tmux switch -t xxx
tmux switch -t x

#将编号为x的会话改为名称为yyy的会话
tmux rename-session -t x yyy

#设置tmux的模式为Vim模式或者Emacs模式（写在配置文件里面
set-window-option -g mode-keys vi
```



## 配置

```bash
#配置文件为~/.tmux.conf

#tmux无法显示vim的全部配色的时候，需要设置256色
set -g default-terminal "screen-256color" #在配置文件写入
export TERM=screen-256color #在bashrc写入
```

**参考**

[stack exchange: getting 256 colors to work in tmux](https://unix.stackexchange.com/questions/1045/getting-256-colors-to-work-in-tmux)
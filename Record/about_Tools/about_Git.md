# Git基本指令学习

[git的官方网站](https://git-scm.com/)

```bash
#查看仓库简要信息状况
git status -s
git status

#查看提交、工作区之间的差异
git diff
git diff --cached #已经缓存的改动
git diff --stat #显示摘要

#从仓库中移除xxx文件（mv命令类似
git rm xxx
git rm -r * #递归删除当前目录下的所有内容
git rm -rf * #强制删除
git rm --cached xxx #删除xxx，不删除工作区文件（也就是这里的xxx必须不是工作区文件才能被成功删除

#查看日志
git log #所有的commit
git log --oneline #简单查看commit信息
git log --online --graph #查看分支信息
#其他参数：--author参数查看指定用户的记录；--since和--before，--until和--after指定日期查看；--reverse和--topo-order逆向查看；--stat详细显示每次commit中修改的文件的内容；--pretty=xxx改变显示格式为xxx

#查看团队协作相关的指令
git help workflows

#任何时候，可以使用man和help帮助使用命令
```



**分支**

```bash
#查看目前所在的分支以及远程的分支
git branch -a #显示所有分支，当前分支前面带*
git branch #显示本地存在的分支

#切换到远程的newhack分支，并且在本地命名为xxx
#git checkout -b xxx origin/newhack
#切换到xxx分支
git switch xxx

#切换到创建新的分支xxx
git switch -c xxx

#删除本地xxx分支
git branch -d xxx

#将branchname分支和当前分支进行合并
git merge branchname

#关于stash
git stash #备份当前的工作区的内容，让工作区保证和上次提交的内容一致；同时将当前的工作区内容保存到Git栈中
git pull #保持和远程项目同步
git stash pop #从git栈中读取上次保存的内容，恢复工作区的相关内容
git stash list #显示git栈内的所有备份
git stash clear #清空git栈内的所有备份
git reset --hard #放弃本地修改
```



**.gitignore文件**

```bash
#在根目录下生成文件并写入信息
vim .gitignore

#文件中只要写入不提交的内容即可
/Reference #过滤整个Reference文件夹
```

[对于.gitignore的写法](https://www.jianshu.com/p/74bd0ceb6182)



### 重建版本库

```bash
#删除历史记录，重新建立版本库
rm -rf .git
git init
git add .
git commit -m "rebuild this repo"
git remote add origin <github_repo_url>
git push -f -u origin master
```



### 撤销修改

```bash
#撤销工作区某文件的修改
git checkout -- filename

#撤销暂存区某文件的修改
git reset HEAD filename

#工作区：add之前
#贮藏区：把工作区的内容stash
#暂存区：把工作区的内容add但未commit
#本地仓库：把暂存区commit后
```


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

#取消对文件xxx的追踪
git rm --chached xxx #在本地不删除该文件
git rm --f xxx #删除本地文件

#关联远程分支和取消关联
git remote add origin <url>
git remote remove origin

#给仓库添加子模块
git submodule add https://github.com/chaconinc/DbConnector

#查看团队协作相关的指令
git help workflows

#任何时候，可以使用man和help帮助使用命令
```

[Git - 子模块 (git-scm.com)](https://git-scm.com/book/zh/v2/Git-工具-子模块)



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
git stash save "msg" #备份当前的工作区的内容，让工作区保证和上次提交的内容一致；同时将当前的工作区内容保存到Git栈中
git stash show #展示stash内容相对当前的更改
git pull #保持和远程项目同步
git stash pop [stash@{id}] #从git栈中读取上次保存的内容，恢复工作区的相关内容，并且删除stash中的内容
git stash apply [stash@{id}] #从git栈中读取保存的内容，但是不删除stash中的内容
git stash list #显示git栈内的所有备份
git stash clear #清空git栈内的所有备份
git reset --hard #放弃本地修改
```

[git - How do you stash an untracked file? - Stack Overflow](https://stackoverflow.com/questions/835501/how-do-you-stash-an-untracked-file)



**.gitignore文件**

```bash
#在根目录下生成文件并写入信息
vim .gitignore

#文件中只要写入不提交的内容即可
/Reference #过滤整个Reference文件夹
```

[对于.gitignore的写法](https://www.jianshu.com/p/74bd0ceb6182)



## 重建版本库

```bash
#删除历史记录，重新建立版本库
rm -rf .git
git init
git add .
git commit -m "rebuild this repo"
git remote add origin <github_repo_url>
git push -f -u origin local_name:remote_name #远程创建分支remote_name并将本地local_name分支推送到远程remote_name
```



## 撤销修改

```bash
#撤销工作区某文件的修改
git checkout -- filename

#撤销暂存区某文件的修改
git reset HEAD filename

#回退到某一次commit
git reset commit_id #可以加--hard选项
git stash #把当前的修改先存起来（在没有--hard选项的时候这么写是有用的
git stash list #查看暂存区的内容
git push origin HEAD --force #强制把当前的分支推送上去，忽略冲突
git stash [apply|pop] stash@{?} #恢复?的暂存区内容
git stash drop stash@{?} #删除?的暂存区内容

#工作区：add之前
#贮藏区：把工作区的内容stash
#暂存区：把工作区的内容add但未commit
#本地仓库：把暂存区commit后
```

[某博客：git stash 用法](https://www.cnblogs.com/tocy/p/git-stash-reference.html)

[简书：git stash命令的使用](https://www.jianshu.com/p/e9764e61ef90)



## 修改commit信息

```bash
git rebase #更多细节查看man git rebase
git rebase <branch_name> -i #开启交互模式
```

[修改git提交历史](https://www.jianshu.com/p/0f1fbd50b4be)

[【Git】rebase 用法小结 - 简书 (jianshu.com)](https://www.jianshu.com/p/4a8f4af4e803)



## 合并分支

```bash
#将A分支合并到B分支（fast forward
git switch B
git merge A

#合并A分支上的文件a到B分支
git switch B
git checkout A a

#合并分支过程中出现冲突
git status #查看冲突的文件（假设是文件xxx出现冲突
vim xxx #方法一：手动解决冲突
git  mergetool #方法二：使用merge工具解决冲突

#在A分支上合并B分支的commit
git cherry-pick [commitID]
```

[Git 分支 - 分支的新建与合并](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6)

[What's the difference between git switch and git checkout \<branch\>](https://stackoverflow.com/questions/57265785/whats-the-difference-between-git-switch-and-git-checkout-branch)

[一个可以提高开发效率的命令：cherry-pick - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/58962086)
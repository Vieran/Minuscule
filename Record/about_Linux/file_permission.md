# Linux File Permission

*理解Linux的文件权限*

## Linux用户

**相关知识**

> UID：每个用户有一个固定的UID，root的UID是0
>
> 系统账户：系统上运行的服务进程访问资源用的特殊账户（Linux系统为不同的功能创建不同账户，为系统账户预留了500以下的UID

```bash
#查看用户相关信息的文件
cat /etc/passwd #由于安全隐患，绝大多数Linux系统都将用户密码保存在单独的/etc/shadow文件中
#即使可以直接修改/etc/passwd文件，也请尽量不要这么做，否则文件损坏了会导致用户无法登录（甚至root也无法登录

#查看用户密码相关信息（只有root才有这个权限
cat /etc/shadow
#文件中的9个字段分别表示：与/etc/passwd文件中的登录名字段对应的登录名、加密后的密码、自上次修改密码后过去的天数密码（自1970年1月1日开始计算）、多少天后才能更改密、多少天后必须更改密码、密码过期前提前多少天提醒用户更改密码、密码过期后多少天禁用用户账户、用户账户被禁用的日期（用自1970年1月1日到当天的天数表示）、预留字段给将来使用（这些内容不一定全部要写完
```



**添加用户**

```bash
#useradd命令使用系统的默认值以及命令行参数来设置用户账户（系统默认值设置在/etc/default/useradd
/etc/default/useradd -D #查看系统相关的默认值
#输出如下
GROUP=100 #新用户被添加到UID为100的公共组
HOME=/home #新用户的HOME目录将会位于/home/loginname（默认情况下不会创建这个目录
INACTIVE=-1 #新用户账户密码在过期后不会被禁用
EXPIRE= #新用户账户未被设置过期日期
SHELL=/bin/bash #新用户使用bash作为默认shell
SKEL=/etc/skel #系统会将/etc/skel目录下的内容复制到用户的HOME目录下
CREATE_MAIL_SPOOL=yes #系统为该用户账户在mail目录下创建一个用于接收邮件的文件

useradd -m xxx #创建用户xxx，并建立其HOME目录（-m参数）
useradd -D -s /bin/tsch #修改默认参数值（这里是将新用户默认的shell改为tsch
#useradd还有很多其他参数，可以通过man手册查看
```

**删除用户**

```bash
#删除用户xxx
/usr/sbin/userdel -r xxx #加上-r参数会删除用户的HOME目录以及邮件目录（-r参数删除需谨慎
```

**修改用户**

```bash
#修改用户账户的字段（-l修改登录名，-L锁定账户使之无法登陆，-p修改账户密码，-U解锁账户使之可以登陆
usermod -L xxx #锁定账户xxx

#修改账户密码（root可以修改别人的，一般用户只能修改自己的，-e参数可以强制用户下次登录时修改密码
passwd xxx #修改xxx的密码，接下来会提示输入用户旧密码和新密码

#为系统大量用户修改密码可以使用chpasswd，自行查看man手册了解用法
chpasswd < users.txt #将含有userid:passwd对的文件重定向给chpasswd命令

#修改默认用户登陆shell
chsh -s /bin/zsh xxx #修改xxx的默认登陆shell为zsh
chfn #将finger命令的信息村粗和进/etc/passwd文件中的备注字段（危险，很多Linux系统没有这个指令，不介绍了，自行看man手册
change #管理用户账户的有效期
```



## Linux组

**相关知识**

> GID：每个组拥有一个固定的UID（系统账户用的组通常会分配低于500的UID

```bash
#查看用户相关信息
cat /etc/group
#文件中的4个字段分别为：组名；组密码（组密码允许非组内成员通过它临时成为该组成员）；GID；属于该组的用户列表
#注意：当一个用户在/etc/passwd文件中指定某个组作为默认组时，用户账户不会作为该组成员再出现在/etc/group文件中

#创建新的组（在创建新组时，默认没有用户被分配到该组
/usr/sbin/groupadd yyy #创建yyy组
/usr/sbin/usermod -G yyy xxx #添加用户xxx到用户组yyy（-G把这个新组添加到该用户账户的组列表里

#修改组
/usr/sbin/groupmod -n zzz yyy #将yyy组名改为zzz（-n改组名，-g改组UID
```



## 文献权限

```bash
#查看文件权限
ls -l #或者使用ll（ls -l缩写
#输出的第一个字段表示文件类型：-文件，d目录，l链接，c字符型设备，b块设备，n网络设备
#后面有3组三字符的编码，每组定义了3种权限：r可读，w可写，x可执行，-无此权限；依次代表对象的属主、对象的属组、系统其他用户的权限

#查看文件默认权限（/etc/profile或者/etc/login.defs文件中指定了umask的值
umask #查看默认创建文件的权限，是掩码，屏蔽掉不想授予该安全级别的权限（第一位代表沾着位，后面的是八进制模式权限
#八进制之模式权限：0表示无此权限，1表示有此权限，得到3位二进制数字，转换为10进制就是0~7
#原来的权限-umask后三位=真实的权限，比如666-022=644

#改变文件权限
chmod [options] [mode] [file] #命令格式
chmod 760 xxx #改变文件xxx的权限为760
chmod o+r xxx #给其他用户添加文件xxx的读权限
chmod o+u xxx #给其他用户添加和属主一样的对于文件xxx的权限
#u属主，g属组，o其他，a以上所有

#改变属主（其他用法自行man
chown u1 xxx #将文件xxx的属主改为u1
chown g1.u1 xxx #将文件xxx的属主改为g1组的u1
#chgrp命令可以更改文件或目录的默认属组

#注意：只有文件的属主才能改变文件或目录的权限
```



## 共享文件

**相关知识**

> Linux为每个文件和目录存储了3个额外的信息位
>
> 设置用户ID（SUID）：当文件被用户使用时，程序会以文件属主的权限运行
>
> 设置组ID（SGID）：对文件，程序会以文件属组的权限运行；对目录，目录中创建的新文件会以目录的默认属组作为默认属组
>
> 粘着位：进程结束后文件还驻留（粘着）在内存中
>
> *这里同样使用了八进制权限模式，或者使用符号*

```bash
#创建一个属组内所有用户共享的文件夹（这里的属组是shared组
mkdir testdir
chgrp shared testdir #将testdir的属组改为shared
chmod g+s testdir #将testdir文件夹下的所有文件都用父目录（testdir）的属组（shared组）作为属组，而不是创建文件用户的属组（这个命令就是设置SUID为1
umask 002 #设置文件对属组是可写的
```


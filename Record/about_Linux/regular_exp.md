# Regular Expression

*Linux下的正则表达式*

## 类型

> 正则表达式是通过正则表达式引擎（regular expression engine）实现的
>
> **Linux中两种流行的正则表达式引擎**
>
> 1. BRE（basic regular expression），POSIX基础正则表达式
> 2. ERE（extended regular expression），POSIX扩展正则表达式
>
> *大多数Linux工具至少都符合POSIX BRE引擎规范，但是有些工具只符合其子集（比如sed，这是出于速度考虑的）*



## BRE

```bash
#规则：区分大小写
#特殊字符需要转义（前加\）：.*[]^${}\/+?|()
echo "\ is a special character" | sed -n '/\\/p' #输出：\ is a special character

#^行首，$行尾
echo "Books are great" | sed -n '/^Book/p' #输出：Books are great
echo "This ^ is a test" | sed -n '/s ^/p' #输出：This ^ is a test
echo "^ is a test" | sed -n '/\^ i/p' #输出：^ is a test
echo "This is a good book" | sed -n '/books$/p' #无输出
echo "This is a good book" | sed -n '/book$/p' #输出：This is a good book
sed '/^$/d' data.txt #过滤掉文件data.txt中的空行

#.匹配除换行符之外的任意字符（没有字符就匹配不到
sed -n '/.at/p' data.txt

#使用[]定义字符组
echo "Yes yes yEs" | sed -n '/[Yy][Ee]s/p' #输出：Yes yes yEs
sed -n '/[0123456789][0123456789]/p' data.txt #输出文件中存在>=10的数字的行
sed -n '/^[0123456789][0123456789]$/p' data.txt #输出文件中数字为[10,99]的行（整行只是数字

#[^]排除特定字符组
echo "cat" | sed -n '/[^ch]at/p' #无输出

#定义区间
sed -n '/[0-9][0-9]/p' data.txt #输出文件中存在>=10的数字的行
echo "bat" | sed -n '/[a-ch-m]at/p' #输出：bat（这个区间表示a-c并h-m

#特殊组的使用
echo "This is a test" | sed -n '/[[:punct:]]/p' #无输出
echo "This is a test." | sed -n '/[[:punct:]]/p' #输出：This is a test.

#*前面的字符出现0次或者多次
echo "I'm getting a color TV" | sed -n '/colou*r/p' #输出：I'm getting a color TV（u出现次数为0
#.*匹配任意数量的任意字符（换行符除外
echo "a regular pattern expression" | sed -n '/r.*o/p' #输出：a regular pattern expression
#[]*指定可能在文本中出现多次的字符组或字符区间
echo "baaaeeet" | sed -n '/b[ae]*t/p' #输出：baaaeeet
echo "bt" | sed -n '/b[ae]*t/p' #输出：bt
```

| 特殊组      | 描述                                               |
| ----------- | -------------------------------------------------- |
| [[:alpha:]] | 匹配任意字母字符                                   |
| [[:alnum:]] | 匹配任意字母和数字                                 |
| [[:blank:]] | 匹配空格或制表符                                   |
| [[:digit:]] | 匹配0~9之间的数字                                  |
| [[:lower:]] | 匹配小写字母字符a~z                                |
| [[:print:]] | 匹配任意可打印字符                                 |
| [[:punct:]] | 匹配标点符号                                       |
| [[:space:]] | 匹配任意空白字符：<br>空格、制表符、NL、FF、VT和CR |
| [[:upper:]] | 匹配大写字母字符A~Z                                |



## ERE

```bash
#?前的字符出现0次或1次
echo "bt" | gawk '/b[ae]?t/{print $0}' #输出：bt
echo "baet" | gawk '/b[ae]?t/{print $0}' #无输出

#+前的字符可以出现1次或多次，但必须至少出现1次
echo "bt" | gawk '/b[ae]+t/{print $0}' #无输出
echo "baaet" | gawk '/b[ae]+t/{print $0}' #输出：baaet

#{}指定字符出现次数（上限或者间隔
#gawk程序不会识别正则表达式间隔；必须指定gawk程序的--re- interval 命令行选项才能识别正则表达式间隔
echo "bt" | gawk --re-interval '/be{1}t/{print $0}' #无输出
echo "bet" | gawk --re-interval '/be{0,3}t/{print $0}' #输出：bet

```


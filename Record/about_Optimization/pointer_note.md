# C语言指针

*遇到指针和数组总是容易犯错，此文件记录指针及其相关的知识点*

### 指针和数组

> 指针：一个变量，其值为另一个变量的地址
>
> 数组：数组名是一个指针，指向数组的第一个元素，但是数组名不可以重新赋值



### 取地址和解引用

> 取地址：&，返回对象本身的地址
>
> 解引用：*，返回指针所指向的对象（左值）

```c
#include <stdlib.h>
int main() {
    int *p, num = 1;
    p = &num; //把变量num的地址赋值给指针；指针p指向了变量num
    *p = 3; //将指针p解引用，然后赋值为3（向p指向的那一块空间写入数字3）；此时，num的值就变为了3
    int *q = &num; //这样的赋值方法也可以把指针q指向变量num
    return 0;
}
```



### 二维数组

*指针最容易出问题的地方就是二维数组了*

```c
#include <stdlib.h>
#include <stdio.h>

#define row 10
#define column 10

//打印空间连续的二维数组（其实是将它当作一维数组处理了
void dis_matrix(int *a, int r, int c) {
    printf("matrix %p", a); //打印指针的地址
    for (int i = 0; i < r; i++) {
        printf("\n");
        for (int k = 0; k < c; k++)
            printf("matrix[%d][%d] = %d ", i, k, *(a + i * r + k));
    }
}
//*(*(a + i) + k)，先解引用a到一维数组，然后偏移指针，得到对应的行，再解引用得到对应的数值
//但是这种情况至少需要a[][定值]，因为这样编译器才知道需要偏移多少

int main() {
    //申请静态的二维数组，空间连续，上述打印函数可用
    int a[row][column];
    printf("matrix a = %p", a); //打印指针的地址
    dis_matrix(a, row, column); //函数打印出来的地址和上面的一致
    
    //申请动态二维数组，空间连续，上述打印函数可用
    int (*b)[colunm] = (int (*)[column])malloc(row * column * sizeof(int));
    printf("matrix b = %p", a); //打印指针的地址
    dis_matrix(b, row, column); //函数打印出来的地址和上面的一致
    free(b);
    
    //申请动态二维数组，空间不一定连续，上述打印函数不可用
    int **c = (int**)malloc(row * sizeof(int*)); //此时c是一个指针数组
    for (int i = 0; i < row; i++)
        c[i] = (int*)malloc(column * sizeof(int)); //为c中的每一个元素申请空间
    printf("matrix c = %p", a); //打印指针的地址
    dis_matrix(c, row, column); //函数中打印出来的地址和上面的不一致
    //逐步释放指针
    for (int i = 0; i < row; i++)
        free(c[i]);
    free(c);

    return 0;
}
```


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 10

//input: matrix A, right end b, initial value x0
//solve: Ax - b = 0   ===>   AX = b
//output: solution of hte equation x

//multi a with x, and return ax
void multi_a_x(double a[N][N], double x[N], double ax[N]) {
    for (int i = 0; i < N; i++) {
        ax[i] = 0;
        for (int k = 0; k < N; k++)
            ax[i] += a[i][k] * x[k];
    }
}

//return the result of b-ax
void b_minus_ax(double b[N], double ax[N], double result[N]) {
    for (int i = 0; i < N; i++)
        result[i] = b[i] - ax[i];
}

//assert if r == 0, 0 for r != 0,1 for r = 0
int assert_req0(double r[N]) {
    int flag = 1;
    for (int i = 0; i < N; i++)
        if (r[i] == 1) {
            flag = 0;
            return flag;
        }
    return flag;
}

//return r * r(T)
double multi_self_trans(double r[N]) {
    double result = 0;
    for (int i = 0; i < N; i++)
        result += r[i] * r[i];
    return result;
}

//return the result of p(T)*a*p
double multi_pT_a_p(double p[N], double ax[N][N]) {
    int result = 0, pTa = 0;
    for (int i = 0; i < N; i++) {
        pTa = 0;
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++)
                pTa += p[k] * ax[k][j];
        }
        result += pTa * p[i];
    }
    return result;
}

//return x + a*p, store in y
void add_x_ap(double x[N], double a, double p[N], double y[N]) {
    for (int i = 0; i < N; i++)
        y[i] += x[i] + a * p[i];
}

//return r - a*A*p, store in r
void minus_r_aAp(double r[N], double a, double A[N][N], double p[N]) {
    double tmp;
    for (int i = 0; i < N; i++) {
        tmp = 0;
        for (int k = 0; k < N; k++)
            tmp += A[i][k] * p[k];
        r[i] -= a * tmp;
    }
}

int main(int argc, char** argv) {
    //matrix A must be a symmetric positive definite matrix
    //how to initialize the matrix a ?
	double a[N][N], b[N], x[N], ax[N];


    //compute A*x and store it in the ax
    multi_a_x(a, x, ax);
    
    //comput b-ax, store it in array r
	double rk[N];
	double pk[N];
    b_minus_ax(b, ax, rk);
    memcpy(pk, rk, N * sizeof(double));

    double ak, bk, tmp; //tmp for rk-1
    for (int i = 0; i < N; i++)
    {
        if (!assert_req0(rk)) {
            tmp = multi_self_trans(rk);
            ak = tmp / multi_pT_a_p(pk, a);
            add_x_ap(x, ak, pk, x);
            minus_r_aAp(rk, ak, a, pk); 
            bk = multi_self_trans(rk) / tmp;
            add_x_ap(rk, bk, pk, pk);
        } else {
            printf("solved!\n");
            break;
        }
    }
    
    return 0;
}

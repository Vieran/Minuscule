#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//input: matrix A, right end b, initial value x0
//solve: Ax - b = 0   ===>   AX = b
//output: solution of hte equation x

//multi a with x, and return ax
void multi_a_x(double** a, double* x, double* ax, int n) {
    for (int i = 0; i < n; i++) {
        ax[i] = 0;
        for (int k = 0; k < n; k++)
            ax[i] += a[i][k] * x[k];
    }
}

//return the result of b-ax
void b_minus_ax(double* b, double* ax, double* result, int n) {
    for (int i = 0; i < n; i++)
        result[i] = b[i] - ax[i];
}

//assert if r == 0, 0 for r != 0,1 for r = 0
int assert_req0(double* r, int n) {
    int flag = 1;
    for (int i = 0; i < n; i++)
        if (r[i] == 1) {
            flag = 0;
            return flag;
        }
    return flag;
}

//return r * r(T)
double multi_self_trans(double* r, int n) {
    double result = 0;
    for (int i = 0; i < n; i++)
        result += r[i] * r[i];
    return result;
}

//return the result of p(T)*a*p
double multi_pT_a_p(double* p, double** ax, int n) {
    int result = 0, pTa = 0;
    for (int i = 0; i < n; i++) {
        pTa = 0;
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++)
                pTa += p[k] * ax[k][j];
        }
        result += pTa * p[i];
    }
    return result;
}

//return x + a*p, store in y
void add_x_ap(double* x, double a, double* p, double* y, int n) {
    for (int i = 0; i < n; i++)
        y[i] += x[i] + a * p[i];
}

//return r - a*A*p, store in r
void minus_r_aAp(double* r, double a, double** A, double* p, int n) {
    double tmp;
    for (int i = 0; i < n; i++) {
        tmp = 0;
        for (int k = 0; k < n; k++)
            tmp += A[i][k] * p[k];
        r[i] -= a * tmp;
    }
}

int main(int argc, char*** argv) {
    //matrix A must be a symmetric positive definite matrix
    int n = aoti(argv[1]);
    double (*a)[n] = (double (*)[n])malloc(n * n * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));
    double *x = (double*)malloc(n * sizeof(double));
    double *ax = (double*)malloc(n * sizeof(double));

    //how to initialize the matrix a ?

    //compute A*x and store it in the ax
    multi_a_x(a, x, ax, n);
    
    //comput b-ax, store it in array r
    double *rk = (double*)malloc(n * sizeof(double));
    double *pk = (double*)malloc(n * sizeof(double));
    b_minus_ax(b, ax, rk, n);
    memcpy(pk, rk, n * sizeof(double));

    double ak, bk, tmp; //tmp for rk-1
    for (int i = 0; i < n; i++)
    {
        if (!assert_req0(rk, n)) {
            tmp = multi_self_trans(rk, n);
            ak = tmp / multi_pT_a_p(pk, ax, n);
            add_x_ap(x, ak, pk, x, n);
            minus_r_aAp(rk, ak, a, pk, n); 
            bk = multi_self_trans(rk, n) / tmp;
            add_x_ap(rk, bk, pk, pk,n);
        } else {
            printf("solved!\n");
            break;
        }
    }
    
    free(a);
    free(b);
    free(x);
    free(ax);
    free(rk);
    free(pk);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define N 5
#define DEVIATION 1E-9

//input: matrix A, right end b, initial value x0
//solve: Ax - b = 0   ===>   AX = b
//output: solution of hte equation x

//multi a with x, and store the result in ax
void multi_a_x(double a[N][N], double x[N], double ax[N]) {
    for (int i = 0; i < N; i++) {
        ax[i] = 0;
        for (int k = 0; k < N; k++)
            ax[i] += a[i][k] * x[k];
    }
}

//compute b-ax, and store the result in result
void b_minus_ax(double b[N], double ax[N], double result[N]) {
    for (int i = 0; i < N; i++)
        result[i] = b[i] - ax[i];
}

//assert if r is sufficiently small
int assert_req0(double r[N]) {
    for (int i = 0; i < N; i++)
        if (fabs(r[i]) > DEVIATION)
            return 0;
    return 1;
}

//return r(T) * r
double multi_self_trans(double r[N]) {
    double result = 0;
    for (int i = 0; i < N; i++)
        result += r[i] * r[i];
    return result;
}

//return the result of p(T)*a*p
double multi_pT_a_p(double p[N], double a[N][N]) {
    int result = 0, pTa;
    for (int i = 0; i < N; i++) {
        pTa = 0;
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++)
                pTa += p[k] * a[k][j];
        }
        result += pTa * p[i];
    }
    return result;
}

//return x + ak*pk, store in y
void add_x_ap(double x[N], double ak, double pk[N], double y[N]) {
    for (int i = 0; i < N; i++)
        y[i] += x[i] + ak * pk[i];
}

//return rk-ak*A*pk, store in rk
void minus_r_aAp(double rk[N], double ak, double A[N][N], double pk[N]) {
    double tmp;
    for (int i = 0; i < N; i++) {
        tmp = 0;
        for (int k = 0; k < N; k++)
            tmp += A[i][k] * pk[k];
        rk[i] -= ak * tmp;
    }
}

int main(int argc, char** argv) {
    //matrix A must be a symmetric positive definite matrix
    //how to initialize the matrix a ?
	double a[N][N], b[N], x[N], ax[N];

	//initialize x
	for (int i = 0; i < N; i++)
		x[i] = 1;
	for (int i = 0; i < N; i++) {
		b[i] = 1;
		ax[i] = 0;
		for (int k = 0; k < N; k++)
			a[i][k] = 0;
	}
	a[0][0] = 3, a[0][1] = 1, a[1][0] = 2, a[1][1] = 3, a[1][2] = 1;
    a[2][1] = 2, a[2][2] = 3, a[2][3] = 1;
    a[3][2] = 2, a[3][3] = 3, a[3][4] = 1;
    a[4][3] = 2, a[4][4] = 3;

    //compute A*x and store it in the ax
    multi_a_x(a, x, ax);
    
    //comput b-ax, store it in array r
	double rk[N];
	double pk[N];
    b_minus_ax(b, ax, rk);
    memcpy(pk, rk, N * sizeof(double));

    double ak, bk, tmp; //tmp for rk-1
	while(1) {
            tmp = multi_self_trans(rk);
            ak = tmp / multi_pT_a_p(pk, a);
            add_x_ap(x, ak, pk, x);
            minus_r_aAp(rk, ak, a, pk); 
			if (assert_req0(rk))
				break;
            bk = multi_self_trans(rk) / tmp;
            add_x_ap(rk, bk, pk, pk);
    }

	printf("solved!\n");
	for (int i = 0; i < N; i++)
		printf("%lf\n", x[i]);
    
    return 0;
}

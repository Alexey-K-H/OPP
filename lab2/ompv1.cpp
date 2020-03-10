#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <omp.h>

void Matrix_by_vector(int N, double **M, const double *V, double *R)
{
    int j;
    #pragma omp parallel for private(j)
    for(int i = 0; i < N; i++){
        R[i] = 0;
        for(j = 0; j < N; j++)
        {
            R[i] += M[i][j]*V[j];
        }
    }
}

void Minimal_Nevazki(int N, double **A, const double *b, double *X, double eps)
{
    double *R = (double*)malloc(sizeof(double)*N);
    double *Y = (double*)malloc(sizeof(double)*N);
    double *Xn = (double*)malloc(sizeof(double)* N);

    double crit_module;
    double chisl_Tau;
    double del_Tau;

    for(int i = 0; i < N; i++){
        Xn[i] = 0;
    }

    do{
        Matrix_by_vector(N, A, Xn, R);

	#pragma omp parallel for
        for(int i = 0; i < N; i++){
            Y[i] = R[i] - b[i];
        }

        Matrix_by_vector(N, A, Y, R);

        chisl_Tau = 0.0;
        del_Tau = 0.0;

	#pragma omp parallel for reduction(+:chisl_Tau, del_Tau)
        for(int i = 0; i < N; i++)
        {
            chisl_Tau += R[i]*Y[i];
            del_Tau += R[i]*R[i];
        }
        chisl_Tau = chisl_Tau/del_Tau;

	#pragma omp parallel for
        for(int i = 0; i < N; i++){
            X[i] = Xn[i] - chisl_Tau*Y[i];
        }

        Matrix_by_vector(N, A, X, R);

        double crit_1 = 0.0;
        double crit_2 = 0.0;

	#pragma omp parallel for reduction(+:crit_1, crit_2)
        for(int i = 0; i < N; i++){
            crit_1 += pow(R[i] - b[i], 2);
            crit_2 += pow(b[i], 2);
        }
        crit_1 = sqrt(crit_1);
        crit_2 = sqrt(crit_2);
        crit_module = crit_1/crit_2;

	#pragma omp parallel for
        for(int i = 0; i < N; i++){
            Xn[i] = X[i];
        }
    }
    while (crit_module >= eps);

    free(R);
    free(Y);
    free(Xn);
}

int main(int argc, char **argv) {
    int N = 8000;
    double start, end;

    double **A;
    A = (double**)malloc(N *sizeof(double*));
    for(int i = 0; i < N; ++i){
        A[i] = (double*)malloc(sizeof(double) * N);
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(i == j){
                A[i][j] = 2.0;
            } else{
                A[i][j] = 1.0;
            }
        }
    }

    double *u = (double*)malloc(sizeof(double)*N);
    for(int i = 0; i < N; i++){
        u[i] = sin(2*M_1_PI*i);
    }

    double *b = (double*)malloc(sizeof(double) * N);
    Matrix_by_vector(N, A, u, b);

    double *X = (double*) malloc(sizeof(double) * N);
    double epsilon = pow(10, -5);

    start = omp_get_wtime();
    Minimal_Nevazki(N, A, b, X, epsilon);
    end = omp_get_wtime();

    printf("Time: %lf sec. \n", end - start);

    /*for(int i = 0; i < N; i++){
        printf("X[%d] = %lf   u[%d] = %lf\n",i, X[i], i, u[i]);
    }*/

    for (int i = 0; i < N; ++i)
        free(A[i]);
    free(A);

    free(u);
    free(b);
    free(X);

    return 0;
}


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>

void Matrix_by_vector(int N, double **M, const double *V, double *R)
//N - размерность, M - матрица, V - вектор, R - реузультат
{
    for(int i = 0; i < N; i++){
        R[i] = 0;
        for(int j = 0; j < N; j++)
        {
            R[i] += M[i][j]*V[j];
        }
    }
}

int Minimal_Nevazki(int N, double **A, const double *b, double *X, double eps)
//N - размерность, A - матрица, b - вектор свободных членов, X - вектор результат, eps - точность
{
    int count = 0;//Число итераций
    double *R = (double*)malloc(sizeof(double)*N);
    double *Y = (double*)malloc(sizeof(double)*N);
    double *Xn = (double*)malloc(sizeof(double)* N);//приближение

    double crit_module;
    double chisl_Tau;
    double del_Tau;

    for(int i = 0; i < N; i++){
        Xn[i] = 0;//Начальное приближение делаем равным 0
    }

    do{
        Matrix_by_vector(N, A, Xn, R);
        for(int i = 0; i < N; i++){
            Y[i] = R[i] - b[i];
        }

        Matrix_by_vector(N, A, Y, R);

        chisl_Tau = 0.0;
        del_Tau = 0.0;
        for(int i = 0; i < N; i++)
        {
            chisl_Tau += R[i]*Y[i];
            del_Tau += R[i]*R[i];
        }
        chisl_Tau = chisl_Tau/del_Tau;
        for(int i = 0; i < N; i++){
            X[i] = Xn[i] - chisl_Tau*Y[i];
        }


        Matrix_by_vector(N, A, X, R);
        double crit_1 = 0.0;
        double crit_2 = 0.0;
        for(int i = 0; i < N; i++){
            crit_1 += pow(R[i] - b[i], 2);
            crit_2 += pow(b[i], 2);
        }
        crit_1 = sqrt(crit_1);
        crit_2 = sqrt(crit_2);
        crit_module = crit_1/crit_2;

        //Обновляем приближение
        for(int i = 0; i < N; i++){
            Xn[i] = X[i];
        }

        count++;
    }
    while (crit_module >= eps);

    free(R);
    free(Y);
    free(Xn);
    return count;
}

int main(int argc, char **argv) {
    int N = 26500;
    printf("Curr N:%d\n", N);
    struct timespec start, end;

    double **A;
    A = (double**)malloc(N *sizeof(double*));
    for(int i = 0; i < N; ++i){
        A[i] = (double*)malloc(sizeof(double) * N);
    }


    //Заполнение матрицы А
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
        u[i] = sin((2*M_1_PI*i)/N);
    }

    double *b = (double*)malloc(sizeof(double) * N);
    Matrix_by_vector(N, A, u, b);

    double *X = (double*) malloc(sizeof(double) * N);//Вектор решений
    double epsilon = pow(10, -5);//Точность

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int count_steps = Minimal_Nevazki(N, A, b, X, epsilon);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);


    printf("Count steps:%d\n", count_steps);
    printf("Time: %lf sec \n", end.tv_sec - start.tv_sec + 0.000000001*(end.tv_nsec - start.tv_nsec));



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

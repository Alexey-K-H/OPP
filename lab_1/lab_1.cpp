#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

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
    double *R = new double[N];
    double *Y = new double[N];
    double *Xn = new double[N];//приближение

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

    delete[](R);
    delete[](Y);
    delete[](Xn);
    return count;
}

int main(int argc, char **argv) {

    struct timespec start, end;

    int N = atoi(argv[1]);
    std::cout << "N value:" << N << std::endl;
    double **A;
    A = new double *[N];
    for(int i = 0; i < N; ++i)
    {
        A[i] = new double [N];
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

    double u[N];
    for(int i = 0; i < N; i++){
        u[i] = sin((2*M_1_PI*i)/N);
    }

    double b[N];
    Matrix_by_vector(N, A, u, b);

    double X[N];//Вектор решений
    double epsilon = pow(10, -5);//Точность

    std::cout << "Curr epsilon:" << epsilon << std::endl;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int count_steps = Minimal_Nevazki(N, A, b, X, epsilon);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    std::cout << "Count steps:" << count_steps << std::endl;

    printf("Time taken: %lf sec.\n", end.tv_sec - start.tv_sec + 0.000000001*(end.tv_nsec - start.tv_nsec));
    for(int i = 0; i < N; i++){
        std::cout << "X[" << i << "] = " << X[i] << "   u[" << i << "] = " << u[i] << std::endl;
    }

    for (int i = 0; i < N; ++i)
        delete [] A[i];
    delete [] A;

    return 0;
}

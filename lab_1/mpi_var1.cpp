#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int chunk_size(int rank, int size, int N){
    int basic = N/size;
    int rest = N % size;
    return basic + (rank < rest ? 1 : 0);
}

int get_offset(int rank, int size, int N){
    int offset = 0;
    for(int i = 0; i < rank; i++){
        offset += chunk_size(rank, size, N);
    }
    return offset;
}

void Matrix_by_vector(int N, double **M, const double *V, double *R, int rank, int size)
//N - размерность, M - матрица, V - вектор, R - реузультат
{
    int row_count = chunk_size(rank, size, N);
    for(int i = 0; i < row_count; i++){
        R[i] = 0;
        for(int j = 0; j < N; j++)
        {
            R[i] += M[i][j]*V[j];
        }
    }
}

int Minimal_Nevazki(int N, double **A, const double *b, double *X, double eps, int rank, int size, int* sizes, int* offsets)
//N - размерность, A - матрица, b - вектор свободных членов, X - вектор результат, eps - точность
{
    int count = 0;//Число итераций
    int row_count = chunk_size(rank, size, N);
    double *R_part = (double*)malloc(sizeof(double)*row_count);
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
        //Ax
        Matrix_by_vector(N, A, Xn, R_part, rank, size);
        if(size > 1){
        printf("Size > 1\n");
        MPI_Allgatherv(R_part, sizes[rank], MPI_DOUBLE, R, sizes, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        if(size > 1){
            for(int i = 0; i < N; i++){
                Y[i] = R[i] - b[i];
            }
        }else{
            for(int i = 0; i < N; i++){
                Y[i] = R_part[i] - b[i];
            }
        }

        Matrix_by_vector(N, A, Y, R_part, rank, size);
        if(size > 1){
        MPI_Allgatherv(R_part, sizes[rank], MPI_DOUBLE, R, sizes, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        chisl_Tau = 0.0;
        del_Tau = 0.0;

        if(size > 1){
            for(int i = 0; i < N; i++)
            {
                chisl_Tau += R[i]*Y[i];
                del_Tau += R[i]*R[i];
            }
        }else{
            for(int i = 0; i < N; i++)
            {
                chisl_Tau += R_part[i]*Y[i];
                del_Tau += R_part[i]*R_part[i];
            }
        }


        chisl_Tau = chisl_Tau/del_Tau;
        for(int i = 0; i < N; i++){
            X[i] = Xn[i] - chisl_Tau*Y[i];
        }

        Matrix_by_vector(N, A, X, R_part, rank, size);
        if(size > 1){
        MPI_Allgatherv(R_part, sizes[rank], MPI_DOUBLE, R, sizes, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        double crit_1 = 0.0;
        double crit_2 = 0.0;

        if(size > 1){
            for(int i = 0; i < N; i++){
                crit_1 += pow(R[i] - b[i], 2);
                crit_2 += pow(b[i], 2);
            }
        }else{
            for(int i = 0; i < N; i++){
                crit_1 += pow(R_part[i] - b[i], 2);
                crit_2 += pow(b[i], 2);
            }
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
    free(R_part);
    return count;
}

int main(int argc, char **argv) {
    int N = 26500;
    printf("Curr N:%d\n", N);

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Size:%d\n", size);

    double **A;
    int rows_count = chunk_size(rank, size, N);
    int offset = get_offset(rank, size, N);
    A = (double**)malloc(rows_count * sizeof(double*));
    for(int i = 0; i < rows_count; ++i){
        A[i] = (double*)malloc(sizeof(double)*N);
    }


    //Заполнение матрицы А
    for(int i = 0; i < rows_count; i++){
        for(int j = 0; j < N; j++){
            if(i + offset == j){
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

    double *b_part = (double*)malloc(sizeof(double) * rows_count);
    double *b = (double*)malloc(sizeof(double)*N);

    int* sizes = (int*)malloc(sizeof(int) * size);
    for(int i = 0; i < size; i++){
        sizes[i] = chunk_size(i, size, N);
    }

    int* offsets = (int*)malloc(sizeof(double) * size);
    for(int i = 0; i < size; i++){
        offsets[i] = get_offset(i, size, N);
    }

    Matrix_by_vector(N, A, u, b_part, rank, size);
    if(size > 1){
    MPI_Allgatherv(b_part, sizes[rank], MPI_DOUBLE, b, sizes, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    double *X = (double*) malloc(sizeof(double) * N);//Вектор решений
    double epsilon = pow(10, -5);//Точность
    int count_steps;

    double start = MPI_Wtime();
    if(size > 1){
        count_steps = Minimal_Nevazki(N, A, b, X, epsilon, rank, size, sizes, offsets);
    }else{
        count_steps = Minimal_Nevazki(N, A, b_part, X, epsilon, rank, size, sizes, offsets);
    }
    double end = MPI_Wtime();

    printf("Count steps:%d\n", count_steps);
    printf("Time:%lf\n", end - start);

    /*for(int i = 0; i < N; i++){
        printf("X[%d] = %lf   u[%d] = %lf\n",i, X[i], i, u[i]);
    }*/

    for (int i = 0; i < rows_count; ++i)
        free(A[i]);
    free(A);

    free(u);
    free(b);
    free(X);
    free(b_part);
    MPI_Finalize();

    return 0;
}

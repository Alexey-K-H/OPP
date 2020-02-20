#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 10
#define eps 0.00001

int size, rank;

//Count of rows to one process
int get_rows_count_process(int curr_rank){
        int general = N/size;
        int rest = N % size;
        return general + (rank < rest ? 1 : 0);
}

//Offset from the first element in matrix
int get_offset(int curr_rank){
        int res = 0;
        for(int i = 0; i < curr_rank; i++){
                res += get_rows_count_process(i);
        }
        return res;
}

//Create matrix A
double* create_matrix_A(){
        int row_cnt = get_rows_count_process(rank);
        double* part_process = (double*)calloc(row_cnt*N, sizeof(double));
        int offset = get_offset(rank);
        for(int i = 0; i < row_cnt; i++){
                for(int j = 0; j < N;j++){
                        part_process[i*N + j] = (i == (j - offset))? 2 : 1;
                }
        }
        return part_process;
}

//Create vector b
double* create_vector_b(){
        double* vector = (double*)calloc(N, sizeof(double));
        for(int i = 0; i < N; i++){
                vector[i] = N + 1;
        }
        return vector;
}


//Multiple matrix by vector
double* Matrix_by_vector(double *M, double *V)
{
    int row_count_proc = get_rows_count_process(rank);
    double *result = (double*)calloc(row_count_proc, sizeof(double));
    for(int i = 0; i < row_count_proc; i++){
        for(int j = 0; j < N; j++)
        {
            result[i] += M[N*i + j]*V[j];
        }
    }
    return result;
}

//Array of beginigns of parts_processes
int* get_begin_positions(){
        int* positions = (int*)calloc(size, sizeof(int));
        for(int i = 0; i < size; i++){
                positions[i] = get_offset(i);
        }
        return  positions;
}

//Array of sizes of blocks of matrix
int* get_sizes(){
        int* sizes = (int*)calloc(size, sizeof(int));
        for(int i = 0; i < size; i++){
                sizes[i] = get_rows_count_process(i);
        }
        return sizes;
}

double* differ_vectros(double* a, double* b){
        double* res = (double*)calloc(N, sizeof(double));
        for(int i = 0; i < N; i++){
                res[i] = a[i] - b[i];
        }
        return res;
}

double* Minimal_Nevazki(double *A, double *b)
{
    double* X = (double*)calloc(N, sizeof(double));

    int* positions = get_begin_positions();
    int* part_sizes = get_sizes();

    double crit_module;
    double chisl_Tau;
    double del_Tau;

    while(1){
        double* AbX = Matrix_by_vector(A, X);
        double* Ax = (double*)calloc(N, sizeof(double));
        MPI_Allgatherv(AbX, part_sizes[rank], MPI_DOUBLE, Ax, part_sizes, positions, MPI_DOUBLE, MPI_COMM_WORLD);


        double* Y =  differ_vectros(Ax, b);

        double* AbY = Matrix_by_vector(A, Y);
        double* Ay = (double*)calloc(N, sizeof(double));
        MPI_Allgatherv(AbY, part_sizes[rank], MPI_DOUBLE, Ay, part_sizes, positions, MPI_DOUBLE, MPI_COMM_WORLD);


        chisl_Tau = 0.0;
        del_Tau = 0.0;
        for(int i = 0; i < N; i++)
        {
            chisl_Tau += Ay[i]*Y[i];
            del_Tau += Ay[i]*Ay[i];
        }

        double* Xn = (double*)calloc(N, sizeof(double));
        chisl_Tau = chisl_Tau/del_Tau;
        for(int i = 0; i < N; i++){
            Xn[i] = X[i] - chisl_Tau*Y[i];
        }


        double* AbXn = Matrix_by_vector(A, Xn);
        double* AXn = (double*)calloc(N, sizeof(double));
        MPI_Allgatherv(AbXn, part_sizes[rank], MPI_DOUBLE, AXn, part_sizes, positions, MPI_DOUBLE, MPI_COMM_WORLD);

        double crit_1 = 0.0;
        double crit_2 = 0.0;
        for(int i = 0; i < N; i++){
            crit_1 += pow(AXn[i] - b[i], 2);
            crit_2 += pow(b[i], 2);
        }
        crit_1 = sqrt(crit_1);
        crit_2 = sqrt(crit_2);
        crit_module = crit_1/crit_2;

        for(int i = 0; i < N; i++){
            X[i] = Xn[i];
        }

        if(crit_module < eps){
                free(AXn);
                free(AbXn);
                free(Y);
                free(AbY);
                free(Ay);
                free(Xn);
                free(AbX);
                free(Ax);
                break;
        }

    }
    return X;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *A = create_matrix_A();
    double *b = create_vector_b();
    if(rank == 0)
    std::cout << "Curr epsilon:" << eps << std::endl;

    double start_time = MPI_Wtime();
    double *X =  Minimal_Nevazki(A, b);
    double end_time = MPI_Wtime();

    if(rank == 0){
        for(int i = 0; i < N; i++){
           printf("X[%d] = %lf\n", i, X[i]);
        }
        //printf("Time taken: %f\n", end_time - start_time);
    }

    printf("Time taken: %f\n", end_time - start_time);

    free(A);
    free(b);
    free(X);

    MPI_Finalize();
    return 0;
}



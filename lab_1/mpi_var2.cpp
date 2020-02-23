#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 500
#define eps 0.00001

//Count of rows to one process
int get_rows_count_process(int curr_rank, int size){
        int general = N/size;
        int rest = N % size;
        return general + (curr_rank < rest ? 1 : 0);
}

//Offset from the first element in matrix
int get_offset(int curr_rank, int size){
        int res = 0;
        for(int i = 0; i < curr_rank; i++){
                res += get_rows_count_process(i, size);
        }
        return res;
}

//Create matrix A
double* create_matrix_A(int rank, int* block_sizes, int* begin_posit){
        int row_cnt = block_sizes[rank];
        double* part_process = (double*)calloc(row_cnt*N, sizeof(double));
        int offset = begin_posit[rank];
        for(int i = 0; i < row_cnt; i++){
                for(int j = 0; j < N; j++){
                        part_process[i*N + j] = (i == (j - offset))? 2 : 1;
                }
        }
        return part_process;
}

//Create vector b
double* create_vector_b(int rank, int* block_sizes){
        int row_cnt = block_sizes[rank];
        double* vector = (double*)calloc(row_cnt, sizeof(double));
                for(int i = 0; i < row_cnt; i++){
                        vector[i] = N + 1;
                }
        return vector;
}


//Multiple matrix by vector
double* Matrix_by_vector(double *M, double *V, int rank, int size, int* block_sizes, int* offset_posit)
{
    int row_count_proc = block_sizes[rank];
    int offset = offset_posit[rank];
    int process_data_ind = 0;

    double *result = (double*)calloc(row_count_proc, sizeof(double));
    double *v = (double*)calloc(N/size + 1, sizeof(double));
    for(int i = 0; i < row_count_proc; i++){
        v[i] = V[i];
    }

    for(int process = 0; process < size; process++){
        process_data_ind = (rank + process) % size;

        for(int i = 0; i < block_sizes[rank]; i++){
                for(int j = 0; j < block_sizes[process_data_ind]; j++){
                        result[i] += M[i*N + j + offset_posit[process_data_ind]] * v[j];
                }
        }

        MPI_Sendrecv_replace(v, N/size + 1, MPI_DOUBLE, (rank + 1)% size, 100, (rank - 1)% size, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
    return result;
}

//Array of beginigns of parts_processes
int* get_begin_positions(int size){
        int* positions = (int*)calloc(size, sizeof(int));
        for(int i = 0; i < size; i++){
                positions[i] = get_offset(i, size);
        }
        return  positions;
}

//Array of sizes of blocks of matrix
int* get_sizes(int size){
        int* sizes = (int*)calloc(size, sizeof(int));
        for(int i = 0; i < size; i++){
                sizes[i] = get_rows_count_process(i, size);
        }
        return sizes;
}

//differ vectors
double* differ_vectors(double* a, double* b, int rank, int* block_sizes){
        double* res = (double*)calloc(block_sizes[rank], sizeof(double));
                for(int i = 0; i < block_sizes[rank]; i++){
                        res[i] = a[i] - b[i];
                }
        return res;
}

//print matrix
void print_matrix(double* matrix, int rank, int size){
        int part_size = get_rows_count_process(rank, size);
        for(int process = 0; process < size; process++){
                MPI_Barrier(MPI_COMM_WORLD);
                if(rank == process){
                        for(int i = 0; i < part_size; i++){
                                for(int j = 0; j < N; j++){
                                        printf("%0.0f", matrix[i*N + j]);
                                }
                                printf("\n");
                        }
                }
        }
}

//chisl tau
double get_chisl_tau(double* Ay, double* Y, int rank, int* block_sizes){
        double chisl;
        double part_chisl = 0;
        for(int i = 0; i < block_sizes[rank]; i++){
                part_chisl += Ay[i]*Y[i];
        }

        MPI_Allreduce(&part_chisl, &chisl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return chisl;
}

//del tau
double get_del_tau(double* Ay, int rank, int* block_sizes){
        double del;
        double part_del = 0;
        for(int i = 0; i < block_sizes[rank]; i++){
                part_del += Ay[i]*Ay[i];
        }

        MPI_Allreduce(&part_del, &del, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return del;
}

//get crit_1
double get_crit_1(double *Ax, double *b, int rank, int* block_sizes){
        double crit;
        double part_crit = 0;
        for(int i = 0; i < block_sizes[rank]; i++){
                part_crit += pow(Ax[i] - b[i], 2);
        }

        MPI_Allreduce(&part_crit, &crit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return crit;
}

//get crit_2
double get_crit_2(double *b, int rank, int* block_sizes){
        double crit;
        double part_crit = 0;
        for(int i = 0; i < block_sizes[rank]; i++){
                part_crit += pow(b[i], 2);
        }

        MPI_Allreduce(&part_crit, &crit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return crit;
}

//multiple vector by scalar
double* num_by_vector(double scalar, double* vector, int rank, int* block_sizes){
        double* result = (double*)calloc(block_sizes[rank], sizeof(double));
        for(int i = 0; i < block_sizes[rank]; i++){
                result[i] = scalar * vector[i];
        }
        return result;
}

//Minimal Nevazki maethod
double* Minimal_Nevazki(double *A, double *b, int rank, int size, int* block_sizes, int* offset_posit)
{
    double* X = (double*)calloc(N, sizeof(double));

    double crit_module;
    double chisl_Tau;
    double del_Tau;

    while(1){
        double* Ax = Matrix_by_vector(A, X, rank, size, block_sizes, offset_posit);

        double* Y =  differ_vectors(Ax, b, rank, block_sizes);

        double* Ay = Matrix_by_vector(A, Y, rank, size, block_sizes, offset_posit);

        chisl_Tau = get_chisl_tau(Ay, Y, rank, block_sizes);
        del_Tau = get_del_tau(Ay, rank, block_sizes);

        double crit_1 = get_crit_1(Ax, b, rank, block_sizes);
        double crit_2 = get_crit_2(b, rank, block_sizes);

        crit_1 = sqrt(crit_1);
        crit_2 = sqrt(crit_2);
        crit_module = crit_1/crit_2;

        if(crit_module < eps){
                free(Y);
                free(Ay);
                free(Ax);
                break;
        }

        chisl_Tau = chisl_Tau/del_Tau;
        double* ty = num_by_vector(chisl_Tau, Y, rank, block_sizes);
        X = differ_vectors(X, ty, rank, block_sizes);//Update Xn
        free(ty);
    }
    return X;
}

int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *block_sizes = get_sizes(size);
    int *offset_positions = get_begin_positions(size);

    double *A = create_matrix_A(rank, block_sizes, offset_positions);
    double *b = create_vector_b(rank, block_sizes);

    //print_matrix(A, rank, size);
    double start = MPI_Wtime();
    double *X =  Minimal_Nevazki(A, b, rank, size, block_sizes, offset_positions);
    double end = MPI_Wtime();

    /*if(rank == 0){
     for(int i = 0; i < N; i++)
        printf("X[%d]= %lf\n", i, X[i]);
    }*/

    printf("Time taken: %lf sec.\n", end - start);

    free(A);
    free(b);
    free(X);

    MPI_Finalize();
    return 0;
}

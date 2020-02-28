#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <cmath>

#define N 10
#define eps 0.00001

int get_chunk_size(int rank, int size);
int get_offset(int rank, int size);

int* arr_sizes(int size);
int* arr_offset(int size);

double** create_matrix(int rank, int* sizes, int* offsets);
double* create_vector(int rank, int* sizes, int* offsets);

void print_matrix(double** matrix, int rank, int size, int* sizes, int* offsets);
void print_vector(double* vector, int rank, int size, int* sizes);

double* matrix_by_vector(double** matrix, double* vector, int rank, int size, int* sizes, int* offsets);
double* differ_vectors(double* a, double* b, int rank, int* sizes);

double* minimal_nevazki(double** matrix, double* vector, int rank, int size, int* sizes, int* offsets);

double get_chisl_tau(double *Ay, double *y, int rank, int* sizes);
double get_del_tau(double *Ay, int rank, int* sizes);

double get_crit_1(double *Ax, double *b, int rank, int* sizes);
double get_crit_2(double *b, int rank, int* sizes);

double* num_bu_vector(double scalar, double* vector, int rank, int* sizes);

int main(int argc, char **argv){
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *chunk_size = arr_sizes(size);
    int *offset_index = arr_offset(size);

    double **A = create_matrix(rank, chunk_size, offset_index);
    double *b = create_vector(rank, chunk_size, offset_index);

    /*printf("Matrix A\n");*/
    //print_matrix(A, rank, size, chunk_size, offset_index);

    /*printf("Vector b\n");
    print_vector(b, rank, size, chunk_size);*/

    double start = MPI_Wtime();
    double *X =  minimal_nevazki(A, b, rank, size, chunk_size, offset_index);
    double end = MPI_Wtime();

    print_vector(X, rank, size, chunk_size);

    printf("Time taken: %lf sec.\n", end - start);

    free(b);
    free(X);
    for(int i = 0; i < N; i++){
        free(A[i]);
    }
    free(A);

    MPI_Finalize();
    return 0;
}

int get_chunk_size(int rank, int size){
    int base_size = N/size;
    int rest = N % size;
    return base_size + (rank < rest ? 1 : 0);
}

int get_offset(int rank, int size){
    int res = 0;
    for(int i = 0; i < rank; i++){
        res += get_chunk_size(i, size);
    }
    return res;
}

int* arr_sizes(int size){
    int* sizes = (int*)calloc(size, sizeof(int));
    for(int i = 0; i < size; i++){
        sizes[i] = get_chunk_size(i, size);
    }
    return sizes;
}

int* arr_offset(int size){
    int* offsets = (int*)calloc(size, sizeof(int));
    for(int i = 0; i < size; i++){
        offsets[i] = get_offset(i, size);
    }
    return offsets;
}

double** create_matrix(int rank, int* sizes, int* offsets){
    int chunk_size = sizes[rank];
    int offset = offsets[rank];
    double** chunk = (double**)calloc(N, sizeof(double*));
    for (int k = 0; k < N; ++k){
        chunk[k] = (double*)calloc(chunk_size, sizeof(double));
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < chunk_size; j++){
            chunk[i][j] = ((i - offset) == j)? 2 : 1;
        }
    }
    return chunk;
}

double* create_vector(int rank, int* sizes, int* offsets){
    int chunk = sizes[rank];
    double* vector = (double*)calloc(chunk, sizeof(double));
    for(int i = 0; i < chunk; i++){
        vector[i] = N + 1;
    }
    return vector;
}

void print_matrix(double** matrix, int rank, int size, int* sizes, int* offsets){
    int chunk = sizes[rank];
    for(int process = 0; process < size; process++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == process){
            for(int i = 0; i < N; i++){
                for(int j = 0; j < chunk; j++){
                    printf("%0.0f ", matrix[i][j]);
                }
                printf("\n");
            }
        }
    }
}

void print_vector(double* vector, int rank, int size, int* sizes){
    for(int process =  0; process < size; process++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == process){
            for(int i = 0; i < sizes[rank]; i++){
                printf("%0.4f\n", vector[i]);
            }
        }
    }
}

double* matrix_by_vector(double** matrix, double* vector, int rank, int size, int* sizes, int* offsets){
        double* result = (double*)calloc(sizes[rank], sizeof(double));
        double* res = (double*)calloc(N, sizeof(double));

        //print_matrix(matrix, rank, size, sizes, offsets);
        //print_vector(vector, rank, size, sizes);

        for(int num_chunk = 0; num_chunk < sizes[rank]; num_chunk++){
                for(int i = 0; i < N; i++){
                        res[i] += matrix[i][num_chunk]*vector[num_chunk];
                }
        }
        MPI_Allreduce(res, result, sizes[rank], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        free(res);
        return result;
}

double* differ_vectors(double* a, double* b, int rank, int* sizes){
        double* result = (double*)calloc(sizes[rank], sizeof(double));
        for(int i = 0; i < sizes[rank]; i++){
                result[i] = a[i] - b[i];
        }
        return result;
}

double get_chisl_tau(double *Ay, double *y, int rank, int* sizes){
        double chisl;
        double part_chisl = 0;
        for(int i = 0; i < sizes[rank]; i++){
                part_chisl += Ay[i]*y[i];
        }

        MPI_Allreduce(&part_chisl, &chisl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return chisl;
}
double get_del_tau(double *Ay, int rank, int* sizes){
        double del;
        double part_del = 0;
        for(int i = 0; i < sizes[rank]; i++){
                part_del += Ay[i]*Ay[i];
        }

        MPI_Allreduce(&part_del, &del, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return del;
}

double get_crit_1(double *Ax, double *b, int rank, int* sizes){
        double crit_1;
        double part_crit_1 = 0;
        for(int i = 0; i < sizes[rank]; i++){
                part_crit_1 += pow(Ax[i] - b[i], 2);
        }

        MPI_Allreduce(&part_crit_1, &crit_1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return crit_1;
}

double get_crit_2(double *b, int rank, int* sizes){
        double crit_2;
        double part_crit_2 = 0;
        for(int i = 0; i < sizes[rank]; i++){
                part_crit_2 += pow(b[i], 2);
        }

        MPI_Allreduce(&part_crit_2, &crit_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return crit_2;
}

double* num_by_vector(double scalar, double* vector, int rank, int* sizes){
        double* result = (double*)calloc(sizes[rank], sizeof(double));
        for(int i = 0; i < sizes[rank]; i++){
                result[i] = scalar * vector[i];
        }
        return result;
}

double* minimal_nevazki(double** matrix, double* vector, int rank, int size, int* sizes, int* offsets){
        double* X = (double*)calloc(N, sizeof(double));
        //double* X = create_vector(rank, sizes, offsets);

        double crit_module;
        double chislit_t;
        double del_t;

        while(1){
                //Ax
                double* Ax = matrix_by_vector(matrix, X, rank, size, sizes, offsets);
                //print_vector(Ax, rank, size, sizes);

                //Y = Ax - b
                double* Y = differ_vectors(Ax, vector, rank, sizes);
                //print_vector(Y, rank, size, sizes);

                //Ay
                double* Ay = matrix_by_vector(matrix, Y, rank, size, sizes, offsets);
                //print_vector(Ay, rank, size, sizes);

                chislit_t = get_chisl_tau(Ay, Y, rank, sizes);
                del_t = get_del_tau(Ay, rank, sizes);

                double crit_1 = get_crit_1(Ax, vector, rank, sizes);
                double crit_2 = get_crit_2(vector, rank, sizes);

                crit_1 = sqrt(crit_1);
                crit_2 = sqrt(crit_2);

                crit_module = crit_1/crit_2;

                if(crit_module < eps){
                        free(Ax);
                        free(Ay);
                        free(Y);
                        break;
                }
                chislit_t = chislit_t/del_t;
                double* ty = num_by_vector(chislit_t, Y, rank, sizes);
                X = differ_vectors(X, ty, rank, sizes);//Update Xn
                free(ty);
        }

        return X;
}

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int iterCounter = 3;
int L = 50;

typedef struct TaskList{
    int repeatNum;
}TaskList;

TaskList *tl;
int nextPosition;

double globalRes = 0;
int tasksCount = 20;
int processTasksCount;

pthread_t threads[2];
pthread_mutex_t mutex;

void initList(TaskList *taskList, int procTaskCount, int rank, int size, int iterCount);
void createThreads(int rank, int size, int procTaskNum);
void* doTasks(void*);
void* sendTask();

int main(int argc, char** argv) {
    int size;
    int rank;

    int provided;
    double start;
    double end;
    double generalTime;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    pthread_mutex_init(&mutex, NULL);
    processTasksCount = tasksCount / size;

    tl = (TaskList*)malloc(processTasksCount*(sizeof(TaskList)));

    start = MPI_Wtime();
    createThreads(rank, size, processTasksCount);
    end = MPI_Wtime();

    double time = end - start;

    MPI_Reduce(&time, &generalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("Tasks:%d  Time:%f\n", tasksCount, generalTime);
    }

    pthread_mutex_destroy(&mutex);
    free(tl);
    MPI_Finalize();

    return 0;
}

void createThreads(int rank, int size, int procTaskNum){
    pthread_attr_t attr;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    int args[3] = {rank, size, procTaskNum};
    pthread_create(&threads[0], &attr, doTasks, &args);
    pthread_create(&threads[1], &attr, sendTask, NULL);

    pthread_attr_destroy(&attr);
}

void initList(TaskList *taskList, int procTaskCount, int rank, int size, int iterCount){
    for(int i = 0; i < procTaskCount; i++){
        taskList[i].repeatNum = abs(50 - i%100)*abs(rank - (iterCount % size))*L;
    }
}

void* doTasks(void* args){

    int currListNum = 0;

    double start_iteration;
    double end_iteration;

    double time_m;
    double time_n;

    int request;//Запрос на дополнительные задачи

    int iterationCompletedTasksNum;

    int procTaskNum = ((int*)args)[2];
    int rank = ((int*)args)[0];
    int size = ((int*)args)[1];

    while (currListNum != iterCounter){
        initList(tl, procTaskNum, rank, size, currListNum);
        iterationCompletedTasksNum = 0;
        nextPosition = 0;
        start_iteration = MPI_Wtime();

        while(procTaskNum != 0){
            procTaskNum--;

            pthread_mutex_lock(&mutex);

            for(int i = 0; i < tl[nextPosition].repeatNum; i++){
                globalRes += sin(i);
            }

            pthread_mutex_unlock(&mutex);
            iterationCompletedTasksNum++;
            nextPosition++;
        }

        //Проходимся по всем процессам и берем задачи от них, если возможно
        for(int i = 0; i < size; i++){
            request = 1;

            while (1){
                //Отправляем процессам запрос о том, что нужны дополнительные задачи
                MPI_Send(&request, 1, MPI_INT, (rank + i) % size, 0, MPI_COMM_WORLD);

                //Ответ от процессов, есть ли дополнительные свободные задачи
                int response;

                MPI_Recv(&response, 1, MPI_INT, (rank + i) % size, 1, MPI_COMM_WORLD, 0);

                //Если дополнительных задач нет, то завершаем работу цилка и опрашиваем другой процесс
                if(response == -1 || response == 0) break;

                //Иначе выполняется дополнительная задача от другого процесса
                for(int j = 0; j < response; j++){
                    globalRes += sin(j);
                }
                iterationCompletedTasksNum++;
            }
        }

        end_iteration = MPI_Wtime();
        double iterationTimeProc =  end_iteration - start_iteration;
        printf("Process#%d | TasksIterationCount:%d | IterationTime:%f\n", rank, iterationCompletedTasksNum, iterationTimeProc);

        //Ищем time_m и time_n
        MPI_Allreduce(&iterationTimeProc, &time_m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterationTimeProc, &time_n, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0){
            printf("Imbalance:%f\n", time_m - time_n);
            printf("Share of imbalance:%.2f\n", ((time_m - time_n)/time_m) * 100);
        }

        double globalResIteration;
        MPI_Allreduce(&globalRes, &globalResIteration, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        printf("GlobalRes iteration:%.3f\n", globalResIteration);
        currListNum++;
    }

    return NULL;
}

void* sendTask(){
    MPI_Status status;
    int request;
    int response;

    while(1){
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if(request == 0)break;

        int maxLeftTasks;//Наибольшее количество свободных задач среди процессов

        MPI_Allreduce(&processTasksCount, &maxLeftTasks, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(processTasksCount == maxLeftTasks && processTasksCount > 3){
            pthread_mutex_lock(&mutex);
            response = tl[nextPosition].repeatNum;
            nextPosition++;
        }
        else if(processTasksCount == maxLeftTasks && processTasksCount == 0){
            response = -1;
        }
        else if(processTasksCount == maxLeftTasks && processTasksCount <= 3){
            response = 0;
        }

        pthread_mutex_unlock(&mutex);
        MPI_Send(&response, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
    }

    return NULL;
}

# MPI实现矩阵乘法-简单
## 题目
使用MPI实现矩阵乘法。此处为了简单，只需使用MPI实现`n*n`方阵之间的乘法。
## 思路
主进程将第一个相乘矩阵按照行分割成数块，然后分发给子进程；紫禁城将获得的一块矩阵和完整的第二个相乘矩阵相乘，计算得到对应的一块计算结果，返回主进程；主进程收集计算结果，计算过程结束。
## 代码
在这里，我们主要使用`Scatter-Gather`和`BCast`两条指令做数据的分发和接收。
### Scatter-Gather
对于第一个相乘矩阵A，每个进程获得A的各不相同的一部分，所以用`Scatter`分发，用`Gather`接收比较合适。
以下是`MPI_Scatter`的函数原型：
```cpp
int MPI_Scatter(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_datatype,
    int root,
    MPI_Comm communicator
);
```
对于需要分发的消息，需要指定源数据、数据类型和广播消息的进程，并指定发送给每个进程的数据量，以及每个进程接收数据的位置指针。执行这条命令之后，按照进程号顺序，指定数据会分发给每一个进程。
以下是`MPI_Gather`的函数原型：
```cpp
int MPI_Gather(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_datatype,
    int root,
    MPI_Comm communicator
)
```
参数基本相同，从部分数据变量中读取数据，按照进程号顺序被目标进程收集。
注意：使用`Scatter`分配数据后，每个进程分配的部分矩阵具有完全相等的规模。因此。记`comm_sz`为进程数，矩阵的维度需要是`comm_sz`的倍数。我们将矩阵的维度扩展到`comm_sz`的倍数，多余的部分用0填充，保证正确性。
```cpp
// 维度数调整为进程数的倍数
if(n % comm_sz != 0){
    n -= n % comm_sz;
    n += comm_sz;
}
```
另外，主进程也要参与数据分发。
### Broadcast
对于第二个相乘矩阵B，每个进程获得完整的B，所以用`BCast`分发比较合适。
以下是`MPI_Bcast`的函数原型：
```cpp
int MPI_Bcast(
    void * data_p;
    int count;
    MPI_Datatype datatype;
    int source_proc;
    MPI_Comm comm;
);
```
`MPI_Bcast`要求所有进程保有同一个变量指针，然后从一个进程的对应位置复制数据，拷贝到其他进程。
### 程序代码
为了方便测试时间，设计函数`matGene`生成指定大小的矩阵，`vecGene`生成指定大小的向量，生成矩阵之后求乘积，再输出结果。设计代码如下：
```cpp
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

// To Run: 
// mpicxx main.cpp 
// mpiexec -n 4 ./a.out 64

void matGene(int *A, int size, int actural_size){
    // actural size: the matrix we use to calculate has size actural * actural,
    // actural_size may larger than n
    for (int i = 0; i < actural_size; i++){
        for (int j = 0; j < actural_size; j++){
            if(i < size && j < size) A[i * actural_size + j] = rand() % 5; //A[i][j]
            else A[i * actural_size + j] = 0;
        }
    }
}

void matMulti(int *A, int *B, int*C, int row, int n){
    // A: row * n, B: n * n
    for (int i = 0; i < row; i++){
        for (int j = 0; j < n; j++){
            C[i*n + j] = 0;
            for (int k = 0; k < n; k++) 
                C[i*n + j] += A[i*n + k] * B[k*n + j];
        }
    }
}

void matShow(int *A, int row, int size, int actural_size){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < size; j++){
            cout << A[i * actural_size + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]){
    // init
    int n = atoi(argv[1]); // matrix dimension
    double beginTime, endTime; // time record
    srand(time(NULL));

    // MPI Common Head
    int my_rank = 0, comm_sz = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Status status;

    if (comm_sz == 1){ // no parallel
        // Prepare data
        int* A = new int[n * n + 2];
        int* B = new int[n * n + 2];
        int* C = new int[n * n + 2];
        int saveN = n;
        matGene(A, saveN, n);
        matGene(B, saveN, n);

        // Calculate C[i][j] & Time
        beginTime = MPI_Wtime();
        matMulti(A, B, C, n, n);
        endTime = MPI_Wtime();
        cout << "Time: " << endTime - beginTime << endl;

        // Output
        /*
        cout << "A" << endl;
        matShow(A, saveN, saveN, n);
        cout << "B" << endl;
        matShow(B, saveN, saveN, n);
        cout << "C" << endl;
        matShow(C, saveN, saveN, n);
        */
        delete[] A;
        delete[] B;
        delete[] C;
    }

    else{ // parallel: main process collect the result and also involve in calculation

        // prepare data

        int saveN = n;

        // must equal scatter: actural n is bigger than input
        // var n: actural n, var saveN: required size n
        if(n % comm_sz != 0){
            n -= n % comm_sz;
            n += comm_sz;
        }

        int each_row = n / comm_sz;

        // Matrixs

        // matrix vars specified
        int* A;
        int* B = new int[n * n + 2];
        int* C;
        // beginRow = each_row * (my_rank-1), endRow = each_row * my_rank;
        int* partA = new int[each_row * n + 2]; // A[beginRow:endRow, :]
        int* partC = new int[each_row * n + 2]; // C[beginRow:endRow, :]

        // space allocation of A and C: only for the main process
        if(my_rank == 0){
            A = new int[n * n + 2];
            C = new int[n * n + 2];
        }
        
        if (my_rank == 0){

            // Prepare data
            cout << "n = " << n << endl;
            matGene(A, saveN, n);
            matGene(B, saveN, n);  

            beginTime = MPI_Wtime();

        }

        // data distribution & calculate results & collect 

        // Send: Scatter A, Bcast whole B
        MPI_Scatter(&A[0 * n + 0], each_row * n, MPI_INT, &partA[0 * n + 0], \
        each_row * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[0 * n + 0], n * n, MPI_INT, 0, MPI_COMM_WORLD);

        // All processes involve calculation
        matMulti(partA, B, partC, each_row, n);

        // Recv: Gather C
        MPI_Gather(&partC[0 * n + 0], each_row * n, MPI_INT, &C[0 * n + 0], \
        each_row * n, MPI_INT, 0, MPI_COMM_WORLD);

        if (my_rank == 0){

            endTime = MPI_Wtime();
            cout << "Time: " << endTime - beginTime << endl;

            // Output
            /*
            cout << "A" << endl;
            matShow(A, saveN, saveN, n);
            cout << "B" << endl;
            matShow(B, saveN, saveN, n);
            cout << "C" << endl;
            matShow(C, saveN, saveN, n);
            */
        
        }

        // delete

        if(my_rank == 0){
            delete[] A;
            delete[] C;
        }

        delete[] B;
        delete[] partA;
        delete[] partC;

    }

    MPI_Finalize();
    return 0;
}
```
## 实验结果
使用如下命令编译并执行：`mpicxx main.cpp && mpiexec -n num_processes ./a.out mat_dim`，执行可执行文件时带的两个参数分别是矩阵维度和使用的进程数。
测试时间如下：
| 线程数 | 1 | 2 | 4 | 8 | 16 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 256 | 
| 512 | 
| 1024 | 
| 2048 | 
| 4096 | 
表格中最左侧一栏为测试时矩阵维度，表格中时间单位为秒。
![enter image description here](https://raw.githubusercontent.com/WinterShiver/WinterShiver.github.io/master/resources/1.png)
以上为程序运行时间的测试结果展示，为方便展示，横纵坐标均取对数。由图像可见两点：
* 通过直线的大致斜率可见，矩阵乘法的算法是$O(n^3)$的；
* 通过多进程并行执行程序，程序确实能获得对应的加速比；
* 维度数较小时，测量误差较大；超过进程数的加速比是测量误差所导致的。

# MPI实现矩阵乘法-Cannon算法
## 题目
使用MPI实现矩阵乘法。此处为了简单，只需使用MPI实现`n*n`方阵之间的乘法。
## 思路
Cannon算法的思路：如果方阵的维度数是`a`的倍数，那么可以将方阵`a*a`分块，计算每一块的乘积只需要A的对应行和B的对应列。在这里，我们额外要求进程数为`a`，此时每个进程负责计算`a`个块，而且我们规定是每一行的`a`个块。
应用这样分块的方法，可以减小进程中需要给临时乘数矩阵`partA`和`partB`分配的空间。
## 代码
在这里，我们主要使用最基本的`Send`和`Recv`两条指令做数据的分发和接收。
以下是`MPI_Send`的函数原型：
```cpp
int MPI_Send(void *buf, 
             int count, 
             MPI_Datatype datatype, 
             int goal, 
             int tag, 
             MPI_Comm comm
);
```
对于需要分发的消息，需要指定源数据和数据类型；指定接收消息的进程之后，需要指定数据量和进程接收数据的位置指针。在MPI中，每一对`Send`和`Recv`是在系统中建立一个半双工信道，确认信息到达后关闭。为了管理这些通话，参数`tag`管理每个对话的ID.执行这条命令之后，按照进程号顺序，指定数据会分发给每一个进程。
以下是`MPI_Gather`的函数原型：
```cpp
int MPI_Recv(void *buf, 
             int count, 
             MPI_Datatype datatype, 
             int source, 
             int tag, 
             MPI_Comm comm，MPI_Status*status
);
```
参数基本相同，准备接收指定进程的数据，收到之后才能继续运行程序，否则一直阻塞。
另外，主进程也要参与部分矩阵的计算，但是并不参与数据分发。数据是直接从A和B的对应位置拷贝到partA和partB的。
## 程序代码
程序如下所示，在`a`次循环中，主进程把对应的行列发送给每个进程，每个进程依次计算自己对应行的矩阵块，计算完毕返回结果，主进程接收结果并重新赋值写入矩阵C.以下是程序中比较关键的几个设计点：
1.主进程的部分矩阵直接通过赋值得到，而不通过通信，因为进程无法`Send`消息给自己；
2.根据思路部分的分析，程序只允许完全平方数的进程数，如果输入的进程数不是完全平方数会报错；
3.矩阵的维度数必须是`a`的倍数，用如下的程序确定矩阵的维度数：
```cpp
if(n % comm_sz != 0){
    n -= n % comm_sz;
    n += comm_sz;
}
```
将扩大的矩阵部分全部填充为0，这样原来计算部分的计算结果不会出错。
为了方便测试时间，设计函数`matGene`生成指定大小的矩阵，`vecGene`生成指定大小的向量，生成矩阵之后求乘积，再输出结果。
设计代码如下：
```cpp
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace std;

// To Run: 
// mpicxx main.cpp 
// mpiexec -n 4 ./a.out 64

// Cannon Algorithm + Block Multiplication
// Main process: process 0, data distribution & collect calculate results
// Main process involves in calculation, but no communication: 
// A B straight copy to partA partB within itself
// Others: calculate for a block of A * B

void matGene(int *A, int size, int actual_size){
    // actual size: the matrix we use may have a larger dimension than n * n
    for (int i = 0; i < actual_size; i++){
        for (int j = 0; j < actual_size; j++){
            if(i < size && j < size) A[i * actual_size + j] = rand() % 5; //A[i][j]
            else A[i * actual_size + j] = 0;
        }
    }
}

void matMulti(int *A, int *B, int*C, int m, int n, int p){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < p; j++){
            C[i*p + j] = 0;
            for (int k = 0; k < n; k++) 
                C[i*p + j] += A[i*n + k] * B[k*p + j];
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
    // Only Deal With Square Matrixs

    // Calculate Parameters Definition
    int n = atoi(argv[1]); // matrix dimension
    // int beginRow, endRow; // the range of rows calculating in certain process
    double beginTime, endTime; // time record
    srand(time(NULL));

    // MPI Common Head
    int my_rank = 0, comm_sz = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Status status;

    int* A;
    int* B;
    int* C;

    if (comm_sz == 1){ // no parallel
        // Prepare data
        A = new int[n * n + 2];
        B = new int[n * n + 2];
        C = new int[n * n + 2];
        int saveN = n;
        matGene(A, saveN, n);
        matGene(B, saveN, n);

        // Calculate C[i][j] & Time
        beginTime = MPI_Wtime();
        matMulti(A, B, C, n, n, n);
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

        int a = sqrt(comm_sz);
        if(my_rank == 0 && comm_sz != a * a){
            cout << "Not Full Square" << endl;
            MPI_Finalize();
            return 0;
        }


        int saveN = n;
        // must equal scatter: actual n is bigger than input
        if(n % a != 0){
            n -= n % a;
            n += a;
        }   

        int each_row = n / a;
        int* A;
        int* B;
        int* C;
        int* partA = new int[each_row * each_row + 2]; // A[beginRow:endRow, :]
        int* partB = new int[each_row * each_row + 2]; // B[:, beginColumn:endColumn]
        int* partC = new int[each_row * each_row + 2]; // C[beginRow:endRow, beginColumn:endColumn]
        int beginRow, beginColumn;
            
        // Data generation
        if (my_rank == 0){  
            // Prepare data
            cout << "n = " << n << endl;
            A = new int[n * n + 2];
            B = new int[n * n + 2];
            C = new int[n * n + 2];
            matGene(A, saveN, n);
            matGene(B, saveN, n); 
            for(int ii = 0; ii < n; ii++){
                for(int jj = 0; jj < n; jj++){
                    C[ii * n + jj] = 0;
                }
            }  
            beginTime = MPI_Wtime();   
        }

        for(int k = 0; k < a; k++){
            int begin_part = k * each_row;
            // k th
            // Data Distributing
            if (my_rank == 0){
                for(int i = 0; i < comm_sz; i++){
                    // A[beginRow:beginRow+each_row, begin_part:begin_part+each_row]
                    // B[begin_part:begin_part+each_row, beginColumn:beginColumn+each_row]
                    beginRow = (i / a) * each_row;
                    beginColumn = (i % a) * each_row;
                    if(i == 0){
                        // Copy Straightly
                        for(int ii = 0; ii < each_row; ii++){
                            for(int jj = 0; jj < each_row; jj++){
                                partA[ii * each_row + jj] = A[(beginRow + ii) * n + (begin_part + jj)];
                                partB[ii * each_row + jj] = A[(begin_part + ii) * n + (beginColumn + jj)];
                            }
                        }
                    }
                    else{
                        for(int ii = 0; ii < each_row; ii++){
                            MPI_Send(&A[(beginRow + ii) * n + begin_part], each_row, MPI_INT, i, i * each_row + ii, MPI_COMM_WORLD);
                            MPI_Send(&B[(begin_part + ii) * n + beginColumn], each_row, MPI_INT, i, (i + comm_sz) * each_row + ii, MPI_COMM_WORLD);
                        }
                    }
                }
            }
            // Data Receive
            if (my_rank != 0){
                for(int ii = 0; ii < each_row; ii++){
                    MPI_Recv(&partA[ii * each_row + 0], each_row, MPI_INT, 0, my_rank * each_row + ii, MPI_COMM_WORLD, &status);
                    MPI_Recv(&partB[ii * each_row + 0], each_row, MPI_INT, 0, (my_rank + comm_sz) * each_row + ii, MPI_COMM_WORLD, &status);
                }
            }
            // Calculation
            matMulti(partA, partB, partC, each_row, each_row, each_row);
            // Return Result
            if (my_rank != 0){
                for(int ii = 0; ii < each_row; ii++){
                    MPI_Send(&partC[ii * each_row + 0], each_row, MPI_INT, 0, (my_rank + 2 * comm_sz) * each_row + ii, MPI_COMM_WORLD);
                }
            }
            // Data Collection & add
            if (my_rank == 0){
                // C[beginRow:beginRow+each_row, beginColumn:beginColumn+each_row]
                for(int i = 0; i < comm_sz; i++){
                    beginRow = (i / a) * each_row;
                    beginColumn = (i % a) * each_row;
                    if(i == 0){
                        // Copy Straightly
                        for(int ii = 0; ii < each_row; ii++){
                            for(int jj = 0; jj < each_row; jj++){
                                C[(beginRow + ii) * n + (beginColumn + jj)] += partC[ii * each_row + jj];
                            }
                        }
                    }
                    else{  
                        for(int ii = 0; ii < each_row; ii++){
                            int* tmp_partC = new int[each_row + 2];
                            MPI_Recv(&tmp_partC[0], each_row, MPI_INT, i, (i + 2 * comm_sz) * each_row + ii, MPI_COMM_WORLD, &status);
                            for(int jj = 0; jj < each_row; jj++){
                                C[(beginRow + ii) * n + (beginColumn + jj)] += tmp_partC[jj];
                            }
                            delete[] tmp_partC;
                        }
                    }
                }
            }
        }

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
            delete[] A;
            delete[] B;
            delete[] C;
        }

        delete[] partA;
        delete[] partB;
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

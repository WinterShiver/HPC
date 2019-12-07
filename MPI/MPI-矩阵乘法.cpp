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
        MPI_Scatter(&A[0 * n + 0], each_row * n, MPI_INT, &partA[0 * n + 0], each_row * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[0 * n + 0], n * n, MPI_INT, 0, MPI_COMM_WORLD);

        // All processes involve calculation
        matMulti(partA, B, partC, each_row, n);

        // Recv: Gather C
        MPI_Gather(&partC[0 * n + 0], each_row * n, MPI_INT, &C[0 * n + 0], each_row * n, MPI_INT, 0, MPI_COMM_WORLD);

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

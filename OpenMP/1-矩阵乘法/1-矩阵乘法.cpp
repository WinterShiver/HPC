#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stack>
using namespace std;

// Run: g++ -fopenmp main.cpp

void matGene(double *A, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            A[i * size + j] = rand() % 10; //A[i][j]
        }
    }
}

void vecGene(double *A, int size){
    for (int i = 0; i < size; i++){
        A[i] = rand() % 5; //A[i]
    }
}

void matShow(double *A, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cout << A[i * size + j] << " "; //A[i][j]
        }
        cout << endl;
    }
}

void vecShow(double *A, int size){
    for (int i = 0; i < size; i++){
        cout << A[i] << endl; //A[i]
    }
}

int main(int argc, char *argv[]) {

    // data preparation

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    double* A = new double[n * n + 2];
    double* x = new double[n + 2];
    srand(time(NULL));
    matGene(A, n);
    vecGene(x, n);

    // show data
    /*
    cout << "A = " << endl;
    matShow(A, n);
    cout << "x = " << endl;
    vecShow(x, n);
    */ 

    // parallel for

    double* b = new double[n + 2];

    double start_time = omp_get_wtime();

    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<n; i++){
        b[i] = 0;
        for(int j=0; j<n; j++){
            b[i] += A[i * n + j] * x[j];
        }
    }

    double parallel_time = omp_get_wtime() - start_time;

    // show result
    /*
    cout << "b = " << endl;
    vecShow(b, n);
    */

    // show time 

    cout << "Time: " << parallel_time << endl;
    
    return 0;
}

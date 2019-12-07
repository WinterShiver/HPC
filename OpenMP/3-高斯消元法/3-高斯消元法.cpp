#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stack>
using namespace std;

// Run: g++ -fopenmp gauss.cpp

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
    int n = atoi(argv[1]);
    double* A = new double[n * n + 2];
    double* b = new double[n + 2];
    int* serial_num = new int[n + 2];
    srand(time(NULL));
    matGene(A, n);
    vecGene(b, n);

    /*
    cout << "A = " << endl;
    matShow(A, n);
    cout << "b = " << endl;
    vecShow(b, n); */


    for(int i = 0; i < n; i++){
        serial_num[i] = -1;
    }

    stack<int> s1;

    double start_time = omp_get_wtime();
    int num_threads = atoi(argv[2]);

    for(int j = 0; j < n; j++){
        double max_coff = 0;
        int max_index;
        for(int i = 0; i < n; i++){
            if(serial_num[i] == -1 && abs(A[i * n + j]) > abs(max_coff)){
                max_coff = A[i * n + j];
                max_index = i;
                // cout << i << " " << j << endl;
            }
        }
        serial_num[max_index] = j;
        s1.push(max_index);
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i < n; i++){
            if(serial_num[i] == -1){
                double tmp_coff = A[i * n + j] / A[max_index * n + j];
                A[i * n + j] = 0;
                for(int k = j + 1; k < n; k++){
                    A[i * n + k] -= tmp_coff * A[max_index * n + k];
                }
                b[i] -= tmp_coff * b[max_index];
            }
        }
    }

    double* result = new double[n + 2];

    for(int j = n - 1; j >= 0; j--){
        int this_index = s1.top();
        s1.pop();
        serial_num[this_index] = -1;
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i < n; i++){
            if(serial_num[i] != -1){
                double tmp_coff = A[i * n + j] / A[this_index * n + j];
                A[i * n + j] = 0;
                b[i] -= tmp_coff * b[this_index];
            }
        }
        b[this_index] /= A[this_index * n + j];
        A[this_index * n + j] = 1;
        result[j] = b[this_index];
    }

    // cout << "x = " << endl;
    // vecShow(result, n);
    cout << "Time: " << omp_get_wtime() - start_time << endl;
    
    return 0;
}


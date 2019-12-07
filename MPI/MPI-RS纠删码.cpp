#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stack>
using namespace std;

// To Run: 
// mpicxx main.cpp 
// mpiexec -n 4 ./a.out 64

// Cannon Algorithm + Block Multiplication
// Main process: process 0, data distribution & collect calculate results
// Main process involves in calculation, but no communication: 
// A B straight copy to partA partB within itself
// Others: calculate for a block of A * B

void initA(double* A, int size){
    for(int i = 0; i < size; i ++){
        for(int j = 0; j < size; j ++){
            A[i * size + j] = 0;
        }
    }
    for(int i = 0; i < size; i ++){
        A[i * size + i] = 1;
    }
    for(int j = 0; j < size; j ++){
        int base = j + 1;
        int curr = 1;
        for(int i = 0; i < size; i ++){
            A[(i + size) * size + j] = curr;
            curr *= base;
        }
    }
}

void vecGene(double* A, int size){
    for(int i = 0; i < size; i ++){
        A[i] = rand() % 5;
    }
}

void matMulti(double* A, double* B, double*C, int m, int n, int p){
    for (int i = 0; i < m; i ++){
        for (int j = 0; j < p; j ++){
            C[i*p + j] = 0;
            for (int k = 0; k < n; k ++) 
                C[i*p + j] += A[i*n + k] * B[k*p + j];
        }
    }
}

void matShow(double* A, int row, int column){
    for(int i = 0; i < row; i ++){
        for(int j = 0; j < column; j ++){
            cout << A[i * column + j] << "\t";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]){

    // Calculate Parameters Definition

    int n = atoi(argv[1]); // matrix dimension, default 4 or 8
    double beginTime, endTime; // time record, abandoned (not used in this program)
    // srand(time(NULL));
    srand(602);

    // generate data of n-dim

    double* A = new double[2 * n * n + 2];
    double* d = new double[n + 2];
    double* e = new double[2 * n + 2];

    initA(A, n);
    vecGene(d, n);
    matMulti(A, d, e, 2 * n, n, 1);

    cout << "A:" << endl;
    matShow(A, 2 * n, n);
    cout << "d:" << endl;
    matShow(d, n, 1);
    cout << "e:" << endl;
    matShow(e, 2 * n, 1);


    // simulate data lost

    bool* rowIsPreserved = new bool[2 * n + 2];
    int preserved = 0, notPreserved = 0;
    for(int i = 0; i < 2 * n; i ++){
        if(preserved < n && notPreserved < n){
            if(rand() % 2){
                rowIsPreserved[i] = true;
                preserved  ++;
            }
            else{
                rowIsPreserved[i] = false;
                notPreserved  ++;
            }
        }
        else if(preserved < n){
            rowIsPreserved[i] = true;
            preserved  ++;
        }
        else{
            rowIsPreserved[i] = false;
            notPreserved  ++;
        }
    }

    double* B = new double[n * n + 2];
    double* g = new double[n + 2];

    int curr_row = 0;
    for(int i = 0; i < 2 * n; i ++){
        if(rowIsPreserved[i]){
            for(int j = 0; j < n; j ++){
                B[curr_row * n + j] = A[i * n + j];
            }
            g[curr_row] = e[i];
            curr_row  ++;
        }
    }

    cout << "B:" << endl;
    matShow(B, n, n);
    cout << "g:" << endl;
    matShow(g, n, 1);

    delete[] A;
    delete[] d;
    delete[] e;
    delete[] rowIsPreserved;

    // convert Ad = e to Bd = g

    // MPI Common Head

    int my_rank = 0, comm_sz = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Status status;

    // ensure divided exactly

    if(my_rank == 0 && n % comm_sz != 0){
        cout << "Not Divided Exactly" << endl;
        MPI_Finalize();
        return 0;
    }

    // Gauss elimination

    int* serial_num = new int[n + 2];
    if(my_rank == 0){
        for(int i = 0; i < n; i ++){
            serial_num[i] = -1;
        }
    }
    
    stack<int> s1; // only maintain correctness of s1 in main process

    double* this_column;
    double* this_column_in_this_process;

    this_column_in_this_process = new double [n / comm_sz + 2];

    // reduce to up-right triangle

    if(my_rank == 0){
        cout << "Reduce to up-right triangle begin" << endl;
    }

    // iter of reduce to up-right triangle

    for(int j = 0; j < n; j ++){

        // curr column extracted and scattered

        if(my_rank == 0){
            this_column = new double[n + 2];
            for(int i = 0; i < n; i ++){
                this_column[i] = B[i * n + j];
            }
        }

        /*
        if(my_rank == 0){
            cout << "This column in iter " << j << " : ";
            for(int i = 0; i < n; i ++){
                cout << this_column[i] << "\t";
            }
            cout << endl;
        }
        */

        MPI_Scatter(&this_column[0], n / comm_sz, MPI_DOUBLE, &this_column_in_this_process[0], \
                n / comm_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // used rows: broadcast

        MPI_Bcast(&serial_num[0], n, MPI_INT, 0, MPI_COMM_WORLD);

        // each process: find max and send

        double max_coff = 0;
        int max_index;
        
        for(int i = 0; i < n / comm_sz; i ++){
            if(serial_num[i + n / comm_sz * my_rank] == -1 && \
                abs(max_coff) < abs(this_column_in_this_process[i])){
                max_coff = this_column_in_this_process[i];
                max_index = i + n / comm_sz * my_rank;
            }
        }

        // ready to gather result

        double* this_iter_max_coff;
        int* this_iter_max_index;

        if(my_rank == 0){
            this_iter_max_coff = new double[comm_sz + 2];
            this_iter_max_index = new int[comm_sz + 2];
        }

        // gather result

        MPI_Gather(&max_coff, 1, MPI_DOUBLE, &this_iter_max_coff[0], \
            1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&max_index, 1, MPI_INT, &this_iter_max_index[0], \
            1, MPI_INT, 0, MPI_COMM_WORLD);

        // get max

        if(my_rank == 0){
            for(int i = 0; i < comm_sz; i ++){
                if(abs(max_coff) < abs(this_iter_max_coff[i])){
                    max_coff = this_iter_max_coff[i];
                    max_index = this_iter_max_index[i];
                }
            }
        }

        // validation 1-1
        /*
        if(my_rank == 0){
            cout << "Validation 1-1 of Iter " << j << endl;
            cout << max_coff << "\t" << max_index << endl;
            cout << "serial num: ";
            for(int i = 0; i < n; i ++){
                cout << serial_num[i] << "\t";
            }
            cout << endl;
        }
        */

        // record the max one in this iter

        if(my_rank == 0){
            serial_num[max_index] = j;
            s1.push(max_index);
        }

        // validation 1-2
        /*
        if(my_rank == 0){
            cout << "Validation 1-2 of Iter " << j << endl;
            cout << "serial num (after push): ";
            for(int i = 0; i < n; i ++){
                cout << serial_num[i] << "\t";
            }
            cout << endl;
        }
        */

        // used rows: broadcast again, so upgrades not necessary could be avoided

        MPI_Bcast(&serial_num[0], n, MPI_INT, 0, MPI_COMM_WORLD);

        // before reduce of B: bg = [B:g]

        double* bg;

        if(my_rank == 0){
            bg = new double[n * (n + 1) + 2];
            for(int i = 0; i < n; i ++){
                for(int k = 0; k < n; k ++){
                    bg[i * (n + 1) + k] = B[i * n + k];
                }
                bg[i * (n + 1) + n] = g[i];
            }
        } 

        // before reduce of B: the row to reduce broadcasted 

        double* the_row_to_reduce = new double[(n + 1) + 2];

        if(my_rank == 0){
            for(int i = 0; i < n + 1; i ++){
                the_row_to_reduce[i] = bg[max_index * (n + 1) + i];
            }
        }

        MPI_Bcast(&the_row_to_reduce[0], n + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // reduce of B and g: gather

        double* partBg = new double[(n / comm_sz) * (n + 1) + 2];

        MPI_Scatter(&bg[0 * (n + 1) + 0], (n / comm_sz) * (n + 1), MPI_DOUBLE, &partBg[0 * (n + 1) + 0], \
            (n / comm_sz) * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // reduce of B and g: reduce

        for(int i = 0; i < n / comm_sz; i ++){
            if(serial_num[i + n / comm_sz * my_rank] != -1){
                continue;
            }
            double tmp_coff = partBg[i * (n + 1) + j] / the_row_to_reduce[j];
            partBg[i * (n + 1) + j] = 0; // ensure reduced successfully
            for(int k = j + 1; k < n + 1; k ++){
                partBg[i * (n + 1) + k] -= tmp_coff * the_row_to_reduce[k];
            }
        }

        // reduce of B and g: gather

        MPI_Gather(&partBg[0 * (n + 1) + 0], (n / comm_sz) * (n + 1), MPI_DOUBLE, &bg[0 * (n + 1) + 0], \
            (n / comm_sz) * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // [B:g] = B, g

        for(int i = 0; i < n; i ++){
            for(int k = 0; k < n; k ++){
                B[i * n + k] = bg[i * (n + 1) + k];
            }
            g[i] = bg[i * (n + 1) + n];
        }
        
        // validation 2

        /*
        if(my_rank == 0){
            cout << "Validation 2 of Iter " << j << endl;
            cout << "B:" << endl;
            matShow(B, n, n);
            cout << "g:" << endl;
            matShow(g, n, 1);
        }
        */

        if(my_rank == 0){
            delete[] this_column;
            delete[] this_iter_max_coff;
            delete[] this_iter_max_index;
            delete[] bg;
        }

        delete[] the_row_to_reduce;
        delete[] partBg;
        
    }

    if(my_rank == 0){
        cout << "Reduce to up-right triangle end" << endl;
        cout << "B:" << endl;
        matShow(B, n, n);
        cout << "g:" << endl;
        matShow(g, n, 1);
    }

    delete[] serial_num;
    delete[] this_column_in_this_process;

    // reduce to diagram

    if(my_rank == 0){
        cout << "Reduce to diagram begin" << endl;
    }

    // notice: we do not need to upgrade B, but we still need the information of B 
    //         but here we upgrade B, for the convenient of use

    // bg = [B : g]

    double* bg;

    if(my_rank == 0){
        bg = new double[n * (n + 1) + 2];
        for(int i = 0; i < n; i ++){
            for(int k = 0; k < n; k ++){
                bg[i * (n + 1) + k] = B[i * n + k];
            }
            bg[i * (n + 1) + n] = g[i];
        }
    } 

    if(my_rank == 0){
        cout << "Initial bg in diagram" << endl;
        cout << "Bg:" << endl;
        matShow(bg, n, n + 1);
    }

    // iter of reduce to diagram

    for(int j = n - 1; j >= 0; j --){

        int row_in_this_iter;
        double reduce_coff;

        if(my_rank == 0){

            // determine the row in this iter

            row_in_this_iter = s1.top();
            s1.pop();

            // determine normalization coff

            reduce_coff = bg[row_in_this_iter * (n + 1) + n] / bg[row_in_this_iter * (n + 1) + j];

        }

        MPI_Bcast(&row_in_this_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&reduce_coff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // upgrade bg: prepare communication memory, scatter

        double* partBg = new double[(n / comm_sz) * (n + 1) + 2];

        MPI_Scatter(&bg[0 * (n + 1) + 0], (n / comm_sz) * (n + 1), MPI_DOUBLE, &partBg[0 * (n + 1) + 0], \
            (n / comm_sz) * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // upgrade g: reduce

        for(int i = 0; i < n / comm_sz; i ++){
            partBg[i * (n + 1) + n] -= partBg[i * (n + 1) + j] * reduce_coff;
            partBg[i * (n + 1) + j] = 0;
        }

        // upgrade bg: gather

        MPI_Gather(&partBg[0 * (n + 1) + 0], (n / comm_sz) * (n + 1), MPI_DOUBLE, &bg[0 * (n + 1) + 0], \
            (n / comm_sz) * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // small modify

        bg[row_in_this_iter * (n + 1) + n] = reduce_coff;
        bg[row_in_this_iter * (n + 1) + j] = 1;

        // validation 1 in diagram
        /*
        if(my_rank == 0){
            cout << "Validation 1 in diagram, iter " << j << endl;
            cout << "row_in_this_iter: " << row_in_this_iter << endl;
            cout << "Bg:" << endl;
            matShow(bg, n, n + 1);
        }
        */
    }

    // [B:g] = B, g

    for(int i = 0; i < n; i ++){
        for(int k = 0; k < n; k ++){
            B[i * n + k] = bg[i * (n + 1) + k];
        }
        g[i] = bg[i * (n + 1) + n];
    }

    // show result

    if(my_rank == 0){
        cout << "Reduce to diagram end" << endl;
        cout << "B:" << endl;
        matShow(B, n, n);
        cout << "g:" << endl;
        matShow(g, n, 1);
    }

    if(my_rank == 0){
        delete[] bg;
    }

    MPI_Finalize();

    delete[] B;
    delete[] g;
    
    return 0;
}

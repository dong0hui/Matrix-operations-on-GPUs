#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

//#define BLOCK_WIDTH 512

__global__ void printMatrix(float **d_matrix, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i < size && i >= 0) {
        if (j < size + 1 && j >=0) {
            printf("i is %d, j is %d, %f  \n", i, j, d_matrix[i][j]);
        }
    }
}


__global__ void changeFirstElementToOne(float **d_matrix, int pivot, int size, int firstElement) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i == pivot) {
        if (j >= pivot && j < size+1) {
            d_matrix[i][j] = d_matrix[i][j] / firstElement;
        }
    }
}

__global__ void eliminationKernel(float **d_matrix, int pivot, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i > pivot && i < size) {
        if (j > pivot && j < size+1) {
            d_matrix[i][j] = d_matrix[i][j] - d_matrix[i][pivot] * d_matrix[pivot][j];
        }
    }
}

__global__ void setPivotColumnToZero(float **d_matrix, int pivot, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i > pivot && i < size) {
        if (j == pivot) {
            d_matrix[i][j] = 0.0;
        }
    }
}

__global__ void backSubstitution(float **d_matrix, int subLine, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < subLine && i >= 0) {
        if (j == size) {
            d_matrix[i][j] = d_matrix[i][j] - d_matrix[i][subLine] * d_matrix[subLine][size];
        }
    }
}

__global__ void setSubColToZero(float **d_matrix, int subLine, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < subLine && i >= 0) {
        if (j == subLine) {
            d_matrix[i][j] = 0.0;
        }
    }
}

__global__ void writeToDResult(float **d_matrix, int size, float *d_result) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i < size && i >= 0) {
        if (j == size) {
           d_result[i] = d_matrix[i][j];
        }
    }
}

int main(void) {
    // read in data
    std::vector<int> int_list;
    std::string line_;
    std::ifstream file_("test100.txt");

    if (!file_) {
        std::cout << "Cannot open file.\n";
        return 0;
    }

    int size;       // size of the matrix and vectors
    file_ >> size;
    float **matrix; // matrix of the linear system
    matrix = new float*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new float[size+1];
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size + 1; j++) {
                file_ >> matrix[i][j];
        }
    }


    // initialize variable
    float * result, * d_result; // result vector
    float **d_matrix;
    float **d_matrix_h;
    d_matrix_h = (float**)malloc(size * sizeof(float *));

    // alloc space for device copies of a
    cudaMalloc((void **) &d_result, size * sizeof(float));
    cudaMalloc((void **) &d_matrix, size * sizeof(float*));


    // alloc space for host copies of result
    result = (float *)malloc(size * sizeof(float));

    // copy from host to device
    for (int i = 0; i < size; i++) {
        cudaMalloc((void**)&(d_matrix_h[i]), (size+1) * sizeof(float));
        cudaMemcpy(d_matrix_h[i], matrix[i], (size+1) * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_matrix, d_matrix_h, size * sizeof(float *), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((size - 1 + threadsPerBlock.x)/threadsPerBlock.x, (size + 1 - 1 + threadsPerBlock.y)/threadsPerBlock.y);
    
    struct timespec cudalustart = {0,0}; //time of constructing GF
    struct timespec cudaluend = {0,0};
    clock_gettime(CLOCK_REALTIME,&cudalustart);
    // gaussian elimination
    for (int i = 0; i < size; i++) { // i is the pivot line here.
        // change first element of the pivot line to 1
        float firstElement;
        cudaMemcpy(&firstElement, &d_matrix_h[i][i], sizeof(float), cudaMemcpyDeviceToHost);
//        std::cout << firstElement << std::endl;

        changeFirstElementToOne<<<numBlocks, threadsPerBlock>>>(d_matrix, i, size, firstElement);
        // the line under pivot line minus pivot line
        eliminationKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, i, size);
        setPivotColumnToZero<<<numBlocks, threadsPerBlock>>>(d_matrix, i, size);

    }

    // back substitution
    for (int i = size - 1; i > 0; i--) { // form the last line to first line
        // current line is i. every line i 's "b"
        backSubstitution<<<numBlocks, threadsPerBlock>>>(d_matrix, i, size);
        setSubColToZero<<<numBlocks, threadsPerBlock>>>(d_matrix, i, size);
    }

    // write result from d_matrix to d_result
    writeToDResult<<<numBlocks, threadsPerBlock>>>(d_matrix, size, d_result);

    // copy result back to host
    cudaMemcpy(result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME,&cudaluend);
std::cout<<"The time is "<<(cudaluend.tv_sec-cudalustart.tv_sec)*1000+(cudaluend.tv_nsec-cudalustart.tv_nsec)/1000000<<"ms\n";

/*    // print the result
    for (int x = 0; x < size; x++) {
        std::cout << result[x] << std::endl;
    }
*/
    // clean up
    free(matrix); free(result); free(d_matrix_h);
    cudaFree(d_matrix); cudaFree(d_result);

    return 0;
}

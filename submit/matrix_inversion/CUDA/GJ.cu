#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

__global__ void printMatrix(float **d_matrix, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i < size && i >= 0) {
        if (j < size && j >=0) {
            printf("i is %d, j is %d, %f  \n", i, j, d_matrix[i][j]);
        }
    }
}

__global__ void changeFirstElementToOne(float **d_matrix, float **d_inversion, int pivot, int size, float firstElement) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i == pivot) {
        if (j >= 0 && j < size) {
            d_matrix[i][j] = d_matrix[i][j] / firstElement;
            d_inversion[i][j] = d_inversion[i][j] / firstElement;
        }
    }
}


__global__ void GJKernel(float **d_matrix, float **d_inversion, int pivot, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i >= 0 && i < size && i != pivot && j < size && j >= 0) {
        if (j != pivot) {
            d_matrix[i][j] = d_matrix[i][j] - d_matrix[i][pivot] * d_matrix[pivot][j];
        }
        d_inversion[i][j] = d_inversion[i][j] - d_matrix[i][pivot] * d_inversion[pivot][j];
    }
}

__global__ void setPivotColumnToZero(float **d_matrix, int pivot, int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (i >= 0 && i < size && i != pivot) {
        if (j == pivot) {
            d_matrix[i][j] = 0.0;
        }
    }
}

int main(void) {
    // read in data
    std::ifstream file_("test100.txt");

    if (!file_) {
        std::cout << "Cannot open file.\n";
        return 0;
    }

    int size;       // size of the matrix
    file_ >> size;
    float **matrix; // matrix to inverse
    matrix = new float*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new float[size];
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
                file_ >> matrix[i][j];
        }
    }

    // initialize variable
    float **inversion, **d_inversion; // result
    float **d_inversion_h;
    d_inversion_h = (float**)malloc(size * sizeof(float *));
    float **d_matrix;
    float **d_matrix_h;
    d_matrix_h = (float**)malloc(size * sizeof(float *));

    // alloc space for device copies
    cudaMalloc((void **)&d_inversion, size * sizeof(float*));
    cudaMalloc((void **)&d_matrix, size * sizeof(float*));

    // alloc space for host copies
    inversion = (float**)malloc(size * sizeof(float *));

    // initial inversion
    for (int i = 0; i < size; i++) {
        inversion[i] = (float*)malloc(size * sizeof(float));
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) inversion[i][j] = 1.0;
            else inversion[i][j] = 0.0;
        }
    }

    // copy from host to device
    for (int i = 0; i < size; i++) {
        cudaMalloc((void**)&(d_matrix_h[i]), size * sizeof(float));
        cudaMemcpy(d_matrix_h[i], matrix[i], size * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_matrix, d_matrix_h, size * sizeof(float*), cudaMemcpyHostToDevice);

    for (int i = 0; i < size; i++) {
        cudaMalloc((void**)&(d_inversion_h[i]), size * sizeof(float));
        cudaMemcpy(d_inversion_h[i], inversion[i], size * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_inversion, d_inversion_h, size * sizeof(float*), cudaMemcpyHostToDevice);

    // threadsPerBlock, numBlocks
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((size - 1 + threadsPerBlock.x)/threadsPerBlock.x, (size - 1 + threadsPerBlock.y)/threadsPerBlock.y);

    struct timespec cudalustart = {0,0}; //time of constructing GF
    struct timespec cudaluend = {0,0};
    clock_gettime(CLOCK_REALTIME,&cudalustart);
    // Gauss-Jordan
    for (int i = 0; i < size; i++) {
        // change first element of the pivot line to 1
        float firstElement;
        cudaMemcpy(&firstElement, &d_matrix_h[i][i], sizeof(float), cudaMemcpyDeviceToHost);
        changeFirstElementToOne<<<numBlocks, threadsPerBlock>>>(d_matrix, d_inversion, i, size, firstElement);
        GJKernel<<<numBlocks, threadsPerBlock>>>(d_matrix, d_inversion, i, size);
        setPivotColumnToZero<<<numBlocks, threadsPerBlock>>>(d_matrix, i, size);
    }


    // copy result from d_inversion to inversion
    for (int i = 0; i < size; i++) {
        cudaMemcpy(inversion[i], d_inversion_h[i], size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    clock_gettime(CLOCK_REALTIME,&cudaluend);
    std::cout<<"The time is "<<(cudaluend.tv_sec-cudalustart.tv_sec)*1000+(cudaluend.tv_nsec-cudalustart.tv_nsec)/1000000<<"ms\n";

/*
    // print the result
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << inversion[i][j] << "  ";
        }
        std::cout << std::endl;
    }
*/
    // clean up
    free(matrix); free(inversion); free(d_matrix_h); free(d_inversion_h);
    cudaFree(d_matrix); cudaFree(d_inversion);

    return 0;
}
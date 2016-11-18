#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <malloc.h>
#include "CL/cl.h"

const char *ProgramSource0 =
"__kernel void changeFirstElementToOne(\n"\
"__global float *d_matrix, __global int* pivot, int size, __global float* firstElement)\n"\
"{\n"\
"    int i = get_global_id(0);\n"\
"    int j = get_global_id(1);\n"\
"    int lenY = get_global_size(1);\n"\
"    if (i == (*pivot)) {\n"\
"        if (j >= (*pivot) && j < size + 1) {\n"\
"            d_matrix[i * lenY + j] = d_matrix[i * lenY + j] / (*firstElement);\n"\
"        }\n"\
"    }\n"\
"}\n";

const char *ProgramSource1 =
"__kernel void modifiedEliminationKernel(__global float *d_matrix, __global int* pivot , int size)\n"\
"{\n"\
"    int i = get_global_id(0);\n"\
"    int j = get_global_id(1);\n"\
"    int lenY = get_global_size(1);\n"\
"    if (i >= 0 && i < size && i != (*pivot)) {\n"\
"        if (j > (*pivot) && j < size+1) {\n"\
"            d_matrix[i * lenY + j] = d_matrix[i*lenY+j] - d_matrix[i*lenY+(*pivot)]*d_matrix[(*pivot)*lenY+j];\n"\
"        }\n"\
"    }\n"\
"}\n";

const char *ProgramSource2 =
"__kernel void setPivotColumnToZero(__global float *d_matrix, __global int* pivot, int size) {\n"\
"    int i = get_global_id(0);\n"\
"    int j = get_global_id(1);\n"\
"    int lenY = get_global_size(1);\n"\
"    if (i >= 0 && i < size && i != (*pivot)) {\n"\
"        if (j == (*pivot)) {\n"\
"            d_matrix[i * lenY + j] = 0.0;\n"\
"        }\n"\
"    }\n"\
"}\n";

const char *ProgramSource5 =
"__kernel void writeToDResult(__global float *d_matrix, int size, __global float *d_result) {\n"\
"int i = get_global_id(0);\n"\
"    int j = get_global_id(1);\n"\
"    int lenY = get_global_size(1);\n"\
"    if (i < size && i >= 0) {\n"\
"        if (j == size) {\n"\
"            d_result[i] = d_matrix[i * lenY + j];\n"\
"        }\n"\
"   }\n"\
"}\n";

int main(void) {
    cl_context context;
    cl_context_properties properties[3];
    cl_command_queue command_queue;
    cl_int err;
    cl_uint num_of_platforms = 0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices = 0;
    cl_mem d_matrix, d_result;

    cl_program program_changeFirstElementToOne;
    cl_kernel kernel_changeFirstElementToOne;
    cl_program program_modifiedEliminationKernel;
    cl_kernel kernel_modifiedEliminationKernel;
    cl_program program_setPivotColumnToZero;
    cl_kernel kernel_setPivotColumnToZero;
    cl_program program_writeToDResult;
    cl_kernel kernel_writeToDResult;

    // read in data
    FILE *file;
    file = fopen("test100.txt", "r");
    int size;
    fscanf(file, "%d", &size);
    int data_size = size * (size + 1);
    float *matrix; // matrix of the linear system
    matrix = malloc(data_size * sizeof(float));

    int i = 0;
    for (i = 0; i < data_size; i++)
    {
        if (!fscanf(file, "%f", &matrix[i]))
        {
            break;
        }
    }

    fclose(file);

    // retreive a list of platforms avaible
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
        printf("Unable to get platform_id\n");
        return 1;
    }

    // try to get a support GPU device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS) {
        printf("Unable to get device_id\n");
        return 1;
    }

    // context properties list - must be terminated with 0
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)platform_id;
    properties[2] = 0;

    // create a context with the GPU device
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

    // create command queue using the context and device
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);

////////////////////////create program
    // create program -- changeFirstElementToOne
    program_changeFirstElementToOne = clCreateProgramWithSource(context, 1, (const char**)&ProgramSource0, NULL, &err);

    // compile program
    if (clBuildProgram(program_changeFirstElementToOne, 1, &device_id, NULL, NULL, NULL) != CL_SUCCESS) {
        printf("Error building program0\n");
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_changeFirstElementToOne, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(program_changeFirstElementToOne, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        return 1;
    }
    // create kernel
    kernel_changeFirstElementToOne = clCreateKernel(program_changeFirstElementToOne, "changeFirstElementToOne", &err);


    // create program -- modifiedEliminationKernel
    program_modifiedEliminationKernel = clCreateProgramWithSource(context, 1, (const char**)&ProgramSource1, NULL, &err);
    // compile program
    if (clBuildProgram(program_modifiedEliminationKernel, 1, &device_id, NULL, NULL, NULL) != CL_SUCCESS) {
        printf("Error building program1\n");
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_modifiedEliminationKernel, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(program_modifiedEliminationKernel, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        return 1;
    }
    // create kernel
    kernel_modifiedEliminationKernel = clCreateKernel(program_modifiedEliminationKernel, "modifiedEliminationKernel", &err);

    // create program -- setPivotColumnToZero
    program_setPivotColumnToZero = clCreateProgramWithSource(context, 1, (const char**)&ProgramSource2, NULL, &err);
    // compile program
    if (clBuildProgram(program_setPivotColumnToZero, 1, &device_id, NULL, NULL, NULL) != CL_SUCCESS) {
        printf("Error building program2\n");
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_setPivotColumnToZero, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(program_setPivotColumnToZero, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        return 1;
    }
    // create program
    kernel_setPivotColumnToZero = clCreateKernel(program_setPivotColumnToZero, "setPivotColumnToZero", &err);


    // create program -- writeToDResult
    program_writeToDResult = clCreateProgramWithSource(context, 1, (const char**)&ProgramSource5, NULL, &err);
    // compile program
    if (clBuildProgram(program_writeToDResult, 1, &device_id, NULL, NULL, NULL) != CL_SUCCESS) {
        printf("Error building program5\n");
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_writeToDResult, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        // Get the log
        clGetProgramBuildInfo(program_writeToDResult, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        return 1;
    }
    // create kernel
    kernel_writeToDResult = clCreateKernel(program_writeToDResult, "writeToDResult", &err);

/////////////////////////////////

// create buffers
    d_matrix = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, NULL);

    // load data into the d_matrix buffer
    clEnqueueWriteBuffer(command_queue, d_matrix, CL_TRUE, 0, sizeof(float) * data_size, matrix, 0, NULL, NULL);

    cl_mem d_pivot;
    d_pivot = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
    cl_mem d_firstElement;
    d_firstElement = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

    float firstElement = 1.0;

    // global, local size
    const size_t global[2] = {size, size + 1};
    const size_t local[2] = {16, 16};

    // set Arg
    clSetKernelArg(kernel_changeFirstElementToOne, 0, sizeof(cl_mem), &d_matrix);
    clSetKernelArg(kernel_changeFirstElementToOne, 1, sizeof(cl_mem), &d_pivot);
    clSetKernelArg(kernel_changeFirstElementToOne, 2, sizeof(cl_int), &size);
    clSetKernelArg(kernel_changeFirstElementToOne, 3, sizeof(cl_mem), &d_firstElement);

    clSetKernelArg(kernel_modifiedEliminationKernel, 0, sizeof(cl_mem), &d_matrix);
    clSetKernelArg(kernel_modifiedEliminationKernel, 1, sizeof(cl_mem), &d_pivot);
    clSetKernelArg(kernel_modifiedEliminationKernel, 2, sizeof(cl_int), &size);

    clSetKernelArg(kernel_setPivotColumnToZero, 0, sizeof(cl_mem), &d_matrix);
    clSetKernelArg(kernel_setPivotColumnToZero, 1, sizeof(cl_mem), &d_pivot);
    clSetKernelArg(kernel_setPivotColumnToZero, 2, sizeof(cl_int), &size);

    clSetKernelArg(kernel_writeToDResult, 0, sizeof(cl_mem), &d_matrix);
    clSetKernelArg(kernel_writeToDResult, 1, sizeof(cl_int), &size);
    clSetKernelArg(kernel_writeToDResult, 2, sizeof(cl_mem), &d_result);

    struct timespec cudalustart = {0,0}; //time of constructing GF
    struct timespec cudaluend = {0,0};
    clock_gettime(CLOCK_REALTIME,&cudalustart);

    // gaussian elimination
    for (i = 0; i < size; i++) {
        clEnqueueReadBuffer(command_queue, d_matrix, CL_TRUE, sizeof(float)*(i*(size+1)+i), sizeof(float), &firstElement, 0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, d_firstElement, CL_TRUE, 0, sizeof(float), &firstElement, 0, NULL, NULL);
        int pivot = i;
        clEnqueueWriteBuffer(command_queue, d_pivot, CL_TRUE, 0, sizeof(int), &pivot, 0, NULL, NULL);

        clEnqueueNDRangeKernel(command_queue, kernel_changeFirstElementToOne, 2, 0, global, local, 0, NULL, NULL);
        clEnqueueNDRangeKernel(command_queue, kernel_modifiedEliminationKernel, 2, 0, global, local, 0, NULL, NULL);
        clEnqueueNDRangeKernel(command_queue, kernel_setPivotColumnToZero, 2, 0, global, local, 0, NULL, NULL);
    }

    // writeToDResult
    clEnqueueNDRangeKernel(command_queue, kernel_writeToDResult, 2, 0, global, local, 0, NULL, NULL);

    // clFinish
    clFinish(command_queue);

    // copy result back to host
    float * result;
    result = (float *)malloc(size * sizeof(float));
    clEnqueueReadBuffer(command_queue, d_result, CL_TRUE, 0, sizeof(float) * size, result, 0, NULL, NULL);

clock_gettime(CLOCK_REALTIME,&cudaluend);
printf("Thetime is %d   ms\n", (cudaluend.tv_sec-cudalustart.tv_sec)*1000+(cudaluend.tv_nsec-cudalustart.tv_nsec)/1000000);

/*
   // print the result
    printf("\n\n\n\n");
    int x = 0;
    for (x = 0; x < size; x++) {
        printf("%f \n", result[x]);
    }
*/
    // clean up - release OpenCL resources
    clReleaseMemObject(d_matrix);
    clReleaseMemObject(d_result);
    clReleaseMemObject(d_pivot);
    clReleaseMemObject(d_firstElement);

    clReleaseProgram(program_changeFirstElementToOne);
    clReleaseProgram(program_modifiedEliminationKernel);
    clReleaseProgram(program_setPivotColumnToZero);
    clReleaseProgram(program_writeToDResult);

    clReleaseKernel(kernel_changeFirstElementToOne);
    clReleaseKernel(kernel_modifiedEliminationKernel);
    clReleaseKernel(kernel_setPivotColumnToZero);
    clReleaseKernel(kernel_writeToDResult);


    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
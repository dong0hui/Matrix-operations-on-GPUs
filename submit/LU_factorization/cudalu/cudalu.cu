#include "cublas_v2.h"
#include <algorithm> /*min*/
#include <math.h> /*fabs*/
#include <iostream>
#include <cuda.h>
#include <string>
#include "/home1/02511/donghui/lu_decompose/helper_cuda.h"
#include <fstream>
#include <sstream>
#include <time.h>

class InfoStat {
	public:
	std::string _str;
	int _info;
};

//***************************************************************************************  
// Reference: http://docs.nvidia.com/cuda/cublas
//void DecomposeLU( int M, int N, int lda , float* A, int* permute, float epsilon, InfoStat& stat)  
//  
// M:Num of rows of A  
// N:Num of column of A  
// A:Float Matrix of size M*N  
//                on the output contains the result of the LU decomposition  
//                The diagonal elements for L are not stored in A ( assuming they are all 1)  
//lda:Leading dim of A lda < std::max(1,M)  
//P:Permutation vector of size M  
//epsilon:Epsilon (used to test for singularities)  
//stat:return status  
//compile on stampede: nvcc -arch=compute_35 -code=sm_35 -lcublas -o lu_decomp.o lu_decomp.cu
// **************************************************************************************  
void DecomposeLU(int M, int N, int lda , float* A, int* P, float epsilon, InfoStat& stat)  
{  
     //Preconditions where we cannot do LU decomposition
     if ( M<=0 || N<=0 || lda < std::max(1,M) )  
     {  
          stat._info = -1;  
          if (M<=0)  
              stat._str = "M<=0";  
          if (N<=0)  
              stat._str = "M<=0";  
          if (lda < std::max(1,M))  
              stat._str = "lda < std::max(1,M)";  
          return;  
     } 
	 
	 /*
	 Get handle to the CUBLAS context
	 */
	 cublasHandle_t cublasHandle = 0;
	 cublasStatus_t cublasStatus;
	 cublasStatus = cublasCreate(&cublasHandle);
	 
	 checkCudaErrors(cublasStatus);
	 
	 float* d_A; //A on device
	 checkCudaErrors(cudaMalloc((void **) &d_A, M*N*sizeof(float)));
	 cudaMemcpy(d_A, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	 
	 float minusOne = -1.0f;
	 int pivotRow;
     int minDim = std::min( M, N );  
     for (int k=0; k<minDim-1; k++)  
     {  
        cublasIsamax(cublasHandle, M-k, d_A + k*lda, 1, &pivotRow);
		pivotRow += k-1; // row relative to the current submatrix  
        int kp1 = k+1;  
        P[k] = pivotRow;  
        if (pivotRow!=k)  
        {  
            cublasSswap(cublasHandle, N, d_A+pivotRow, lda, d_A+k, lda);  
        }  
        float valcheck;  
        cublasGetVector(1,sizeof(float),d_A+k+k*lda, 1, &valcheck, 1);  
        if (fabs(valcheck) < epsilon)  
        {  
            stat._info =k+1;  
            stat._str = " Matrix is Singular ";  
            return;  
        }  
		float valcheckInv = 1.0f/valcheck;
        if (kp1 < M)  
        {  
            cublasSscal(cublasHandle, M-kp1, &valcheckInv,d_A+kp1+ k*lda, 1);  
        }  
        if ( kp1 < minDim )  
        {  
            cublasSger (cublasHandle, M-kp1, N-kp1, &minusOne, d_A+kp1+ k*lda, 1, d_A+k+ kp1*lda, lda, d_A+kp1*lda+kp1, lda);  
        }  
     }  
     // copy memory from device to host
	cudaMemcpy(A, d_A, M*N*sizeof(float), cudaMemcpyDeviceToHost);
}  

int main(int argc, char** argv) {
	//1st param is M, N, or lda
	int M = (argc > 1) ? atoi(argv[1]):3;
	int N = M;
	float* A = (float *)malloc(sizeof(float)*M*N);
	/*
	A[0] = 8.0; A[1] = 4.0; A[2] = 6.0;
	A[3] = 2.0; A[4] = 9.0; A[5] = 7.0;
	A[6] = 9.0; A[7] = 4.0; A[8] = 9.0;
	*/
	std::string line_;
	std::ifstream file_("test10000.txt");
	if(file_.is_open()) {
        while (getline(file_, line_)) {
            std::stringstream ss(line_);
            int i;
			double elem;
            for (i = 0; i < M*N; i++) {
				ss >> elem;
                A[i] = elem;
                if (ss.peek() == ',' || ss.peek() == ' ') {
                    ss.ignore();
                }
            }
        }
        file_.close();
    } 
	
	int *P = (int *) malloc(sizeof(int) * M);
	for(int i = 0; i < M; i++) {
		P[i] = i;
	}
	float epsilon = 0.001f;
	InfoStat stat;
	
	int lda = M;
	struct timespec cudalustart = {0,0}; //time of constructing GF
	struct timespec cudaluend = {0,0};
	clock_gettime(CLOCK_REALTIME,&cudalustart);
	DecomposeLU(M, N, lda, A, P, epsilon, stat);
	clock_gettime(CLOCK_REALTIME,&cudaluend);
	std::cout<<"Time of basic CUDA LU is "<<(cudaluend.tv_sec-cudalustart.tv_sec)*1000+(cudaluend.tv_nsec-cudalustart.tv_nsec)/1000000<<"ms\n";
	/*
	std::cout <<"stat: "<<stat._str<<std::endl;
	for (int r = 0; r < M; r++) {
		for (int c= 0; c < N; c++) {
			std::cout << A[r + c*M] << " ";
		}
		std::cout << std::endl;
	}
	*/
}

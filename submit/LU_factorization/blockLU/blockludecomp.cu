//****************************************************************************************
//compile on stampede: nvcc -arch=compute_35 -code=sm_35 -lcublas -o lu_decomp.o lu_decomp.cu
//****************************************************************************************
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

#define debug 0 
//***************************************************************************************  
//Reference: 
//<<Handbook on Parallel and Distributed Processing>> 2000
//http://www.netlib.org/utk/papers/siam-review93/node13.html
//http://docs.nvidia.com/cuda/cublas
//1. cublasIsamax/cublasSswap for partial pivoting
//2. cublasSscal for l_ik = a_ik/u_kk calculation
//3. cublasSger for updating submatrix A = A - [l_ik]*[u_kj]
//void DecomposeLU( int M, int N, int lda , float* A, int* permute, float epsilon, InfoStat& stat)  
//  
// M:Num of rows of A  
// N:Num of column of A  
// A:Float Matrix of size M*N  
//                on the output contains the result of the LU decomposition  
//                The diagonal elements for L are not stored in A ( assuming they are all 1)  
//lda:Leading dim of A lda < std::max(1,M), it is number of rows
//P:Permutation vector of size M  
//epsilon:Epsilon (used to test for singularities)  
//stat:return status  
// **************************************************************************************  
bool DecomposeLU(int M, int N, int lda , float* A, int* P, float epsilon, InfoStat& stat)  
{  
     //Preconditions where we cannot do LU decomposition
	 //M and N should be at least 1. lda should be exactly M.
     if ( M<=0 || N<=0 || lda < std::max(1,M) )  
     {  
		stat._info = -1;  
		if (M<=0)  
		  stat._str = "Not enough rows";  
		if (N<=0)  
		  stat._str = "Not enough columns";  
		if (lda < std::max(1,M))  
		  stat._str = "Leading dimension of A is wrong";  
		return false;  
     } 
	 
	 /*
	 Get handle to the CUBLAS context
	 */
	 cublasHandle_t cublasHandle = 0;
	 cublasStatus_t cublasStatus;
	 cublasStatus = cublasCreate(&cublasHandle);
	 
	 //Here helper_cuda.h and helper_string.h in the same directory are used
	 checkCudaErrors(cublasStatus);
	 
	 //A (dimension of MxN) on device memory, it is 1xMN array, with M|M|...|M of size N
	 float* d_A;
	 checkCudaErrors(cudaMalloc((void **) &d_A, M*N*sizeof(float)));
	 cudaMemcpy(d_A, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	 
	 //minusOne is used when updating submatrix of A, A=A+(minusOne)*L_ik*U_kj
	 float minusOne = -1.0f;
	 //Here we use partial pivoting, one find the max in kth column, not the remaining whole matrix
	 int pivotRow;
     int minDim = std::min( M, N );  
     for (int k=0; k<minDim-1; k++)  
     {  
		/* 
		d_A is the address of starting point of d_A, and d_A+k*lda is the kth column
		Because there is only M-k elements left in the kth column in kth step, this pivotRow
		is partial pivoting. But in cuBLAS, index starts from 1.
		*/
        cublasIsamax(cublasHandle, M-k, d_A + k*lda, 1, &pivotRow);
		/*
		if pivotRow is 1 which is 1st element in cuBLAS, 
		now pivotRow is updated to 0 which is 1st element in d_A when k=0 (first step).
		It is the same in second, third, fourth,... steps
		*/
		pivotRow += k-1; // row relative to the current submatrix  
        int kbase1 = k+1;  
        P[k] = pivotRow;  
       /*
	 if (pivotRow!=k)  
        {  
			//Cap S is single-precision, D is Double, C is complex, Z is double complex
			//A is written colume by colume
            cublasSswap(cublasHandle, N, d_A+pivotRow, lda, d_A+k, lda);  
        }  
	*/	
		//e.g. k=1, valcheck=u11
        float valcheck;  
        cublasGetVector(1,sizeof(float),d_A+k+k*lda, 1, &valcheck, 1);  
		
		//In this case, det(A)=0, inv(A) is null.
        if (fabs(valcheck) < epsilon)  
        {  
            stat._info =k+1;  
            stat._str = " Matrix is Singular ";  
            return false;  
        }  
		
		float valcheckInv = 1.0f/valcheck;
        if (kbase1 < M)  
        {  
			//e,g, k=1, Compute l21,l31,.... which are a21/u11, a31/u11,...
            cublasSscal(cublasHandle, M-kbase1, &valcheckInv,d_A+kbase1+ k*lda, 1);  
        }  
        if ( kbase1 < minDim )  
        {  
			//update A with A-[l21, l31]^T*[u12 u13]
            cublasSger (cublasHandle, M-kbase1, N-kbase1, &minusOne, d_A+kbase1+ k*lda, 1, d_A+k+ kbase1*lda, lda, d_A+kbase1*lda+kbase1, lda);  
        }  
     }  
     // copy memory from device to host
	cudaMemcpy(A, d_A, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	//cublasDestroy(cublasHandle);
	//cudaFree(d_A);
	//cudaDeviceReset();
	return true;
}  

//***************************************************************************************
//void DecomposeBlockedLU (	int M, int N,int lda,float *A, int* P, int blockSize, float epsilon, infoStat &stat)
//blockSize is r in my notes.
//P is pivot array
// **************************************************************************************
bool DecomposeBlockedLU(int M, int N, int lda, float *A, int* P, int blockSize, float epsilon, InfoStat &stat) {
	if (M < 0 || N < 0 || lda < std::max(1,M) ) {
		stat._info = -1;
		if (M<=0)
			stat._str = "Row is not enough";
		if (N<=0)
			stat._str = "column is not enough";
		if (lda < std::max(1,M))
			stat._str = "leading dimension is wrong"; 
		return false;
	}
	
	//If one specifies blockSize to be 1, or blockSize is larger than total dimension of A
	//, then we use DecomposeLU above directly.
	int minSize = std::min(M,N);
	if(blockSize > minSize || blockSize == 1) {
		return DecomposeLU(M, N, lda, A, P, epsilon, stat);
	}
	
	//Here we start to implement block LU decomposition
	//Construct CUBLAS handle as before, refer to comments before about the head files
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	checkCudaErrors(cublasStatus);
	
	float *d_A;
	checkCudaErrors(cudaMalloc((void **)&d_A, M*N*sizeof(float)));
	cudaMemcpy(d_A, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	//Basically, we need to allocate memory space for these two numbers, we cannot use constant, but variables
	//Or you can think that GPUs don't know constants as CPUs
	float plusOne = 1.0f;
	float minusOne = -1.0f;
	
	for(int iBlock = 0; iBlock < minSize; iBlock += blockSize) {
		/*
		In the last iteration, remaining block size is quite possible to be smaller than blockSize
		iBlock = 0, blockSize, 2blockSize, ...
		realBS: real block size
		*/
		int realBS = std::min(minSize - iBlock, blockSize);
//#if debug
//std::cout<<"iBlock is"<<iBlock<<std::endl;
//#endif		
		//numElements is the number of elements in the remaining submatrix A10
		int numElements = (M-iBlock)*realBS;
		float* ABlock = (float *)malloc(sizeof(float)*numElements);
		
#if debug
std::cout<<"numElements is "<<numElements<<std::endl;
#endif		
		//Copy device memory to host memory ABlock
		//d_A+iBlock*lda is 0+(k*r)*M in kth step 
		cublasGetVector(numElements, sizeof(float), d_A+iBlock+iBlock*lda, 1, ABlock, 1);
		
		//DecomposeLU(int M, int N, int lda , float* A, int* P, float epsilon, InfoStat& stat)
		//M = M-iBlock; N = realBS; lda; A=ABlock; Decompose A10
		//bool ok = DecomposeLU( M-iBlock, realBS, lda, ABlock, P+iBlock, epsilon, stat);
		bool ok = DecomposeLU( M-iBlock, realBS, M-iBlock, ABlock, P+iBlock, epsilon, stat);
		if (!ok){
			return false;
		}
		cublasSetVector(numElements, sizeof(float), ABlock, 1, d_A+iBlock+iBlock*lda, 1);	
		/*
		//Blocked partial pivoting, in debug mode, one can comment out this section by testing matrix without 0 elements
		//irow is from k*r to k*r+r
		//In referece: p is irow, i is iBlock
		for (int irow = iBlock; irow < std::min(M,iBlock+realBS)-1; irow++) {
			//length of P is M, lda, or number of rows of the large matrix
			//iBlock or realBS? 192-205 not finished
			//P[irow] = iBlock + irow;
			P[irow] += iBlock;
			if (P[irow] != irow) {
				cublasSswap(cublasHandle,iBlock, A+irow , lda, A+ P[irow], lda);
				cublasSswap(cublasHandle,N-iBlock-realBS, d_A+irow+(iBlock+realBS)*lda , lda, d_A+ P[irow]+(iBlock+realBS)*lda, lda);
			}
		}
		*/
		
		cublasStrsm(cublasHandle, CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_UNIT, realBS, N-iBlock-realBS, &plusOne, d_A +iBlock +iBlock*lda, lda, d_A +iBlock + (iBlock+realBS)*lda, lda);

		if (iBlock+realBS < M){
			cublasSgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N, M-iBlock-realBS, N-iBlock-realBS, realBS, &minusOne, d_A+iBlock+realBS+iBlock*lda,lda, d_A+iBlock+(realBS+iBlock)*lda,lda, &plusOne, d_A+iBlock+realBS+(realBS+iBlock)*lda,lda);
		}
	}

	
	// copy memory from device to host
	cudaMemcpy(A, d_A, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	cublasDestroy(cublasHandle);
	cudaFree(d_A);
	cudaDeviceReset();
	return true;
}
	

int main(int argc, char** argv) {
	/*
	int M = 5;
	int N = 5;
	float* A = (float *)malloc(sizeof(float)*M*N);
	A[0] = 8.0; A[1] = 4.0; A[2] = 6.0; A[3] = 2.0; A[4] = 9.0; 
	A[5] = 7.0; A[6] = 9.0; A[7] = 4.0; A[8] = 9.0; A[9] = 4.0;
	A[10] = 3.0; A[11] = 7.0; A[12] = 2.0; A[13] = 6.0; A[14] = 9.0;
	A[15] = 2.0; A[16] = 3.0; A[17] = 7.0; A[18] = 5.0; A[19] = 4.0;
	A[20] = 5.0; A[21] = 8.0; A[22] = 2.0; A[23] = 9.0; A[24] = 7.0;	
	*/
	int M = (argc > 1) ? atoi(argv[1]):3;
	int blockSize = (argc > 1) ? atoi(argv[2]):3;
	int N = M;
	float* A = (float *)malloc(sizeof(float)*M*N);
	
	/*
	for (int r = 0; r < M; r++) {
		for (int c= 0; c < N; c++) {
			std::cout << A[r + c*M] << " ";
		}
		std::cout << std::endl;
	}
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
	struct timespec cudalustart = {0,0}; 
	struct timespec cudaluend = {0,0};
	clock_gettime(CLOCK_REALTIME,&cudalustart);
	//DecomposeLU(M, N, lda, A, P, epsilon, stat);
	DecomposeBlockedLU(M, N, lda, A, P, blockSize, epsilon, stat);
	clock_gettime(CLOCK_REALTIME,&cudaluend);
	std::cout<<"Time of Blocked CUDA LU is "<<(cudaluend.tv_sec-cudalustart.tv_sec)*1000+(cudaluend.tv_nsec-cudalustart.tv_nsec)/1000000<<"ms\n";
	
	std::cout <<"stat: "<<stat._str<<std::endl;
	/*
	for (int r = 0; r < M; r++) {
		for (int c= 0; c < N; c++) {
			std::cout << A[r + c*M] << " ";
		}
		std::cout << std::endl;
	}
	*/
}

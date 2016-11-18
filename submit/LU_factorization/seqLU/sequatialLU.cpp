// See explanation at http://www.sci.utah.edu/~wallstedt
// To compile on Stampede: g++ -O3 -lrt sequatialLU.cpp

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <time.h>

using namespace std;

// These functions print matrices and vectors in a nice format
void coutMatrix(int d,double*m){
	cout<<'\n';
	for(int i=0;i<d;++i){
		for(int j=0;j<d;++j)cout<<setw(14)<<m[i*d+j];
		cout<<'\n';
	}
}

void coutVector(int d,double*v){
	cout<<'\n';
	for(int j=0;j<d;++j)cout<<setw(14)<<v[j];
	cout<<'\n';
}

// The following compact LU factorization schemes are described
// in Dahlquist, Bjorck, Anderson 1974 "Numerical Methods".
//
// S and D are d by d matrices.  However, they are stored in
// memory as 1D arrays of length d*d.  Array indices are in
// the C style such that the first element of array A is A[0]
// and the last element is A[d*d-1].
//
// These routines are written with separate source S and
// destination D matrices so the source matrix can be retained
// if desired.  However, the compact schemes were designed to
// perform in-place computations to save memory.  In
// other words, S and D can be the SAME matrix.  Just
// modify the code like this example:
//
//    void Crout(int d,double*S){
//       for(int k=0;k<d;++k){
//          for(int i=k;i<d;++i){
//             double sum=0.;
//             for(int p=0;p<k;++p)sum+=S[i*d+p]*S[p*d+k];
//             S[i*d+k]=S[i*d+k]-sum; // not dividing by diagonals
//          }
//          for(int j=k+1;j<d;++j){
//             double sum=0.;
//             for(int p=0;p<k;++p)sum+=S[k*d+p]*S[p*d+j];
//             S[k*d+j]=(S[k*d+j]-sum)/S[k*d+k];
//          }
//       }
//    }



// Crout uses unit diagonals for the upper triangle
void Crout(int d,double*S,double*D){
   for(int k=0;k<d;++k){
      for(int i=k;i<d;++i){
         double sum=0.;
         for(int p=0;p<k;++p)sum+=D[i*d+p]*D[p*d+k];
         D[i*d+k]=S[i*d+k]-sum; // not dividing by diagonals
      }
      for(int j=k+1;j<d;++j){
         double sum=0.;
         for(int p=0;p<k;++p)sum+=D[k*d+p]*D[p*d+j];
         D[k*d+j]=(S[k*d+j]-sum)/D[k*d+k];
      }
   }
}
void solveCrout(int d,double*LU,double*b,double*x){
   double y[d];
   for(int i=0;i<d;++i){
      double sum=0.;
      for(int k=0;k<i;++k)sum+=LU[i*d+k]*y[k];
      y[i]=(b[i]-sum)/LU[i*d+i];
   }
   for(int i=d-1;i>=0;--i){
      double sum=0.;
      for(int k=i+1;k<d;++k)sum+=LU[i*d+k]*x[k];
      x[i]=(y[i]-sum); // not dividing by diagonals
   }
}



// Doolittle uses unit diagonals for the lower triangle
void Doolittle(int d,double*S,double*D){
   for(int k=0;k<d;++k){
      for(int j=k;j<d;++j){
         double sum=0.;
         for(int p=0;p<k;++p)sum+=D[k*d+p]*D[p*d+j];
         D[k*d+j]=(S[k*d+j]-sum); // not dividing by diagonals
      }
      for(int i=k+1;i<d;++i){
         double sum=0.;
         for(int p=0;p<k;++p)sum+=D[i*d+p]*D[p*d+k];
         D[i*d+k]=(S[i*d+k]-sum)/D[k*d+k];
      }
   }
}
void solveDoolittle(int d,double*LU,double*b,double*x){
   double y[d];
   for(int i=0;i<d;++i){
      double sum=0.;
      for(int k=0;k<i;++k)sum+=LU[i*d+k]*y[k];
      y[i]=(b[i]-sum); // not dividing by diagonals
   }
   for(int i=d-1;i>=0;--i){
      double sum=0.;
      for(int k=i+1;k<d;++k)sum+=LU[i*d+k]*x[k];
      x[i]=(y[i]-sum)/LU[i*d+i];
   }
}



// Cholesky requires the matrix to be symmetric positive-definite
void Cholesky(int d,double*S,double*D){
   for(int k=0;k<d;++k){
      double sum=0.;
      for(int p=0;p<k;++p)sum+=D[k*d+p]*D[k*d+p];
      D[k*d+k]=sqrt(S[k*d+k]-sum);
      for(int i=k+1;i<d;++i){
         double sum=0.;
         for(int p=0;p<k;++p)sum+=D[i*d+p]*D[k*d+p];
         D[i*d+k]=(S[i*d+k]-sum)/D[k*d+k];
      }
   }
}
// This version could be more efficient on some architectures
// Use solveCholesky for both Cholesky decompositions
void CholeskyRow(int d,double*S,double*D){
   for(int k=0;k<d;++k){
      for(int j=0;j<d;++j){
         double sum=0.;
         for(int p=0;p<j;++p)sum+=D[k*d+p]*D[j*d+p];
         D[k*d+j]=(S[k*d+j]-sum)/D[j*d+j];
      }
      double sum=0.;
      for(int p=0;p<k;++p)sum+=D[k*d+p]*D[k*d+p];
      D[k*d+k]=sqrt(S[k*d+k]-sum);
   }
}
void solveCholesky(int d,double*LU,double*b,double*x){
   double y[d];
   for(int i=0;i<d;++i){
      double sum=0.;
      for(int k=0;k<i;++k)sum+=LU[i*d+k]*y[k];
      y[i]=(b[i]-sum)/LU[i*d+i];
   }
   for(int i=d-1;i>=0;--i){
      double sum=0.;
      for(int k=i+1;k<d;++k)sum+=LU[k*d+i]*x[k];
      x[i]=(y[i]-sum)/LU[i*d+i];
   }
}



int main(int argc, char** argv){
   // the following checks are published at:
   // http://math.fullerton.edu/mathews/n2003/CholeskyMod.html

   // We want to solve the system Ax=b
   
   /*
   double A[25]={8.,7.,3.,2.,5.,
                 4.,9.,7.,3.,8.,
                 6.,4.,2.,7.,2.,
                 2.,9.,6.,5.,9.,
                 9.,4.,9.,4.,7.};
   
   double LU[25];
   
   double b[5]={-2.,4.,3.,-5.,1.};

   double x[5];
   */
   
   int M = (argc > 1) ? atoi(argv[1]):3;
	int N = M;
	double* A = (double *)malloc(sizeof(double)*M*N);
	double* LU = (double *)malloc(sizeof(double)*M*N);	
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
	
	struct timespec cudalustart = {0,0};
	struct timespec cudaluend = {0,0};
	clock_gettime(CLOCK_REALTIME,&cudalustart);
   Crout(M,A,LU);
	clock_gettime(CLOCK_REALTIME,&cudaluend);
   std::cout<<"Time of sequential LU is "<<(cudaluend.tv_sec-cudalustart.tv_sec)*1000+(cudaluend.tv_nsec-cudalustart.tv_nsec)/1000000<<"ms\n";
   
   /*
   cout<<"Crout";
   coutMatrix(5,LU);
   solveCrout(5,LU,b,x);
   cout<<"solveCrout";
   coutVector(5,x);
   
   Doolittle(5,A,LU);
   cout<<"Doolittle";
   coutMatrix(5,LU);
   solveDoolittle(5,LU,b,x);
   cout<<"solveDoolittle";
   coutVector(5,x);

   Cholesky(5,A,LU);
   cout<<"Cholesky";
   coutMatrix(5,LU);
   solveCholesky(5,LU,b,x);
   cout<<"solveCholesky";
   coutVector(5,x);
   */
}

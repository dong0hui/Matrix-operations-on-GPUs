Test Files:
	test.txt contains matrix of 1000x1000
	test10000.txt contains matrix of 10000x10000

CUDA Files:
	cudalu.cu

Out Files:
	cudaluWoutput.out: will output LU matrix, take test.txt as input
	cudalu.out:won't output LU matrix, but only runtime. Take test.txt as input
	cudalu10000.out: only output runtime, take test10000.txt as input

Job Files:
	cudalu: run cudalu10000.out with matrix sizes of 5/100/500/1000/2000/5000/10000
	cudaluWoutput: run cudaluWoutput.out with matrix sizes of 5/100/500/1000

Result Files:
	cudaLuRuntime: output of cudalu
	cudaLuRuntimeWoutput: output of cudaluWoutput


Compile:
	nvcc -arch=compute_35 -code=sm_35 -lcublas -o outfile.out cudafile.cu	
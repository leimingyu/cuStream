#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "math.h"
#include "cublas_v2.h"

// ref: https://www.olcf.ornl.gov/tutorials/concurrent-kernels-ii-batched-library-calls/

int main(int argc, char **argv)
{
	const int streamsNum = 2;
	int N=1<<10; // 1K x 1K matrix

	if (argc == 2) {
		N = atoi(argv[1]);
	}

	if (argc > 2) {
		fprintf(stderr, "Too many arguments! ./stream_sync N .\n");
		exit(1);
	}


	std::cout << "Running " << N << " (floats) as the input data size." << std::endl;
	std::cout << "Launching " << streamsNum << " cuda streams." << std::endl;
	
	// host 
	float *h_a = NULL;
	float *h_b = NULL;
	float *h_c = NULL;
	cudaMallocHost((void**)&h_a, sizeof(float) * N * N); // N x N
	cudaMallocHost((void**)&h_b, sizeof(float) * N * N);
	cudaMallocHost((void**)&h_c, sizeof(float) * N * N);

	float *h_x = NULL;
	float *h_y = NULL;
	float *h_z = NULL;
	cudaMallocHost((void**)&h_x, sizeof(float) * N * N); // N x N
	cudaMallocHost((void**)&h_y, sizeof(float) * N * N);
	cudaMallocHost((void**)&h_z, sizeof(float) * N * N);

	// init 
	for(int i=0; i<N; i++) {
		for(int j=0; j<N; j++) {
			int lid = i * N + j;
			if(i == j) {
				h_a[lid] = sin(lid);	
				h_b[lid] = sin(lid);	
				h_c[lid] = cos(lid) * cos(lid);	

				h_x[lid] = sin(lid);	
				h_y[lid] = sin(lid);	
				h_z[lid] = cos(lid) * cos(lid);	

			}
			else{
				h_a[lid] = 0.; 
				h_b[lid] = 0.; 
				h_c[lid] = 0.; 

				h_x[lid] = 0.; 
				h_y[lid] = 0.; 
				h_z[lid] = 0.; 
			}
		}
	}

	// device 
	float*d_a = NULL; 
	float*d_b = NULL; 
	float*d_c = NULL; 
	cudaMallocHost((void**)&d_a, sizeof(float) * N * N);
	cudaMallocHost((void**)&d_b, sizeof(float) * N * N);
	cudaMallocHost((void**)&d_c, sizeof(float) * N * N);

	float*d_x = NULL; 
	float*d_y = NULL; 
	float*d_z = NULL; 
	cudaMallocHost((void**)&d_x, sizeof(float) * N * N);
	cudaMallocHost((void**)&d_y, sizeof(float) * N * N);
	cudaMallocHost((void**)&d_z, sizeof(float) * N * N);

	// streams
	cudaStream_t streams[streamsNum];
	for(int i=0; i<streamsNum; i++) {
		cudaStreamCreate(&streams[i]);
	}

	float alpha = 1.;
	float beta  = 1.;

    // cublas 
    cublasHandle_t handle0;
    cublasCreate(&handle0);

    cublasHandle_t handle1;
    cublasCreate(&handle1);

	// set matrices on device
	cublasSetMatrixAsync(N, N, sizeof(float), h_a, N, d_a, N, streams[0]);
	cublasSetMatrixAsync(N, N, sizeof(float), h_b, N, d_b, N, streams[0]);
	cublasSetMatrixAsync(N, N, sizeof(float), h_c, N, d_c, N, streams[0]);

	cublasSetMatrixAsync(N, N, sizeof(float), h_x, N, d_x, N, streams[1]);
	cublasSetMatrixAsync(N, N, sizeof(float), h_y, N, d_y, N, streams[1]);
	cublasSetMatrixAsync(N, N, sizeof(float), h_z, N, d_z, N, streams[1]);

	// sgemm on streams
    // SGEMM: C = alpha*A*B + beta*C
	cublasSetStream(handle0, streams[0]);
	cublasSgemm(handle0, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_a, N,
                    d_b, N,
                    &beta,
                    d_c, N);

	cublasSetStream(handle1, streams[1]);
	cublasSgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_x, N,
                    d_y, N,
                    &beta,
                    d_z, N);

	cublasGetMatrixAsync(N, N, sizeof(float), d_c, N, h_c, N, streams[0]);
	cublasGetMatrixAsync(N, N, sizeof(float), d_z, N, h_z, N, streams[1]);
	
	cudaDeviceSynchronize(); // NOTE: this is needed to make sure prev dev opt is done! 

	// free
	for(int i=0; i<streamsNum; i++) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	//free(h_a);
	//free(h_b);

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	cudaFreeHost(h_x);
	cudaFreeHost(h_y);
	cudaFreeHost(h_z);

	return 0;
}

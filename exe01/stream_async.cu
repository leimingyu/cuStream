#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void testKernel(float*x, int len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < len) {
		float sum = x[tid];
		int iter = 0;

		while(iter++ < len) {
			sum += 1;
		}
		x[tid] = sum;
	}

}

int main(int argc, char **argv)
{
	const int streamsNum = 2;
	int N=1<<10; // 1K

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
	cudaMallocHost((void**)&h_a, sizeof(float) * N);
	cudaMallocHost((void**)&h_b, sizeof(float) * N);

	// init 
	for(int i=0; i<N; i++) {
		h_a[i] = 0;
		h_b[i] = 0;
	}

	// device 
	float*d_a = NULL; 
	float*d_b = NULL; 
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);

	// streams
	cudaStream_t streams[streamsNum];
	for(int i=0; i<streamsNum; i++) {
		cudaStreamCreate(&streams[i]);
	}

	// h2d
	cudaMemcpyAsync(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(d_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice, streams[1]);

	// kernel
	dim3 block = dim3(128,1,1);
	dim3 grid = dim3((N + block.x - 1) / block.x,1,1);

	testKernel <<< grid, block, 0, streams[0] >>> (d_a, N);
	testKernel <<< grid, block, 0, streams[1] >>> (d_b, N);

	// d2h
	cudaMemcpyAsync(h_a, d_a, sizeof(float)*N, cudaMemcpyDeviceToHost, streams[0]);
	cudaMemcpyAsync(h_b, d_b, sizeof(float)*N, cudaMemcpyDeviceToHost, streams[1]);

	cudaDeviceSynchronize(); // NOTE: this is needed to make sure prev dev opt is done! 

	int error_a = 0; 
	for(int i=0; i<N; i++) {
		if(h_a[i] != N) {
			printf("h_a[%d] = %f\n",i, h_a[i]);
			error_a += 1;
		}	
	}
	if(error_a == 0) {
		printf("Pass test on h_a!\n");
	}

	int error_b = 0; 
	for(int i=0; i<N; i++) {
		if(h_b[i] != N) {
			printf("h_b[%d] = %f\n",i, h_b[i]);
			error_b += 1;
		}	
	}
	if(error_b == 0) {
		printf("Pass test on h_b!\n");
	}


	// free
	for(int i=0; i<streamsNum; i++) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);

	return 0;
}

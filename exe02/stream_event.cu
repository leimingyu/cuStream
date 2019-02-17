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


__global__ void subKernel (float*a, float*b, float*c, int len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < len) {
		c[tid] = a[tid] - b[tid];	
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
	/*
	// pageable
	float *h_a = (float*)malloc(sizeof(float) * N);
	float *h_b = (float*)malloc(sizeof(float) * N);
	*/


	float *h_a = NULL;
	float *h_b = NULL;
	float *h_c = NULL;
	cudaMallocHost((void**)&h_a, sizeof(float) * N);
	cudaMallocHost((void**)&h_b, sizeof(float) * N);
	cudaMallocHost((void**)&h_c, sizeof(float) * N);

	// init 
	for(int i=0; i<N; i++) {
		h_a[i] = 0;
		h_b[i] = 0;
	}

	// device 
	float*d_a = NULL; 
	float*d_b = NULL; 
	float*d_c = NULL; 
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_c, sizeof(float) * N);

	// streams
	cudaStream_t streams[streamsNum];
	cudaEvent_t  events[streamsNum]; // events for streams

	for(int i=0; i<streamsNum; i++) {
		cudaStreamCreate(&streams[i]);
		cudaEventCreate(&events[i]);
	}

	// h2d
	cudaMemcpyAsync(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice, streams[0]);
	cudaMemcpyAsync(d_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice, streams[1]);

	// kernel
	dim3 block = dim3(128,1,1);
	dim3 grid = dim3((N + block.x - 1) / block.x,1,1);

	testKernel <<< grid, block, 0, streams[0] >>> (d_a, N); // a + x
	cudaEventRecord(events[0], streams[0]);
	testKernel <<< grid, block, 0, streams[1] >>> (d_b, N); // b + x
	cudaEventRecord(events[1], streams[1]);

	cudaEventSynchronize(events[0]);
	cudaEventSynchronize(events[1]);

	subKernel <<< grid, block, 0, streams[0] >>> (d_a, d_b, d_c, N); // a - b 

	// d2h
	cudaMemcpyAsync(h_c, d_c, sizeof(float)*N, cudaMemcpyDeviceToHost, streams[0]);

	cudaDeviceSynchronize(); // NOTE: this is needed to make sure prev dev opt is done! 

	int error_c = 0; 
	for(int i=0; i<N; i++) {
		if(h_c[i] > 1e-8) {  // h_c should be 0
			printf("h_c[%d] = %f\n",i, h_c[i]);
			error_c += 1;
		}	
	}
	if(error_c == 0) {
		printf("Pass test on h_c!\n");
	}


	// free
	for(int i=0; i<streamsNum; i++) {
		cudaStreamDestroy(streams[i]);
		cudaEventDestroy(events[i]);
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	return 0;
}

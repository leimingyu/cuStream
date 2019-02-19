#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

typedef struct
{
	int len;
	// host 
	float *h_input, *h_output;
	// device 
	float *d_input, *d_output;
	// cuda stream
	cudaStream_t stream;

} MGPUdata;


__global__ void testKernel(float*x, float*y, int len)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < len) {
		float sum = x[tid];
		int iter = 0;

		while(iter++ < len) {
			sum += 1;
		}
		y[tid] = sum;
	}

}

int main(int argc, char **argv)
{
	int GpuNum = 1;
	cudaGetDeviceCount(&GpuNum);
	printf("CUDA device = %i\n", GpuNum);

	const int N = 100000; // 100K
	MGPUdata mgpu[GpuNum];

	for(int i=0; i<GpuNum; i++) {
		mgpu[i].len	= N;
	}

	// set up for each gpu
	for(int i=0; i<GpuNum; i++) {
		cudaSetDevice(i); // NOTE: needed for each device

		// create streams for each gpu
		cudaStreamCreate(&mgpu[i].stream);
		// host
		cudaMallocHost(&mgpu[i].h_input,  sizeof(float) * mgpu[i].len);
		cudaMallocHost(&mgpu[i].h_output, sizeof(float) * mgpu[i].len);
		// device
		cudaMalloc(&mgpu[i].d_input,  sizeof(float) * mgpu[i].len);
		cudaMalloc(&mgpu[i].d_output, sizeof(float) * mgpu[i].len);
		// initialize
		for (int j=0; j<mgpu[i].len; j++)
		{
			mgpu[i].h_input[j] = 0.; 
		}

	}

	dim3 block = dim3(128, 1, 1);
	dim3 grid  = dim3((N + block.x - 1)/ block.x, 1, 1);

	// data transfer  + kernel
	for(int i=0; i<GpuNum; i++)
	{
		cudaSetDevice(i);
		// h2d
		cudaMemcpyAsync(mgpu[i].d_input, mgpu[i].h_input, mgpu[i].len * sizeof(float), cudaMemcpyHostToDevice, mgpu[i].stream);
		// kernel
		testKernel <<< grid, block, 0, mgpu[i].stream >>>(mgpu[i].d_input, mgpu[i].d_output, mgpu[i].len);
		// d2h
		cudaMemcpyAsync(mgpu[i].h_output, mgpu[i].d_output,mgpu[i].len * sizeof(float), cudaMemcpyDeviceToHost, mgpu[i].stream);
	}



	// cleanup
	for(int i=0; i<GpuNum; i++) {
		cudaSetDevice(i);
		cudaStreamSynchronize(mgpu[i].stream);

		cudaFreeHost(mgpu[i].h_input);
		cudaFreeHost(mgpu[i].h_output);
		cudaFree(mgpu[i].d_input);
		cudaFree(mgpu[i].d_output);
		cudaStreamDestroy(mgpu[i].stream);
	}


	return 0;
}

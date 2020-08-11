#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "helper_cuda.h"

class Dog 
{
public:
	Dog();
	~Dog();
	cudaStream_t *cuStreams;
	void prepare(int N);
private:
	int streamNum;
};


Dog::Dog() : streamNum(1), cuStreams(NULL)
{
}

Dog::~Dog()
{
	if(cuStreams) {
		for(int i=0; i<streamNum; i++) {
			checkCudaErrors(cudaStreamDestroy(cuStreams[i]));	
		}

		free(cuStreams);
	}
}

void Dog::prepare(int N)
{
	printf("Dog: creating %d streams\n", N);
	streamNum = N;
	cuStreams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * N);

	for(int i=0; i<streamNum; i++) {
		cudaStreamCreate(&cuStreams[i]);
		printf("Dog: stream = %ld\n", (long)cuStreams[i]);
	}
}

class Cat 
{
public:
	Cat();
	~Cat();
	cudaStream_t *cuStreams;
	void prepare(int N);
private:
	int streamNum;
};


Cat::Cat() : streamNum(1), cuStreams(NULL)
{
}

Cat::~Cat()
{
	if(cuStreams) {
		for(int i=0; i<streamNum; i++) {
			checkCudaErrors(cudaStreamDestroy(cuStreams[i]));	
		}

		free(cuStreams);
	}
}

void Cat::prepare(int N)
{
	printf("Cat: creating %d streams\n", N);
	streamNum = N;
	cuStreams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * N);

	for(int i=0; i<streamNum; i++) {
		cudaStreamCreate(&cuStreams[i]);
		printf("Cat: stream = %ld\n", (long)cuStreams[i]);
	}
}


int main(int argc, char **argv)
{
	Dog d1;
	Cat c1;

	d1.prepare(2);
	c1.prepare(2);

	return 0;
}

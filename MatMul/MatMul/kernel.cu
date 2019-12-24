
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>




#define BLOCK_SIZE 16

static void comp(float* c, float* cc,int N) {
	int counter = 0;
	for(int n=0;n<N;n++)
		for (int m = 0; m < N; m++) {
			//printf("%f %f\n",c[m + n * N], cc[m + n * N]);
			if (c[m + n * N] == cc[m + n * N])
				counter++;
		}
	if (counter == N * N) {
		printf("equal");
	}
	else
	{
		printf("not equal");
	}
}

__global__ void kernel(float *a,float *b, int n, float *c)
{
	int   bx = blockIdx.x;    
	int   by = blockIdx.y;
	int   tx = threadIdx.x;        
	int   ty = threadIdx.y;
	float sum = 0.0f;      
	int   ia = n * BLOCK_SIZE * by + n * ty;   // a [i][0]
	int   ib = BLOCK_SIZE * bx + tx;


	for (int k = 0; k < n; k++)
		sum += a[ia + k] * b[ib + k * n];

	int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	c[ic + n * ty + tx] = sum;
}



int main()
{
	int N = 2048;
	int m, n, k;

	float timerValueGPU = 0.0f;
	float timerValueCPU = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;

	int numBytes = N * N * sizeof(float);
	float* adev, * bdev, * cdev, * a, * b, * c, * cc;

	a = (float*)malloc(numBytes);
	b = (float*)malloc(numBytes);
	c = (float*)malloc(numBytes);
	cc = (float*)malloc(numBytes);

	for (n = 0; n < N; n++)
		for (m = 0; m < N; m++) {
			a[m + n * N] = 2.0f * m + n;
			b[m + n * N] = 2.0f * n;
		}

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);

	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);

	double avGPUTime = 0;
	for (int t = 0; t < 10; t++) {
		cudaEventRecord(start, 0);

		cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

		kernel << < blocks, threads >> > (adev, bdev, N, cdev);
		err = cudaPeekAtLastError();
		if (err != cudaSuccess)
			printf(cudaGetErrorString(err));
		cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timerValueGPU, start, stop);
		avGPUTime += timerValueGPU;

	}
	avGPUTime = avGPUTime / 10;
	printf("\n GPU calculation time %f msec\n", avGPUTime);

	


	clock_t startc;
	clock_t stopc;
	double avCPUTime = 0;
	for (int t = 0; t < 10; t++) {
		startc = clock();



		for (n = 0; n < N; n++)
			for (m = 0; m < N; m++) {
				cc[m + n * N] = 0.f;
				for (k = 0; k < N; k++)
					cc[m + n * N] += a[k + n * N] * b[k * N + m];
			}

		stopc = clock();
		avCPUTime += ((double)(stopc - startc)) / ((double)CLOCKS_PER_SEC);
	}
	avCPUTime = avCPUTime / 10;
	printf("\n CPU calculation time %f msec\n", avCPUTime);
	//printf("\n Ctime: %f\n", ((double)(stopc - startc)) / ((double)CLOCKS_PER_SEC));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	comp(c,cc,N);
	float sumc=0, sumcc=0;
	for (int n = 0; n < N; n++)
		for (int m = 0; m < N; m++) {
			//printf("%f %f\n",c[m + n * N], cc[m + n * N]);
			sumc += c[m + n * N];
			sumcc += cc[m + n * N];
		}
	cudaFree(&adev);
	cudaFree(&bdev);
	cudaFree(&cdev);
	printf("sumc = %g", sumc);
		printf( " sumcc = %g", sumcc);
		if (sumc == sumcc) printf("matrix are equal");
	delete a;
	delete b;
	delete c;
	delete cc;

	return 0;
}

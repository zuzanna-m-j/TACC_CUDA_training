#include <stdio.h>
#include <random>
#include <iostream>
#include <fstream>
using namespace std;

__global__ void stridedMemAccess(float *d_in, int stride){

    int tid = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
    d_in[tid] = d_in[tid] + stride;
}


int main(){

    ofstream outfile("bandwidth.data");

    int blockSize = 256;
    int gridSize = 1024 * 5;

    int N = gridSize * blockSize;

    printf("Processing %d fp32 elements\n", N);
    int total_size = N * 33;

    float *d_in;
    float gpu_time;

    cudaEvent_t start, stop;  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t err;


    cudaMalloc(&d_in, total_size * sizeof(float));
    float *h_in = (float *)malloc(total_size * sizeof(float));

    for (int i = 0; i < N * 33; i++) {
        h_in[i] = 1.0;
    }

    //kernel warm up
    cudaMemcpy(d_in, h_in, total_size*sizeof(float), cudaMemcpyHostToDevice);
    stridedMemAccess<<<gridSize, blockSize>>>(d_in,0);


    for (int stride = 1; stride <= 32; stride++) {

        cudaMemcpy(d_in, h_in, total_size*sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(start);

        stridedMemAccess<<<gridSize, blockSize>>>(d_in,stride);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time, start, stop);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        printf("Stride: %d, Bandwidth (GB/s) %f\n", stride,  N*4*2/gpu_time/1e6);
        outfile << stride << " " << N*4*2/gpu_time/1e6 << endl;
        

    }
    cudaFree(d_in);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    outfile.close();
    return 0;

}

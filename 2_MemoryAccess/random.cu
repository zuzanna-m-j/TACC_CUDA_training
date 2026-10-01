#include <stdio.h>
#include <random>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include <random>      
#include <chrono>       

using namespace std;

// input arrays d_in and d_rng have size N = gridSize * blockSize = total_num_threads
// d_in has input on which thread will perform addition operation
// d_rng contains random integers in the range [0,N-1] --> all possible indexes of d_in shuffled in random order

__global__ void randomMemAccess(float *d_in, int *d_rng){

    // assign one random number from d_rng to each thread
    // each thread operates on one element of d_in[n], index "n" determined by its assigned random number
    d_in[n] = d_in[n] + 1.0;
}


int main(){


    int blockSize = 256;
    int gridSize = 1024 * 5;

    int N = gridSize * blockSize;

    printf("Processing %d fp32 elements\n", N);

    float *d_in;
    int *d_rng;
    float gpu_time = 0;

    cudaEvent_t start, stop;  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t err;


    cudaMalloc(&d_in, N * sizeof(float));

    float *h_in = (float *)malloc(N * sizeof(float));

    // Random number generator

    int *h_rng = (int *)malloc(N * sizeof(int));
    cudaMalloc(&d_rng, N * sizeof(int));

    for (int i = 0; i < N; i++){
        h_rng[i] = i;
    }

    unsigned seed = chrono::system_clock::now().time_since_epoch().count(); //time-based seed
    shuffle(h_rng, h_rng + N, default_random_engine(seed));

    cout << "First 20 random numbers: " << endl;
    for (int i = 0; i < 19; i++){
        cout << h_rng[i] <<", ";
    }
    cout << h_rng[19] << endl;

    cudaMemcpy(d_rng, h_rng, N*sizeof(int), cudaMemcpyHostToDevice);

    // End random rumber generator

    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0;
    }

    // Warmp-up
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    randomMemAccess<<<gridSize, blockSize>>>(d_in,d_rng);


    int n_iter = 100;

    for (int i = 0; i < n_iter; i++){
        float t_gpu_time;
        cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);

        randomMemAccess<<<gridSize, blockSize>>>(d_in, d_rng);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_gpu_time, start, stop);
        gpu_time += t_gpu_time;

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }
    gpu_time = gpu_time/n_iter;
    printf("Random access bandwidth (GB/s) %f\n",   N*4*3/gpu_time/1e6);


    cudaFree(d_in);
    cudaFree(d_rng);
    free(h_in);
    free(h_rng);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;

}

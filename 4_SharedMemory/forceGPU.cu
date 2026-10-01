#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>


#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>

__global__ void ForceGPUShared(float *d_force, float *d_pos, int N, int spacer)
{
  extern __shared__ float shared_pos[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    shared_pos[tid] = d_pos[gid * spacer];
    d_force[gid] = 0.0;
    float my_pos = d_pos[gid * spacer];

    __syncthreads();

    for(int m = 0; m < 3; m++)
        d_force[gid] += (shared_pos[(tid+m)%N] - my_pos) + (shared_pos[(N + tid-m)%N] - my_pos);

}

__global__ void ForceGPU(float *d_force, float *d_pos, int N, int spacer)
{

    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x;
    d_force[tid + offset] = 0.0;
    float my_pos = d_pos[spacer*(tid + offset)];
    for(int m = 0; m < 3; m++)
        d_force[tid + offset] += (d_pos[spacer*((tid+m)%N + offset)] - my_pos) + (d_pos[spacer*((N + tid-m)%N + offset)] - my_pos);

}


void ForceCPU(float *force, float *pos, int N, int len, int spacer){

    for (int i = 0; i < N; i++){
	    force[i]=0.0;
        float my_pos = pos[i*spacer];
        int block_id = floor(i/len);
        int inx = i%len;
        int offset = len * block_id;
        for(int m = 0; m < 3; m++)
            force[i] += (pos[spacer*((inx+m)%len + offset)] - my_pos) + (pos[spacer*((len+inx-m)%len + offset)] - my_pos);

    }
}


int main(void)
{

    int gridSize = 200;
    int blockSize = 1024;
    int N = blockSize * gridSize;

    int gpu_iter = 4000;
    int cpu_iter = 1000;

    int spacer = 64;

    float *force = (float *)malloc(sizeof(float) * N);
    float *force_cpu = (float *)malloc(sizeof(float) * N);
    float *pos = (float *)malloc(sizeof(float)*spacer * N);
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    float gpu_time = 0.0;
    float gpu_time_t = 0.0;
    float gpu_time_s = 0.0;
    double cpu_time = 0.0;
    double cpu_time_t = 0.0;

    cudaError_t err;

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0,0.2);

    float *d_pos, *d_force;

    cudaMalloc(&d_pos, spacer * N * sizeof(float)); 
    cudaMalloc(&d_force, N * sizeof(float)); 

    // ==== SHARED MEMORY === //

    for (int i = 0; i < N*spacer; i++) {
        pos[i] = float(i) + distribution(generator);
    }
    cudaMemcpy(d_pos, pos, spacer * N*sizeof(float), cudaMemcpyHostToDevice);


    //kernel warm-up
    cudaMemcpy(d_pos, pos,spacer* N*sizeof(float), cudaMemcpyHostToDevice);
    ForceGPU<<<1, 1>>>(d_force, d_pos, blockSize, spacer);
    cudaMemcpy(force, d_force, N*sizeof(float), cudaMemcpyDeviceToHost);


    // time GPU kernel
    for (int j = 0; j < gpu_iter; j++){

        gpu_time_t = 0.0;

        cudaEventRecord(start);
        ForceGPU<<<gridSize, blockSize>>>(d_force, d_pos, blockSize, spacer);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_t, start, stop);
        gpu_time += gpu_time_t;
    }
    cudaMemcpy(force, d_force, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // check result
    ForceCPU(force_cpu, pos, N, blockSize, spacer);
        
    for (int i = 0; i < N; i++){
        if (force_cpu[i] != force[i])
            printf("Error: CPU[%d]!=GPU[%d] (%f, %f)\n", i, i, force_cpu[i], force[i]);
    }

    // time GPU kernel with shared memory
    for (int j = 0; j < gpu_iter; j++){

        gpu_time_t = 0.0;

        cudaEventRecord(start);
        ForceGPUShared<<<gridSize, blockSize,blockSize*sizeof(float)>>>(d_force, d_pos, blockSize, spacer);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_t, start, stop);
        gpu_time_s += gpu_time_t;
    }
    cudaMemcpy(force, d_force, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
        
    for (int i = 0; i < N; i++){
        if (force_cpu[i] != force[i])
            printf("Error: CPU[%d]!=GPU[%d] (%f, %f)\n", i, i, force_cpu[i], force[i]);
    }


    // ==== CPU === //

    // Time the CPU execution time

    for (int j = 0; j < cpu_iter; j++){

        auto start_cpu = std::chrono::high_resolution_clock::now();
        ForceCPU(force_cpu, pos, N, blockSize, spacer);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        cpu_time_t = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
        cpu_time += cpu_time_t;
    }
        
    cpu_time *= 1e-6; //convert to milliseconds



    printf("=== Performance Statistics ===\n");
    printf("Spacer:%d\n", spacer);
    printf("CPU time: %.6f ms\n", cpu_time/float(cpu_iter));
    printf("GPU time: %.6f ms\n", gpu_time/float(gpu_iter));
    printf("GPU time with shared memory: %.6f ms\n", gpu_time_s/float(gpu_iter));


    // free the memory

    cudaFree(d_force);
    cudaFree(d_pos);
    free(force);
    free(force_cpu);
    free(pos);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;

}

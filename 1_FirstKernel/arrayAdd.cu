#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to square each element of an array
__global__ void arrayAdd(float *d_a, float *d_b, float *d_out, int size)
{
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid < size)
    {
        d_out[tid] = d_a[tid] + d_b[tid];
    }
}

int main()
{
    // Array size
    const int N = 1<<25;
    size_t bytes = N * sizeof(float);

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_time = 0.0f;
        
    cudaError_t err;

    // Host arrays
    float *h_in = (float*)malloc(bytes);
    float *h_in2 = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    
    // Initialize input array on host
    for (int i = 0; i < N; i++)
    {
        h_in[i] = i;
        h_in2[i] = 2*i;
    }
    
    // Device arrays
    float *d_out, *d_a, *d_b;
    
    // Allocate memory on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    
    // Copy input array from host to device
  
    
    cudaEventRecord(start);

    cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Copy H->D took %.6f ms\n", gpu_time);

    cudaMemcpy(d_b, h_in2, bytes, cudaMemcpyHostToDevice);
    

    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = ceil(float(N)/threadsPerBlock);
    
    // Launch the kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEventRecord(start);
    arrayAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

        
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Kernel warm up time: %.6f ms\n", gpu_time);


    cudaEventRecord(start);
    arrayAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

        
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Matrix addition took %.6f ms\n", gpu_time);


    printf("Effective Bandwidth (GB/s): %.0f\n", N*4*3/gpu_time/1e6);
    

    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaEventRecord(start);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Copy D->H took %.6f ms\n", gpu_time);


    // Verify results
    for (int i = 0; i < N; i++)
    {
        if (h_out[i] != h_in[i] + h_in2[i])
        {
            printf("Verification failed at index %d: expected %d, got %d\n", 
                   i, h_in[i] + h_in2[i], h_out[i]);
            break;
        }
    }

    // Free device memory
    cudaFree(d_b);
    cudaFree(d_a);
    cudaFree(d_out);
    
    // Free host memory
    free(h_in);
    free(h_out);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

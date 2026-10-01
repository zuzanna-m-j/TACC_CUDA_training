#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>

__global__ void kernel(int *in_data, int *out_data, int *out_tid, int N)
{   
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Finish the kernel

    
}


int main()
{
    // Array size
    const int N = 96;
    size_t bytes = N * sizeof(int);

    // Timing variables
        
    cudaError_t err;

    // Host arrays
    int *in_data = (int*)malloc(2*bytes);
    int *out_data = (int*)malloc(bytes);
    int *out_tid = (int*)malloc(bytes);
    
    // Initialize input array on host
    for (int i = 0; i < 2*N; i++)
    {
        in_data[i] = i;
    }
    
    // Device arrays
    int *d_in_data, *d_out_data, *d_out_tid;
    
    // Allocate memory on the device
    cudaMalloc(&d_in_data, 2* bytes);
    cudaMalloc(&d_out_data, bytes);
    cudaMalloc(&d_out_tid, bytes);
    
    // Copy input array from host to device

    cudaMemcpy(d_in_data, in_data, 2*bytes, cudaMemcpyHostToDevice);    

    // Set up execution configuration
    int threadsPerBlock = 32;
    int blocksPerGrid = 1;
    
    // Launch the kernel
    printf("CUDA kernel launch with num_blocks = %d and num_threads = %d \n", blocksPerGrid, threadsPerBlock);

    //kernel warm-up
    kernel<<<1, 1>>>(d_in_data,d_out_data,d_out_tid, N);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in_data,d_out_data,d_out_tid, N);
        
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaMemcpy(out_data, d_out_data, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_tid, d_out_tid, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    int correct = 1;
    for (int i = 0; i < N; i++){
        if(out_data[i] !=  i*4+1 || out_tid[i] != i%32){
            printf("Error: Your results are out_data[%d] = %d, processed by thread %d || Correct result: out_data[%d] =  %d, processed by thread %d\n",i, out_data[i],out_tid[i],i,i*4+1,i%32); 
            correct = 0;
        }
    }
    if (correct == 1)
        printf("Results are correct!\n");


    

    // Free device memory
    cudaFree(d_in_data);
    cudaFree(d_out_data);
    cudaFree(d_out_tid);
    
    // Free host memory
    free(in_data);
    free(out_data);
    free(out_tid);

    

    return 0;
}

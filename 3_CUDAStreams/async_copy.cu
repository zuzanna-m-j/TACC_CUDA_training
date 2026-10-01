#include <stdio.h>
#include <assert.h>

__global__ void cube(float *d_data, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        d_data[i] = pow(float(i),3);
    }
}
int main()
{

    
    const int num_streams = 6;
    const int N = 1 << 20; // bitwise shift to the left, result is 1 * 2^10


    cudaStream_t streams[num_streams];
    float *data[num_streams]; // array of pointers, 2D array
    float *test_data[num_streams]; // array of pointers, 2D array
    float *d_data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_data[i], N * sizeof(float));
        cudaMallocHost(&data[i], N * sizeof(float));
        test_data[i] = (float*)malloc(N * sizeof(float));
    }

    for (int i = 0; i < num_streams; i++) {
        for (int j = 0; j < N; j++) {
            data[i][j] = -2.0;
            test_data[i][j] = pow(float(j),3);
        }
    }


    for (int i = 0; i < num_streams; i++) { 
        cudaMemcpyAsync(d_data[i], data[i], N*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cube<<<4, 128, 0, streams[i]>>>(d_data[i], N);
        cudaMemcpyAsync(data[i], d_data[i], N*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();


    for (int i = 0; i < num_streams; i++) {
        for (int j = 0; j < 5; j++) {
            assert(data[i][j] == test_data[i][j] && "Calculation error - array elements are not the same!");
        }
    }

    // Clean-up

    for (int i = 0; i < num_streams; i++) { 
        cudaStreamDestroy(streams[i]);
        cudaFreeHost(data[i]);
        cudaFree(d_data[i]);
        free(test_data[i]);
    }
    
    return 0;
}

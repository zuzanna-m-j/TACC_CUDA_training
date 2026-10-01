#include <stdio.h>

__global__ void cube(float *d_data, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        d_data[i] = pow(float(i),3);
    }
}
int main()
{
    const int num_streams = 12;
    const int N = 1 << 30; // bitwise shift to the left, result is 1 * 2^30

    cudaStream_t streams[num_streams];
    float *d_data[num_streams]; // array of pointers, 2D array

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_data[i], N * sizeof(float));
    }

    for (int i = 0; i < num_streams; i++) { 
        cube<<<128, 128, 0, streams[i]>>>(d_data[i], N);

        //uncomment below to add a dummy kernel on the default stream
        // cube<<<1, 1>>>(0, 0);
    }

    // Wait for all streams to finish

    cudaDeviceSynchronize();

    // Clean-up

    for (int i = 0; i < num_streams; i++) { 
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

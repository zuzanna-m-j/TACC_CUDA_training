#define FULL_MASK 0xffffffff
#include <stdio.h>

__device__ void IntToBin(unsigned num) {
  int i;
  printf("Active mask is %u --> ", num);
  for (i = 32 - 1; i >= 0; i--) {
      printf("%d", (num >> i) & 1); 
  }
  printf("\n");
}

__global__ void warp_shfl_down(int* x, int M)
{
    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x%32 < M);
    int val;
    if (threadIdx.x%32 < M){
        val = x[threadIdx.x];
        if (threadIdx.x == 0)
          IntToBin(mask);

        for (int offset = 16; offset > 0; offset /= 2){
          val += __shfl_down_sync(mask, val, offset);

          if (threadIdx.x == 0){
            printf("Lane 0 requested val from lane %d; sum is now %d\n", offset, val);
            if (offset > M)
              printf("Lane %d was not in the active mask!\n",offset);
          }
        }
      }
      __syncthreads();
      if (threadIdx.x%32 == 0)
        printf("From warp %d, sum is %d\n", threadIdx.x/32,val);
}

int main(void)
{
  int N = 1024;
  int *data, *d_data;

  data = (int*)malloc(N*sizeof(int));
  cudaMalloc(&d_data, N*sizeof(int)); 

  for (int i = 0; i < N; i++) {
    data[i] = 1 + int(i/32);
  }

  
  cudaMemcpy(d_data, data, N*sizeof(int), cudaMemcpyHostToDevice);
  warp_shfl_down<<<1,128>>>(d_data,8);



  cudaFree(d_data);
  free(data);

  return 0;
}

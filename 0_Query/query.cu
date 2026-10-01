#include <stdio.h> 

int main() {

  int numOfDevices;

  cudaGetDeviceCount(&numOfDevices);


  printf("\n\n * You have %d CUDA-enabled %s *\n\n",numOfDevices, numOfDevices == 1? "device":"devices");

//enumerate though devices
  for (int i = 0; i < numOfDevices; i++) {

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i); //puts all properties in the properties variable
    printf(" === Information for device number %d ===\n", i);
    //Properties can be accessed by their name
    printf("  Device name: %s\n", properties.name);
    
    printf("  Compute Capability: %d.%d\n", properties.major, properties.minor);
    printf("  Maximum threads per SM:  %d\n", properties.maxThreadsPerMultiProcessor);
    printf("  Maximum threads per block (blockSize): %d\n", properties.maxThreadsPerBlock);
    printf("  Max thread block size in x:%d, y:%d, z:%d \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
    printf("  Max grid size in x:%d, y:%d, z:%d\n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);


    printf("  Total Global Memory: %.0f GB\n", static_cast<float>(properties.totalGlobalMem/(1024.0*1024.0*1024.0)));
    printf("  Peak Memory Bandwidth:  %.0f GB/s\n\n", 2.0*properties.memoryClockRate*(properties.memoryBusWidth/8)/1.0e6);
  }
}

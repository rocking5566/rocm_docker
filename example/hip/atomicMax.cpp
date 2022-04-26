#include "hip/hip_runtime.h"
#include "util.hpp"
#include <stdio.h>
#include <stdlib.h>

__global__ void atomicMax(float* arry, int n, float reg)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n)
        atomicMax(arry + id, reg);
}

int main(int argc, char* argv[])
{
    int n            = 32;
    size_t arry_size = n * sizeof(float);
    float* host_src  = (float*)malloc(arry_size);
    float* host_dst  = (float*)malloc(arry_size);

    float* device_arry;

    HIP_ASSERT(hipMalloc(&device_arry, arry_size));

    for(int i = 0; i < n; i++)
        host_src[i] = i;

    HIP_ASSERT(hipMemcpy(device_arry, host_src, arry_size, hipMemcpyHostToDevice));

    int blockSize = 8;                               // Number of threads in each thread block
    int gridSize  = (int)ceil((float)n / blockSize); // Number of thread blocks in grid

    hipLaunchKernelGGL(atomicMax, dim3(gridSize), dim3(blockSize), 0, 0, device_arry, n, 5);
    hipDeviceSynchronize();
    HIP_ASSERT(hipMemcpy(host_dst, device_arry, arry_size, hipMemcpyDeviceToHost));

    for(int i = 0; i < n; i++)
        printf("host_src[%d] = %f, host_dst[%d] = %f\n", i, host_src[i], i, host_dst[i]);

    HIP_ASSERT(hipFree(device_arry));
    free(host_src);
    free(host_dst);
    return 0;
}

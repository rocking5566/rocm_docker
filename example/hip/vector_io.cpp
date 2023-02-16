#include "hip/hip_runtime.h"
#include "util.hpp"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 512
#define VECTOR_WIDTH 4
using ARRY_TYPE = float;
using VECTOR_TYPE = __attribute__((__ext_vector_type__(VECTOR_WIDTH))) ARRY_TYPE;

template <typename T>
__global__ void copy(const T* __restrict__ src, T* __restrict__ dst, int n)
{
    int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
    if(idx < n)
        dst[idx] = src[idx];
}

int main(int argc, char* argv[])
{
    const int nrepeat = 10;

    int n            = 1502 * 4096 / VECTOR_WIDTH;
    size_t arry_size = n * sizeof(VECTOR_TYPE);

    VECTOR_TYPE* device_src;
    VECTOR_TYPE* device_dst;

    hipMalloc(&device_src, arry_size);
    hipMalloc(&device_dst, arry_size);

    int gridSize = (int)ceil((float)n / BLOCK_SIZE);
    hipEvent_t start, stop;
    float total_time = 0;

    // warmup
    hipLaunchKernelGGL(
        copy<VECTOR_TYPE>, dim3(gridSize), dim3(BLOCK_SIZE), 0, 0, device_src, device_dst, n);

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, nullptr);

    for(int i = 0; i < nrepeat; ++i)
    {
        hipLaunchKernelGGL(
            copy<VECTOR_TYPE>, dim3(gridSize), dim3(BLOCK_SIZE), 0, 0, device_src, device_dst, n);
    }

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_time, start, stop);

    printf("time = %fms, bandwidth = %f GB/S\n",
           total_time / nrepeat,
           ((float)2 * arry_size / 1000000) / (total_time / nrepeat));

    hipFree(device_src);
    hipFree(device_dst);
    return 0;
}

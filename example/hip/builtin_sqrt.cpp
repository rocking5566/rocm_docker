#include "hip/hip_runtime.h"
#include "util.hpp"
#include <stdio.h>
#include <stdlib.h>

using ARRY_TYPE = float;

__global__ void builtin(ARRY_TYPE* arry, int n)
{
    for(int i = 0; i < n; ++i)
    {
        arry[i] = __builtin_amdgcn_sqrt(arry[i]);
        // arry[i] = ::sqrtf(arry[i]);
    }
}

int main(int argc, char* argv[])
{
    int n            = 32;
    size_t arry_size = n * sizeof(ARRY_TYPE);
    ARRY_TYPE* host_src = (ARRY_TYPE*)malloc(arry_size);
    ARRY_TYPE* host_dst = (ARRY_TYPE*)malloc(arry_size);

    ARRY_TYPE* device_arry;

    HIP_ASSERT(hipMalloc(&device_arry, arry_size));

    for(int i = 0; i < n; i++)
        host_src[i] = i;

    HIP_ASSERT(hipMemcpy(device_arry, host_src, arry_size, hipMemcpyHostToDevice));

    int blockSize = 8;                               // Number of threads in each thread block
    int gridSize  = (int)ceil((float)n / blockSize); // Number of thread blocks in grid

    hipLaunchKernelGGL(builtin, dim3(gridSize), dim3(blockSize), 0, 0, device_arry, n);
    hipDeviceSynchronize();
    HIP_ASSERT(hipMemcpy(host_dst, device_arry, arry_size, hipMemcpyDeviceToHost));

    for(int i = 0; i < n; i++)
        printf("host_src[%d] = %f, host_dst[%d] = %f\n", i, host_src[i], i, host_dst[i]);

    HIP_ASSERT(hipFree(device_arry));
    free(host_src);
    free(host_dst);
    return 0;
}

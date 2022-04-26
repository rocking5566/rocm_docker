#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "util.hpp"

__device__ float
llvm_amdgcn_raw_buffer_atomic_add_fp32(float vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

__global__ void atomicAdd(float* arry, int n, float adder)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n)
    {
        int32x4_t dst_wave_buffer_resource = make_wave_buffer_resource(arry, n);
        llvm_amdgcn_raw_buffer_atomic_add_fp32(
            adder, dst_wave_buffer_resource, id * sizeof(float), 0, 0);
    }
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

    hipLaunchKernelGGL(atomicAdd, dim3(gridSize), dim3(blockSize), 0, 0, device_arry, n, 5);
    hipDeviceSynchronize();
    HIP_ASSERT(hipMemcpy(host_dst, device_arry, arry_size, hipMemcpyDeviceToHost));

    for(int i = 0; i < n; i++)
        printf("host_src[%d] = %f, host_dst[%d] = %f\n", i, host_src[i], i, host_dst[i]);

    HIP_ASSERT(hipFree(device_arry));
    free(host_src);
    free(host_dst);
    return 0;
}

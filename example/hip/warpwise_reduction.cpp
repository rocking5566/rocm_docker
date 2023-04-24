#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include "util.hpp"

template <typename to_t, typename from_t>
__device__ to_t bit_cast(const from_t& v)
{
    // TODO: how to deal with sizeof(to_t) larger than sizeof(from_t)
    static_assert(sizeof(to_t) == sizeof(from_t));
    return __builtin_bit_cast(to_t, v);
}

template <typename data_t, int wave_size>
__device__ inline data_t wave_reduce_sum(const data_t& thread_data)
{
    // wave_size must be power of 2
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = false;

    data_t result = thread_data;

    int track = 63;

    if constexpr(wave_size > 1)
    {
        result +=
            bit_cast<data_t, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, data_t>(result),
                                                           0xb1,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl)); // quad_perm:[1,0,3,2]
    }
    if constexpr(wave_size > 2)
    {
        result +=
            bit_cast<data_t, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, data_t>(result),
                                                           0x4e,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl)); // quad_perm:[2,3,0,1]
    }
    if constexpr(wave_size > 4)
    {
        result += bit_cast<data_t, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, data_t>(result),
                                                                 0x114,
                                                                 row_mask,
                                                                 bank_mask,
                                                                 bound_ctrl)); // row_shr:4
    }
    if constexpr(wave_size > 8)
    {
        result += bit_cast<data_t, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, data_t>(result),
                                                                 0x118,
                                                                 row_mask,
                                                                 bank_mask,
                                                                 bound_ctrl)); // row_shr:8
    }
    if constexpr(wave_size > 16)
    {
        result += bit_cast<data_t, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, data_t>(result),
                                                                 0x142,
                                                                 row_mask,
                                                                 bank_mask,
                                                                 bound_ctrl)); // row_bcast:15
    }
    if constexpr(wave_size > 32)
    {
        result += bit_cast<data_t, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, data_t>(result),
                                                                 0x143,
                                                                 row_mask,
                                                                 bank_mask,
                                                                 bound_ctrl)); // row_bcast:31
    }

    // now the reduced value is in the last lane of wave
    return bit_cast<data_t, int>(
        __builtin_amdgcn_readlane(bit_cast<int, data_t>(result), wave_size - 1));
}

__global__ void wave_reduce_kernel(float* input, float* output)
{
    float v      = input[threadIdx.x];
    float result = wave_reduce_sum<float, 64>(v);
    if(threadIdx.x == 0)
        *output = result;
}

int main()
{
    int total_floats = 64;
    int gridSize     = 1;
    int blockSize    = 64;

    float* dev_in;
    float* dev_out;
    float* host_in  = new float[total_floats];
    float* host_out = new float[1];
    HIP_ASSERT(hipMalloc(&dev_in, sizeof(float) * total_floats));
    HIP_ASSERT(hipMalloc(&dev_out, sizeof(float) * 1));

    for(int i = 0; i < total_floats; ++i)
        host_in[i] = static_cast<float>(i);

    HIP_ASSERT(hipMemcpy(dev_in, host_in, sizeof(float) * total_floats, hipMemcpyHostToDevice));

    // warmup
    for(int i = 0; i < 3; ++i)
        hipLaunchKernelGGL(
            wave_reduce_kernel, dim3(gridSize), dim3(blockSize), 0, 0, dev_in, dev_out);
    const int nrepeat = 10;
    hipEvent_t start, stop;
    float total_time = 0;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, nullptr);

    for(int i = 0; i < nrepeat; ++i)
        hipLaunchKernelGGL(
            wave_reduce_kernel, dim3(gridSize), dim3(blockSize), 0, 0, dev_in, dev_out);

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_time, start, stop);
    printf("time = %fms\n", total_time / nrepeat);

    HIP_ASSERT(hipMemcpy(host_out, dev_out, sizeof(float) * 1, hipMemcpyDeviceToHost));

    printf("out:%f\n", host_out[0]);

    delete[] host_in;
    delete[] host_out;
    HIP_ASSERT(hipFree(dev_in));
    HIP_ASSERT(hipFree(dev_out));
}

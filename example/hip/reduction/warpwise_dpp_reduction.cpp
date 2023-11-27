#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include "../util.hpp"

template <typename to_t, typename from_t>
__device__ to_t bit_cast(const from_t& v)
{
    // TODO: how to deal with sizeof(to_t) larger than sizeof(from_t)
    static_assert(sizeof(to_t) == sizeof(from_t));
    return __builtin_bit_cast(to_t, v);
}

template <typename T>
__device__ inline T warpwise_dpp_reduce(const T& thread_data)
{
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = false;

    T result = thread_data;

    if constexpr(warpSize > 1)
    {
        result += bit_cast<T, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, T>(result),
                                                            0xb1,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl)); // quad_perm:[1,0,3,2]
    }
    if constexpr(warpSize > 2)
    {
        result += bit_cast<T, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, T>(result),
                                                            0x4e,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl)); // quad_perm:[2,3,0,1]
    }
    if constexpr(warpSize > 4)
    {
        result += bit_cast<T, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, T>(result),
                                                            0x114,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl)); // row_shr:4
    }
    if constexpr(warpSize > 8)
    {
        result += bit_cast<T, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, T>(result),
                                                            0x118,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl)); // row_shr:8
    }
    if constexpr(warpSize > 16)
    {
        result += bit_cast<T, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, T>(result),
                                                            0x142,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl)); // row_bcast:15
    }
    if constexpr(warpSize > 32)
    {
        result += bit_cast<T, int>(__builtin_amdgcn_mov_dpp(bit_cast<int, T>(result),
                                                            0x143,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl)); // row_bcast:31
    }

    // now the reduced value is in the last lane of wave
    return bit_cast<T, int>(__builtin_amdgcn_readlane(bit_cast<int, T>(result), warpSize - 1));
}

template <typename T, int BlockSize>
__global__ void blockwise_reduce_kernel(T* input, T* output, int K)
{
    T v = input[blockIdx.x * K + threadIdx.x];
    T t = warpwise_dpp_reduce<T>(v);

    if constexpr(BlockSize == warpSize)
    {
        if(threadIdx.x == 0)
            output[blockIdx.x] = t;
    }
    else
    {
        constexpr int LdsSize = BlockSize / warpSize;
        __shared__ T work_buf[LdsSize];

        if(threadIdx.x % warpSize == 0)
        {
            int warpIdx       = threadIdx.x / warpSize;
            work_buf[warpIdx] = t;
            __syncthreads();

            for(int i = LdsSize >> 1; i > 0; i >>= 1)
            {
                if(warpIdx < i)
                    work_buf[warpIdx] += work_buf[warpIdx + i];

                __syncthreads();
            }
        }

        if(threadIdx.x == 0)
            output[blockIdx.x] = work_buf[0];
    }
}

using data_t = float;

void generate_data(data_t* tensor, int M, int K)
{
    for(int m = 0; m < M; ++m)
        for(int k = 0; k < K; ++k)
            tensor[m * K + k] = static_cast<data_t>(k);
}

int main()
{
    const int M         = 64;
    const int K         = 256;
    const int gridSize  = M;
    const int blockSize = K;

    data_t* dev_in;
    data_t* dev_out;
    data_t* host_in  = new data_t[M * K];
    data_t* host_out = new data_t[M];
    HIP_ASSERT(hipMalloc(&dev_in, sizeof(data_t) * M * K));
    HIP_ASSERT(hipMalloc(&dev_out, sizeof(data_t) * M));

    generate_data(host_in, M, K);

    HIP_ASSERT(hipMemcpy(dev_in, host_in, sizeof(data_t) * M * K, hipMemcpyHostToDevice));
    const auto reduce_kernel = blockwise_reduce_kernel<data_t, blockSize>;
    // warmup
    for(int i = 0; i < 3; ++i)
        hipLaunchKernelGGL(
            reduce_kernel, dim3(gridSize), dim3(blockSize), 0, 0, dev_in, dev_out, K);
    const int nrepeat = 100;
    hipEvent_t start, stop;
    float total_time = 0;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, nullptr);

    for(int i = 0; i < nrepeat; ++i)
        hipLaunchKernelGGL(
            reduce_kernel, dim3(gridSize), dim3(blockSize), 0, 0, dev_in, dev_out, K);

    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&total_time, start, stop);
    printf("time = %fms\n", total_time / nrepeat);

    HIP_ASSERT(hipMemcpy(host_out, dev_out, sizeof(data_t) * M, hipMemcpyDeviceToHost));

    printf("kernel:%f, ref:%d\n", host_out[0], (K - 1) * K / 2);

    delete[] host_in;
    delete[] host_out;
    HIP_ASSERT(hipFree(dev_in));
    HIP_ASSERT(hipFree(dev_out));
}

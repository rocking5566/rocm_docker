#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include "../util.hpp"

template <typename T, int BlockSize>
__global__ void blockwise_reduce_kernel(T* input, T* output, int K)
{
    float v = input[blockIdx.x * K + threadIdx.x];
    __shared__ T work_buf[BlockSize];
    work_buf[threadIdx.x] = v;
    __syncthreads();

    for(int i = BlockSize >> 1; i > 0; i >>= 1)
    {
        if(threadIdx.x < i)
            work_buf[threadIdx.x] += work_buf[threadIdx.x + i];

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        output[blockIdx.x] = work_buf[0];
    }
}

void generate_data(float* tensor, int M, int K)
{
    for(int m = 0; m < M; ++m)
        for(int k = 0; k < K; ++k)
            tensor[m * K + k] = static_cast<float>(k);
}

int main()
{
    const int M         = 64;
    const int K         = 64;
    const int gridSize  = M;
    const int blockSize = 64;

    float* dev_in;
    float* dev_out;
    float* host_in  = new float[M * K];
    float* host_out = new float[M];
    HIP_ASSERT(hipMalloc(&dev_in, sizeof(float) * M * K));
    HIP_ASSERT(hipMalloc(&dev_out, sizeof(float) * M));

    generate_data(host_in, M, K);

    HIP_ASSERT(hipMemcpy(dev_in, host_in, sizeof(float) * M * K, hipMemcpyHostToDevice));
    const auto reduce_kernel = blockwise_reduce_kernel<float, blockSize>;
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

    HIP_ASSERT(hipMemcpy(host_out, dev_out, sizeof(float) * M, hipMemcpyDeviceToHost));

    printf("kernel:%f, ref:%d\n", host_out[0], (K - 1) * K / 2);

    delete[] host_in;
    delete[] host_out;
    HIP_ASSERT(hipFree(dev_in));
    HIP_ASSERT(hipFree(dev_out));
}

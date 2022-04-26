#pragma once

#define HIP_ASSERT(x) (assert((x) == hipSuccess))
using int32x4_t = int __attribute__((ext_vector_type(4)));

template <typename T>
union BufferResource
{
    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    T* address[2];
    int32_t range[4];
    int32_t config[4];
};

template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, int element_space_size)
{
    BufferResource<T> wave_buffer_resource;

    // wavewise base address (64 bit)
    wave_buffer_resource.address[0] = const_cast<T*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range[2] = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config[3] = 0x00020000;

    return wave_buffer_resource.content;
}
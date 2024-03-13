FROM rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1

# RUN git clone -b amd-stg-open https://github.com/RadeonOpenCompute/llvm-project.git \
#     && cd llvm-project && mkdir -p build && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm \
#     && make -j install export HIP_CLANG_PATH=/dockerx/llvm-project/bin/


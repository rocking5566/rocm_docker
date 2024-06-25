FROM rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2

# RUN git clone -b amd-stg-open https://github.com/RadeonOpenCompute/llvm-project.git \
#     && cd llvm-project && mkdir -p build && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm \
#     && make -j install export HIP_CLANG_PATH=/dockerx/llvm-project/bin/


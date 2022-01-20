# Build Docker image
```sh
$ make build
```

# Run Container
```sh
$ make bash
```

# build ck
```sh
$ CXX=hipcc cmake . -B build -D HIP_ONLINE_COMPILER_FLAGS="-DCK_AMD_GPU_GFX908" -D CMAKE_CXX_FLAGS="-DCK_AMD_GPU_GFX908 -O3 --amdgpu-target=gfx908 -mllvm --amdgpu-spill-vgpr-to-agpr=0" -DCMAKE_PREFIX_PATH=/opt/rocm
$ cmake --build build -t gemm_driver_offline -j
```
or
```sh
$ mkdir build
$ cd build
$ cmake                                                                \
-D CMAKE_BUILD_TYPE=Release                                            \
-D HIP_ONLINE_COMPILER_FLAGS="-DCK_AMD_GPU_GFX908"                     \
-D CMAKE_CXX_FLAGS="-DCK_AMD_GPU_GFX908 --amdgpu-target=gfx908 -O3 "   \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                              \
-D CMAKE_PREFIX_PATH=/opt/rocm                                         \
..
$ make -j gemm_xdl
```
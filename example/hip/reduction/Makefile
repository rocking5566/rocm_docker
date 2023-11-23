clean:
	rm *.exe

build:
	/opt/rocm/bin/hipcc warpwise_reduction.cpp -std=c++17 -O3 -Wunused --amdgpu-target=gfx90a -o warpwise_reduction.exe
	/opt/rocm/bin/hipcc blockwise_reduction.cpp -std=c++14 -O3 --amdgpu-target=gfx90a -o blockwise_reduction.exe

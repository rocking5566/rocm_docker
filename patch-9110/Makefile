tag?=9110-patched-0617
IMAGE_NAME=rocm/ck-amd-private:$(tag)
REPO_ROOT?=$(shell git rev-parse --show-toplevel)
MOUNT_DIR=-v $(REPO_ROOT):/work
# MOUNT_DIR=-v /home/rocking/work/composable_kernel:/work

ROCM_PARAM=--device=/dev/kfd --device=/dev/dri \
	  --ipc=host --shm-size 16G \
	  --group-add video --cap-add=SYS_PTRACE \
	  --security-opt seccomp=unconfined

build:
	docker build -t $(IMAGE_NAME) --build-arg tag=$(tag) -f Dockerfile .

bash:
	docker run -it -w /work --privileged --rm $(MOUNT_DIR) $(ROCM_PARAM) --net=host $(IMAGE_NAME) bash

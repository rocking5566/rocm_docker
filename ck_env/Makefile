tag?=2022.03.06
IMAGE_NAME=composable_kernel:$(tag)
MOUNT_DIR=-v /home/rocking/work/composable_kernel:/work

ROCM_PARAM=--device=/dev/kfd --device=/dev/dri \
	  --ipc=host --shm-size 16G \
	  --group-add video --cap-add=SYS_PTRACE \
	  --security-opt seccomp=unconfined

build:
	docker build -t $(IMAGE_NAME) -f Dockerfile .

bash:
	docker run -it -w /work --privileged --rm $(MOUNT_DIR) $(ROCM_PARAM) --net=host $(IMAGE_NAME) bash

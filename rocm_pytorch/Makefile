tag?=rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.3.0
IMAGE_NAME=rocm/pytorch:$(tag)
MOUNT_DIR=-v ${HOME}/work:/work

ROCM_PARAM=--device=/dev/kfd --device=/dev/dri \
	  --ipc=host --shm-size 16G \
	  --group-add video --cap-add=SYS_PTRACE \
	  --security-opt seccomp=unconfined

build:
	docker build -t $(IMAGE_NAME) -f Dockerfile .

bash:
	docker run -it -w /work --privileged --rm $(MOUNT_DIR) $(ROCM_PARAM) --net=host $(IMAGE_NAME) bash

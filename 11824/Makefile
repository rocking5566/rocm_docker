tag?=11824-ubuntu-20.04-stg1
IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-dkms-no-npi-hipclang:$(tag)
# REPO_ROOT?=$(shell git rev-parse --show-toplevel)
# MOUNT_DIR=-v $(REPO_ROOT):/work
MOUNT_DIR=-v ${HOME}:/work

ROCM_PARAM=--device=/dev/kfd --device=/dev/dri \
	  --ipc=host --shm-size 16G \
	  --group-add video --cap-add=SYS_PTRACE \
	  --security-opt seccomp=unconfined

build:
	docker build -t $(IMAGE_NAME) --build-arg tag=$(tag) -f Dockerfile .

bash:
	docker run -it -w /work --privileged --rm $(MOUNT_DIR) $(ROCM_PARAM) --net=host $(IMAGE_NAME) bash

ARG base_repo=compute-artifactory.amd.com:5000/rocm-plus-docker
ARG tag=13045-ubuntu-20.04-stg1
ARG base_image=compute-rocm-dkms-no-npi-hipclang
FROM ${base_repo}/${base_image}:${tag}

RUN apt install -y htop \
                   tree \
                   clang-format-12
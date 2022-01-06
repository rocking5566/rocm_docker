ARG base_repo=compute-artifactory.amd.com:5000/rocm-plus-docker
ARG tag=9110-ubuntu-18.04-stg1
ARG base_image=compute-rocm-dkms-no-npi-hipclang
FROM ${base_repo}/${base_image}:${tag}

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV MIOPEN_VERSION="2.14.0"
RUN wget https://github.com/ROCmSoftwarePlatform/MIOpen/archive/refs/tags/${MIOPEN_VERSION}.zip \
    && unzip ${MIOPEN_VERSION}.zip \
    && cd MIOpen-${MIOPEN_VERSION} \
    && cmake -P install_deps.cmake \
    && rm -rf MIOpen-${MIOPEN_VERSION} ${MIOPEN_VERSION}.zip

RUN apt install -y htop
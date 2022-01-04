ARG base_repo=compute-artifactory.amd.com:5000/rocm-plus-docker
ARG tag=9110-ubuntu-18.04-stg1
ARG base_image=compute-rocm-dkms-no-npi-hipclang
FROM ${base_repo}/${base_image}:${tag}

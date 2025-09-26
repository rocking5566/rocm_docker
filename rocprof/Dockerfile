# Start with a base image that has the necessary tools.
# Ubuntu 22.04 is a good choice since your rocprof-trace-decoder is for that version.
FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install the required dependencies
# The --no-install-recommends flag keeps the image size down
RUN apt-get update &&                          \
    apt-get install -y --no-install-recommends \
    git cmake wget dpkg vim tig net-tools      \
    libdw-dev libsqlite3-dev                   \
&& rm -rf /var/lib/apt/lists/*

# Set the working directory for the build process
WORKDIR /work
ENV GIT_SSL_NO_VERIFY=1

# --- Build and install rocprofiler-sdk ---
# Clone the repository
RUN git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source

# Configure the build with CMake.
RUN cd rocprofiler-sdk-source && mkdir build && cd build
RUN cmake                            \
    -B rocprofiler-sdk-build         \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm \
    -DCMAKE_PREFIX_PATH=/opt/rocm    \
    rocprofiler-sdk-source ..

# Build the project, leveraging multiple cores with nproc
RUN cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)

# Install the built files
RUN cmake --build rocprofiler-sdk-build --target install

# Clean up the source & build folder to reduce image size
RUN rm -rf rocprofiler-sdk-source rocprofiler-sdk-build

# --- Install rocprof-trace-decoder ---
# Download the specific .deb file.
RUN wget "https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.4/rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb"

# Install the downloaded .deb package
RUN dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb

# Clean up the .deb file to reduce image size
RUN rm rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb

# The container is now ready to use with the installed software.
CMD ["/bin/bash"]

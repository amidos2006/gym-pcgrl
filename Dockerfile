FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "python3-opengl"]

# The block below is supposed to help w/ opengl (rendering)

## optional, if the default user is not "root", you might need to switch to root here and at the end of the script to the original user again.
## e.g.
#USER root
#
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        pkg-config \
#        libxau-dev \
#        libxdmcp-dev \
#        libxcb1-dev \
#        libxext-dev \
#        libx11-dev && \
#    rm -rf /var/lib/apt/lists/*
#
## replace with other Ubuntu version if desired
## see: https://hub.docker.com/r/nvidia/opengl/
#COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 \
#  /usr/lib/x86_64-linux-gnu \
#  /usr/lib/x86_64-linux-gnu
#
## replace with other Ubuntu version if desired
## see: https://hub.docker.com/r/nvidia/opengl/
#COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 \
#  /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
#  /usr/share/glvnd/egl_vendor.d/10_nvidia.json
#
#RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
#    ldconfig && \
#    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
#    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload
#
## nvidia-container-runtime
#ENV NVIDIA_VISIBLE_DEVICES \
#    ${NVIDIA_VISIBLE_DEVICES:-all}
#ENV NVIDIA_DRIVER_CAPABILITIES \
#    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
#
##USER sme

WORKDIR /usr/src/app
COPY setup.py README.md ./
RUN pip3 install -e .
COPY . ./
CMD ["python3", "run.py"]


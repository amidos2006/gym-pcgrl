FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "python3-opengl"]


WORKDIR /usr/src/app
COPY setup.py README.md ./
RUN pip3 install stable_baselines
RUN pip3 install -e .
COPY . ./
CMD ["python3", "train.py"]

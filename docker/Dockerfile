FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get -y update \
    && apt-get -y install \
    libopenmpi-dev \
    libpq-dev \ 
    python-dev \
    ffmpeg \ 
    libsm6 \ 
    libxext6 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir /workdir 
RUN mkdir /workdir/RL4MM
WORKDIR /workdir
COPY requirements.txt requirements.txt
RUN pip install --upgrade -r requirements.txt
COPY jupyter_launch.sh jupyter_launch.sh
RUN chmod +x jupyter_launch.sh

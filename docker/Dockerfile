FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04 as base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get -y update \
    && apt-get -y install \
    libopenmpi-dev \
    lsb-release \
    python3-pip \
    postgresql \
    python-dev \
    libpq-dev \ 
    libxext6 \
    libsm6 \ 
    ffmpeg \ 
    tmux \
    wget \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#################################
# For installing postgresql
#RUN apt-get update && apt-get install -y lsb-release && apt-get clean all
#RUN apt-get update && apt-get install -y wget && apt-get clean all
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get -y update \
    && apt-get -y install \
    postgresql \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
##################################
RUN mkdir /workdir 
RUN mkdir /workdir/RL4MM
WORKDIR /workdir
COPY requirements.txt requirements.txt
RUN pip install --upgrade -r requirements.txt
COPY jupyter_launch.sh jupyter_launch.sh
RUN chmod +x jupyter_launch.sh

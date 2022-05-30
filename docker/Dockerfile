#FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04 as base
FROM rayproject/ray:latest-gpu
ARG DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install \
    software-properties-common \
    libopenmpi-dev \
    lsb-release \
    python3.8 \
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
### Python Upgrade
#RUN apt-get -y purge python3.7
#RUN apt-get -y autoremove
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get -y install python3.8 
#RUN rm /usr/bin/python \
    #&& rm /usr/bin/python3 
    #&& ln -s /usr/bin/python3.8 /usr/bin/python  
    #&& ln -s /usr/bin/python3.8 /usr/bin/python3 \
    #&& rm /usr/local/bin/python 
    #&& rm /usr/local/bin/python3 
    #&& ln -s /usr/bin/python3.8 /usr/local/bin/python 
    #&& ln -s /usr/bin/python3.8 /usr/local/bin/python3 \
#    && apt-get install -y python3-pip python-dev python3.8-dev \ 
#    && python3 -m pip install pip --upgrade
####
##########################################
# For installing postgresql and initiating
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt-get -y update \
    && apt-get -y install \
    postgresql \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY setup_psql.sh setup_psql.sh
COPY init.sql init.sql
RUN chmod +x setup_psql.sh
# Switch to user postgres for db setup
USER postgres
RUN sh setup_psql.sh
USER root
##########################################
RUN mkdir /home/RL4MM
COPY env /home/RL4MM/.env
WORKDIR /home
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade -r requirements.txt
COPY launcher.sh launcher.sh
RUN chmod +x launcher.sh

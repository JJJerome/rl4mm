FROM rayproject/ray:latest-gpu AS RL4MM
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
COPY env /home/.env
##########################################
WORKDIR /home
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade -r requirements.txt
COPY launcher.sh launcher.sh
RUN chmod +x launcher.sh
##########################################
#For the A100 cards to work:
FROM RL4MM AS RL4MM_A100 
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


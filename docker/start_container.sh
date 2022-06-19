#!/bin/bash
docker run --rm --shm-size=10.24gb --gpus device=0 -v $2:/home/data/ -v ${PWD}/../:/home/RL4MM/ -p $1:$1 --name rl4mm --user root -dit rl4mm:latest ./launcher.sh $1

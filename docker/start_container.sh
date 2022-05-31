#!/bin/bash
docker run --rm --gpus device=0 -v ${PWD}/../:/home/RL4MM/ -p $1:$1 --name rl4mm --user root -dit rl4mm:latest ./launcher.sh $1

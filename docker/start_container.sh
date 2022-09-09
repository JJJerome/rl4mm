#!/bin/bash
tag='latest'
# tag='populated_db'
# To add a signgle gpu, add the flag --gpus device=0, or to add all gpus add --gpus all
docker run --rm --shm-size=10.24gb -v $2:/home/data/ -v ${PWD}/../:/home/RL4MM/ -p $1:$1 -p 6006:6006 --name rl4mm --user root -dit rl4mm:$tag ./launcher.sh $1

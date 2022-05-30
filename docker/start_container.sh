#!/bin/bash
docker run --rm -v ${PWD}/../:/home/RL4MM/ -p $1:$1 --name rl4mm --user root -dit rl4mm:latest ./launcher.sh $1

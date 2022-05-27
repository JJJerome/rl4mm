#!/bin/bash
#docker run --rm --gpus device=0 -v ${PWD}/../:/workdir/RL4MM/ -p $1:$1 --name rl4mm  --user root -it rl4mm:latest ./jupyter_launch.sh $1
#docker run --rm -v ${PWD}/../:/workdir/RL4MM/ -p $1:$1 --name rl4mm  --user root -it rl4mm:latest ./jupyter_launch.sh $1
docker run --rm -v ${PWD}/../:/workdir/RL4MM/ -p $1:$1 --name rl4mm  --user root -it rl4mm:latest /bin/bash #./jupyter_launch.sh $1

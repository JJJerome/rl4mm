# Docker Instructions

## Building

To build the container (including PSQL setup):

```
sh build_image.sh
```

## Running

Run the start container script (mounting ../, therefore mounting RL4MM), and specify a port for jupyter notebook and the path to your LOBSTER data:

```
sh start_container.sh 8877 /PATH/TO/LOBSTER/DATA/
```

To work in the container via shell:

```
sh exec_rl4mm.sh
```

To look up the jupyter notebook link with token:

```
jupyter notebook list
```

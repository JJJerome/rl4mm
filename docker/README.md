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

(Note: you may need to remove ```--gpus device=0``` from ```start_container.sh```)

To work in the container via shell:

```
sh exec_rl4mm.sh
```

At this point you will be attached to the container. PSQL will be running, but
the database will be empty. You can then populate it by running something like

```
python3 run_populate_database.py -mintd "2018-02-20" -maxtd "2018-04-15" --path_to_lobster_data /home/data/SPY --book_snapshot_freq "S" --max_rows 100000000 --ticker SPY
``

or to avoid ever uncompressing all the .7z files:

`
python3 run_populate_database_from_zipped.py --path_to_lobster_data /home/data/7z --book_snapshot_freq "S"
```

To look up the jupyter notebook link with token:

```
jupyter notebook list
```

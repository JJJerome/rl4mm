# rl4mm

A repository for training reinforcement learning agents to provide liquidity in high-frequency order-driven markets.
It is set up to use data provided by [LOBSTER](https://lobsterdata.com/). 

For an example of using this repository to test a variety of hand-crafted agents using Beta policies to distribute 
orders, see the associated [paper](https://arxiv.org/abs/2207.03352) on arXiv. 

### Example training call

```
python main.py -g 0 --min_date "2018-02-21" --max_date "2018-02-28" --min_date_eval "2018-03-01" --max_date_eval "2018-03-15" -psr AD --num_workers 10 --ticker SPY -epsr PnL
```

## Using rl4mm with Docker

To use the `rl4mm` package from within a docker container, first change directory into the
docker subdirectory using `cd docker` and then follow the instructions below.

### Building

To build the container (including PSQL setup):

```
sh build_image.sh
```

### Running

Run the start container script (mounting ../, therefore mounting rl4mm), and specify a port for jupyter notebook and the path to your LOBSTER data:

```
sh start_container.sh 8877 /PATH/TO/LOBSTER/DATA/
```

(Note: if you wish to add gpus to container, just add ```--gpus device=0``` to ```start_container.sh``` to use one gpu 
or ```--gpus all``` to add all gpus available.)

To work in the container via shell:

```
sh exec_rl4mm.sh
```

At this point you will be attached to the container. PSQL will be running, but
the database will be empty.

### Populating the database
You can populate the database by running (from `rl4mm/rl4mm/database/`) something like
```
python3 populate_database.py -mintd "2018-02-20" -maxtd "2018-04-15" --path_to_lobster_data /home/data/SPY --book_snapshot_freq "S" --max_rows 100000000 --ticker SPY
```
Or to avoid ever uncompressing all the .7z files:
```
python3 run_populate_database_from_zipped.py --path_to_lobster_data /home/data/7z --book_snapshot_freq "S"
```

To look up the jupyter notebook link with token:

```
jupyter notebook list
```


### Using Weights and Biases
To use Weights and Biases to log experiments:
1. install weights and biases using `pip install wandb` and create an account at [wandb.ai](wandb.ai). 
2. Log in on your machine (or from inside the docker container) using the command `wandb login`
3. Add rl4mm to git's safe directories using `git config --global --add safe.directory /PATH/TO/rl4mm` (inside the docker container, `/PATH/TO/rl4mm = /home/rl4mm`).
4. Run, for example, `main.py` with the flag `-wandb True`.


### Setting up a database outside of a docker container
If not running the rl4mm from within a docker container outlined above it is necessary to set up a postgres database 
locally using the following steps:
1. [Download and install docker](https://docs.docker.com/engine/install/ubuntu/)
2. Create a docker container running postgres (with username {USER} and password {PASSWORD}) by running:

    ```sudo docker run --name lobster -e POSTGRES_PASSWORD={PASSWORD} -e POSTGRES_USER={USER} -e POSTGRES_DB=lobster -p {LOCAL_PORT}:5432 --restart unless-stopped -d postgres -c shared_buffers=1GB```
3. Check it is running with command: `docker ps -a`
4. Create a text file named .env in the root of rl4mm directory with the following contents:
    ```
   POSTGRES_HOST=localhost
   POSTGRES_PORT=<port>
   POSTGRES_DB=rl4mm
   POSTGRES_USER=<username>
   POSTGRES_PASSWORD=<password>
   ```

### Running tests locally
We use [unittest](https://docs.python.org/2/library/unittest.html) for unit testing, [mypy](https://flake8.pycqa.org/en/latest/) as a static type checker, [Flake8](https://flake8.pycqa.org/en/latest/) to enforce PEP8 and [Black](https://black.readthedocs.io/en/stable/) to enforce consistent styling.

- Code will be automatically reformatted with: `invoke black-reformat`
- Styling and type checking tests can be run locally with: `invoke check-python`
- Unit tests can be run with: `nosetests`
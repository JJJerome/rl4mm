# RL4MM

A repository for training reinforcement learning agents to provide liquidity in high-frequency order-driven markets.
It is set up to use data provided by [LOBSTER](https://lobsterdata.com/). 

For an example of using this repository to test a variety of hand-crafted agents using Beta policies to distribute 
orders, see the associated [paper](https://arxiv.org/abs/2207.03352) on arXiv. 

### Example training call

```
python main.py -g 0 --min_date "2018-02-21" --max_date "2018-02-28" --min_date_eval "2018-03-01" --max_date_eval "2018-03-15" -psr AD --num_workers 10 --ticker SPY -epsr PnL
```

## Using RL4MM with Docker

To use the `RL4MM` package from within a docker container, first change directory into the
docker subdirectory using `cd docker` and then follow the instructions below.

### Building

To build the container (including PSQL setup):

```
sh build_image.sh
```

### Running

Run the start container script (mounting ../, therefore mounting RL4MM), and specify a port for jupyter notebook and the path to your LOBSTER data:

```
sh start_container.sh 8877 /PATH/TO/LOBSTER/DATA/
```

(Note: you may need to remove ```--gpus device=0``` from ```start_container.sh``` if you do not have any gpus available.)

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


### Using Weights and Biases
To use Weights and Biases to log experiments:
1. install weights and biases using `pip install wandb` and create an account at [wandb.ai](wandb.ai). 
2. Log in on your machine (or from inside the docker container) using the command `wandb login`
3. Add RL4MM to git's safe directories using `git config --global --add safe.directory /PATH/TO/RL4MM` (inside the docker container, `/PATH/TO/RL4MM = /home/RL4MM`).
4. Run, for example, `main.py` with the flag `-wandb True`.


### Config
#### Configure environment variables:
Make a file called `.env` in the project root. This is where we store information about the postgres database.
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
# RL4MM
A repository for simulating limit order book dynamics from past data and using it to train a reinforcement learning agent to make markets.

# Example training call

```
python main.py -g 0 --min_date "2018-02-21" --max_date "2018-02-28" --min_date_eval "2018-03-01" --max_date_eval "2018-03-15" -psr AD --num_workers 10 --ticker SPY -epsr PnL
```


To use Weights and Biases to log experiments:
1. install weights and biases using `pip install wandb` and create an account at [wandb.ai](wandb.ai). 
2. Log in on your machine (or from inside the docker container) using the command `wandb login`
3. Add RL4MM to git's safe directories using `git config --global --add safe.directory /PATH/TO/RL4MM` (inside the docker container, `/PATH/TO/RL4MM = /home/RL4MM`).
4. Run, for example, `main.py` with the flag `-wandb True`.
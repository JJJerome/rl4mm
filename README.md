# RL4MM
A repository for simulating limit order book dynamics from past data and using it to train a reinforcement learning agent to make markets.

# Example training call

```
python main.py -g 0 --min_date "2018-02-21" --max_date "2018-02-28" --min_date_eval "2018-03-01" --max_date_eval "2018-03-15" -psr AD --num_workers 10 --ticker SPY -epsr PnL
```

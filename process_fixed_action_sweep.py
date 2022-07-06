import os
import glob
import json
import pandas as pd
import numpy as np

import argparse


def parse_args():
    # -------------------- Training Args ----------------------
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-ptj", "--path_to_jsons", default="/home/data/outputs/jsons/", help="Path to jsons", type=str)
    args = vars(parser.parse_args())
    return args


def read_json(fpath):
    with open(fpath) as json_file:
        data = json.load(json_file)
    return data


def fname_to_dict(fname):
    tmp = fname.split("_")
    d = dict()

    d["strategy"] = tmp[0]
    d["alpha_bid"] = tmp[1]
    d["beta_bid"] = tmp[2]
    d["alpha_ask"] = tmp[3]
    d["beta_ask"] = tmp[4]
    d["ticker"] = tmp[5]
    d["minq"] = tmp[11]

    return d

def get_start_prices():
    import json
    fname = 'start_prices.json'
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    args = parse_args()
    fpath = args["path_to_jsons"]
    fpath = '/Users/rahul/Dropbox/ICAIF_experimental_results/COMBINED_FIXED_ACTION_LADDER'


    start_prices = get_start_prices()

    episode_length = 60 * 60
    step_length = 5

    lst = []

    for fpath in glob.glob(fpath + "/*.json"):

        data = read_json(fpath)

        fname = os.path.basename(fpath)

        # turn the file name into a dict with the params as key/values
        tmp = fname_to_dict(fname)


        # these are the mean rewards over each episode
        # i.e., rewards_series has n_iterations entries
        rewards_series = pd.Series(data["rewards"])

        # augment dictionary
        tmp["mean"] = int(np.round(rewards_series.describe().loc["mean"]))

        total_pnls = rewards_series * episode_length / step_length

        # print(total_pnls)

        ticker = tmp['ticker']
        start_price = start_prices[ticker]
    
        returns = total_pnls/start_price

        # print(returns)

        tmp['returns'] = list(returns)

        lst.append(tmp)

    df = pd.DataFrame.from_records(lst)

    # print(df.sort_values(["minq", "ticker"]))

    strategy_params = ['alpha_bid','beta_bid','alpha_ask','beta_ask']

    # count the number of positive entries in a series
    count_pos = lambda x: sum([1 if e > 0 else 0 for e in x]) 

    import itertools

    def concat_lists(x):
        return list(itertools.chain.from_iterable(x))

    df_list = []
    df_list.append(df.groupby(strategy_params)['mean'].apply(count_pos))
    df_list.append(df.groupby(strategy_params)['returns'].apply(concat_lists).apply(lambda x: np.mean(x)))
    df_list.append(df.groupby(strategy_params)['returns'].apply(concat_lists).apply(lambda x: np.std(x)))

    from functools import reduce
    df = reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True, right_index=True), df_list)

    df = np.round(df,2)
    df = df.reset_index()
    df.columns = ['alphabid',
                  'betabid',
                  'alphaask',
                  'betaask',
                  'nprofitable', 
                  'meanreturns', 
                  'sdreturns']
    df.to_csv('test.csv',index=False,float_format="%.2f")

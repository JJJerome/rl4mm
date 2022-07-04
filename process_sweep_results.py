import os
import glob
import json
import pandas as pd
import numpy as np

def read_json(fpath):
    with open(fpath) as json_file:
        data = json.load(json_file)
    return data

def fname_to_dict(fname):
    tmp = fname.split("_")
    d = dict()

    d['strategy']   = tmp[0]
    d['alpha_bid']  = tmp[1]
    d['beta_bid']   = tmp[2]
    d['alpha_ask']  = tmp[3]
    d['beta_ask']   = tmp[4]
    d['ticker']     = tmp[5]
    d['minq']       = tmp[11]

    return d

if __name__ == '__main__':

    fpath= '/Users/rahul/Dropbox/titan_sweep_results/outputs/jsons'

    lst = []

    for fpath in glob.glob(fpath + '/*.json'):

        data = read_json(fpath)

        # print(fpath)
        # print(data)

        fname = os.path.basename(fpath) 

        rewards_df = pd.DataFrame(data['rewards'])

        tmp = fname_to_dict(fname)
        tmp['mean'] = int(np.round(rewards_df.describe().loc['mean']))

        lst.append(tmp)

    df = pd.DataFrame.from_records(lst)
    print(df.sort_values(['ticker','minq']))

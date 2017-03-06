# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 16:04:27 2017

@author: Zhengyi
"""

import pandas as pd
from random import randint


def const_bid(df, price, budget):
    result = df[price > df.payprice]
    return result[:budget * 1000 // price]


def rand_bid(df, upper, budget):
    spend = 0
    budget *= 1000
    result = pd.DataFrame(columns=df.columns)
    result = []
    for idx, payprice in enumerate(df.payprice):
        bid_price = randint(0, upper)
        if bid_price > payprice:
            spend += bid_price
            if spend > budget:
                break
            result.append(idx)
    return df.iloc[result]


if __name__ == '__main__':
    import os
    import json
    from utils import dataloader, metrics
    from collections import defaultdict

    data = os.path.join('..', 'data', 'validation.csv')
    out = os.path.join('..', 'out')
    xs = range(1, 300)

    if not os.path.exists(out):
        os.mkdir(out)
        
    const_results = defaultdict(dict)
    rand_results = defaultdict(dict)
    for advertiser_id, df in dataloader(data).group():
        print advertiser_id
        const_result=const_results[str(advertiser_id)]
        rand_result=rand_results[str(advertiser_id)]
        for x in xs:
            const_result[x] = (metrics(const_bid(df, x, 25000)).to_dict())
            rand_result[x] = (metrics(rand_bid(df, x, 25000)).to_dict())

    with open(os.path.join(out, 'const_bid.json'), 'w') as f:
        json.dump(const_results, f)
    with open(os.path.join(out, 'rand_bid.json'), 'w') as f:
        json.dump(rand_results, f)

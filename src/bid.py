# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 16:04:27 2017

@author: Zhengyi
"""

from random import randint


def const_bid(df, price, budget):
    result = df[price > df.payprice]
    result = result[:budget * 1000 // price].copy()
    result.bidprice = price
    return result


def rand_bid(df, upper, budget):
    spend = 0
    budget *= 1000
    win = []
    bidprices = []
    for idx, payprice in enumerate(df.payprice):
        bidprice = randint(0, upper)
        if bidprice > payprice:
            spend += bidprice
            if spend > budget:
                break
            bidprices.append(bidprice)
            win.append(idx)
    results = df.iloc[win].copy()
    results.bidprice = bidprices
    return results


if __name__ == '__main__':
    import os
    import json
    from utils import dataloader, metrics
    from collections import defaultdict

    data = os.path.join('..', 'data', 'validation.csv')
    out = os.path.join('..', 'out')
    xs = range(1, 300)
    budgets = [10000, 20000, 25000]

    if not os.path.exists(out):
        os.mkdir(out)

    const_results = defaultdict(dict)
    rand_results = defaultdict(dict)

    for budget in budgets:
        for advertiser_id, df in dataloader(data).group():
            const_result = const_results[str(advertiser_id)]
            rand_result = rand_results[str(advertiser_id)]
            for x in xs:
                const_result[x] = (metrics(const_bid(df, x, budget)).to_dict())
                rand_result[x] = (metrics(rand_bid(df, x, budget)).to_dict())

        with open(os.path.join(out, 'const_bid_%d.json' % budget), 'w') as f:
            json.dump(const_results, f)
        with open(os.path.join(out, 'rand_bid_%d.json' % budget), 'w') as f:
            json.dump(rand_results, f)

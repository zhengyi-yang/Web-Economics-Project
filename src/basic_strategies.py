# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 16:04:27 2017

@author: Zhengyi
"""


import numpy as np
from sklearn.linear_model import LogisticRegression

import utils


def get_successful_bid(df, bidprice, budget):
    spend = 0
    budget *= 1000
    win = []
    bidprices = []
    for idx, payprice in enumerate(df.payprice):
        price = bidprice[idx]
        if price > payprice:
            spend += price
            if spend > budget:
                break
            bidprices.append(price)
            win.append(idx)
    results = df.iloc[win].copy()
    results.bidprice = bidprices
    return results


def const_bid(df, price, budget):
    bidprice = np.full(len(df), price, dtype=np.int64)
    return get_successful_bid(df, bidprice, budget)


def rand_bid(df, upper, budget):
    bidprice = np.random.randint(0, upper, len(df))
    return get_successful_bid(df, bidprice, budget)


def logistic_bid(df, base_price, budget):
    X_train, y_train = to_value(df)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    CTR = utils.metrics(df).CTR

    pCTR = logreg.predict_proba(X_train)[:, 1]
    bidprice = base_price * pCTR / CTR

    return get_successful_bid(df, bidprice, budget)


def to_value(df):
    y = df.click.values
    X = df.drop(['click', 'bidprice', 'payprice'],
                axis=1)._get_numeric_data().values
    return X, y


if __name__ == '__main__':
    import os
    import json
    from collections import defaultdict

    data = os.path.join('..', 'data', 'validation.csv')
    out = os.path.join('..', 'out')
    xs = range(1, 300)

    base_budget = 6250

    budgets = [base_budget, base_budget / 2,
               base_budget / 4, base_budget / 8, base_budget / 16]

    if not os.path.exists(out):
        os.mkdir(out)

    const_results = defaultdict(dict)
    rand_results = defaultdict(dict)
    logistic_results = defaultdict(dict)

    for budget in budgets:
        for advertiser_id, df in utils.dataloader(data).group():
            const_result = const_results[str(advertiser_id)]
            rand_result = rand_results[str(advertiser_id)]
            logistic_result = logistic_results[str(advertiser_id)]
            for x in xs:
                const_result[x] = (utils.metrics(
                    const_bid(df, x, budget)).to_dict())
                rand_result[x] = (utils.metrics(
                    rand_bid(df, x, budget)).to_dict())
                logistic_result[x] = (utils.metrics(
                    logistic_bid(df, x, budget)).to_dict())

        with open(os.path.join(out, 'const_bid_%d.json' % budget), 'w') as f:
            json.dump(const_results, f)
        with open(os.path.join(out, 'rand_bid_%d.json' % budget), 'w') as f:
            json.dump(rand_results, f)
        with open(os.path.join(out, 'logistic_bid_%d.json' % budget), 'w') as f:
            json.dump(logistic_results, f)

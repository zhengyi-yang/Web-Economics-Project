# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 16:04:27 2017

@author: Zhengyi
"""

from random import randint

# import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import metrics


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


def linear_bid(df, budget):
    """Linear bidding strategy."""
    pass


def logistic_bidprice(train, test, base_price):
    X_train, y_train = to_value(train)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    CTR = metrics(train).CTR

    X_test = test._get_numeric_data().values
    pCTR = logreg.predict_proba(X_test)
    bidprice = base_price * pCTR / CTR
    return bidprice


def to_value(df):
    y = df.click.values
    X = df.drop(['click', 'bidprice', 'payprice'],
                axis=1)._get_numeric_data().values
    return X, y


# def to_num(df):
#    non_numeric={name for name,dtype in df.dtypes.iteritems() if not issubclass(np.dtype(dtype).type, np.number)}
#    dictionary={}
#    for col in non_numeric:
#        keys=set(df[col])
#        dictionary[col]=dict(zip(keys,range(1,len(keys)+1)))
#        df[col]=df[col].map(dictionary[col])
#    return df,dictionary


if __name__ == '__main__':
    import os
    import json
    from utils import dataloader
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

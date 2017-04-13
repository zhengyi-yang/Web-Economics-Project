# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 16:04:27 2017

@author: Zhengyi
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import utils
import ORTB


logreg_cache = None


def const_bid(dataloader, price, budget):
    bidprice = np.full(len(dataloader), price, dtype=np.int64)
    return utils.get_successful_bid(dataloader, bidprice, budget)


def rand_bid(dataloader, upper, budget):
    bidprice = np.random.randint(0, upper, len(dataloader))
    return utils.get_successful_bid(dataloader, bidprice, budget)


def logistic_bid(train_loader,test_loader, base_price, budget, cache=False):
    global logreg_cache

    X_train, y_train = train_loader.to_value()

    if cache and logreg_cache is not None:
        logreg = logreg_cache
    else:
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        if cache:
            logreg_cache = logreg
    
    X_test=test_loader.to_value()[0]
    
    pCTR = logreg.predict_proba(X_test)[:, 1]
    bidprice = ORTB.linear_price(pCTR,base_price)

    return utils.get_successful_bid(test_loader, bidprice, budget)


if __name__ == '__main__':
    import os
    import json

    data = os.path.join('..', 'data', 'validation.csv')
    out = os.path.join('..', 'out')
    xs = range(1, 300)

    base_budget = 25000

    budgets = [base_budget, base_budget / 2,
               base_budget / 4, base_budget / 8, base_budget / 16]

    if not os.path.exists(out):
        os.mkdir(out)

    const_results = dict()
    rand_results = dict()
    logistic_results = dict()

    print 'loading data...'
    loader = utils.dataloader(data)

    print 'start running...'
    for budget in budgets:
        for x in tqdm(xs, desc='budget-%d' % budget):
            const_results[x] = (utils.metrics(
                const_bid(loader, x, budget)).to_dict())
            rand_results[x] = (utils.metrics(
                rand_bid(loader, x, budget)).to_dict())
            logistic_results[x] = (utils.metrics(
                logistic_bid(loader, x, budget, cache=True)).to_dict())

        with open(os.path.join(out, 'const_bid_%d.json' % budget), 'w') as f:
            json.dump(const_results, f)
        with open(os.path.join(out, 'rand_bid_%d.json' % budget), 'w') as f:
            json.dump(rand_results, f)
        with open(os.path.join(out, 'logistic_bid_%d.json' % budget), 'w') as f:
            json.dump(logistic_results, f)

    print 'finished.'

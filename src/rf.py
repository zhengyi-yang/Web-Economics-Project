#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:50:22 2017

@author: zhengyi
"""

from sklearn.ensemble import RandomForestClassifier

import utils


def rf_pCTR(train_path, test_path):

    X_train, y_train = utils.dataloader(train_path).to_value()
    X_test, _ = utils.dataloader(train_path).to_value()

    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)

    pCTR = rf.predict_proba(X_test)[:, 1]

    return pCTR


def get_bidprice(pCTR, CTR, base_price):
    return base_price * pCTR / CTR

#    CTR = utils.dataloader(train_path).metrics.CTR
#    return bidprice # utils.get_successful_bid(dataloader, bidprice, budget)


if __name__ == '__main__':
    import json

    results = {}

    train = '../data/train.csv'
    validation = '../data/validation.csv'
    budget = 6250

    pCTR = rf_pCTR(train, validation)

    baseprices = range(1, 10)

    CTR = utils.dataloader(train).metrics.CTR

    validation_loader = utils.dataloader(validation)

    for base in baseprices:
        bidprice = get_bidprice(pCTR, CTR, base)
        result = utils.get_successful_bid(validation_loader, bidprice, budget)
        results[str(base)] = utils.metrics(result).to_dict()
        print base, results[str(base)]

    with open('rf.json', 'w') as f:
        json.dump(results, f)

    utils.to_csv('rf.json')

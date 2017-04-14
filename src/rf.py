#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:50:22 2017
@author: zhengyi
@author: semihcanturk
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import utils


def rf_pCTR(train_path, test_path):

    X_train, y_train = utils.dataloader(train_path).to_value()
    X_test = utils.dataloader(test_path).to_value()[0]

    rf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [10, 50, 200],
        'max_features': ['sqrt', 'log2'],
    }

    cv_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)

    cv_rf.fit(X_train, y_train)

    print cv_rf.best_params_

    pCTR = cv_rf.predict_proba(X_test)[:, 1]

    return pCTR


def get_bidprice(pCTR, CTR, base_price):
    return base_price * pCTR / CTR


if __name__ == '__main__':
    import json

    results = {}

    train = '../data/train.csv'
    validation = '../data/validation.csv'
    budget = 6250

    pCTR = rf_pCTR(train, validation)

    baseprices = [5, 20, 55] # 5 gives close-to-optimal eCPC and CPM, #20 optimises clicks and 55 is a good
                             # 55 is a good trade-off point between CTR and CPC

    CTR = 0.0007444460603546209

    validation_loader = utils.dataloader(validation)

    for base in baseprices:
        bidprice = get_bidprice(pCTR, CTR, base)
        result = utils.get_successful_bid(validation_loader, bidprice, budget)
        results[str(base)] = utils.metrics(result).to_dict()
        print base, results[str(base)]

    with open('rf.json', 'w') as f:
        json.dump(results, f)

    utils.to_csv('rf.json')

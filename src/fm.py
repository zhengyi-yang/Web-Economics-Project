#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:26:13 2017

@author: zhengyi
"""

from fastFM import sgd

import utils


def fm_bid(dataloader, base_price, budget):
    X_train, y_train = dataloader.to_value(sparse=True, classes=[-1, 1])

    fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0,
                              l2_reg_V=0, rank=2, step_size=1e-7)
    fm.fit(X_train, y_train)

    CTR = dataloader.metrics.CTR

    pCTR = fm.predict_proba(X_train)  # BUG: return only 0 or 1
    bidprice = base_price * pCTR / CTR

    return bidprice  # get_successful_bid(df, bidprice, budget)


#data = os.path.join('..', 'data', 'validation.csv')
#loader = utils.dataloader(data, True)

#from sklearn.metrics import accuracy_score, roc_auc_score
# print 'acc:', accuracy_score(y, y_pred)
# print 'auc:', roc_auc_score(y, y_pred_proba)

# os.environ['LIBFM_PATH']=os.path.abspath('../lib/libfm/bin')

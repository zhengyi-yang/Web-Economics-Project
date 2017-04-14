
# title:  svm.py
# description:  Support Vector Machine for CTR prediction
# username: NAAK254
# author:  Neema K
# date:  11/04/2017
# usage:  python svm.py
# python_version: 3.4.3
# ==============================================================================

from utils import dataloader
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
import numpy as np
import pandas as pd
import os


def svm_strategy(path_to_training, path_to_test, path_to_val):
    """Implement GBRT strategy."""
    training_set = dataloader(path_to_training)
    test_set = pd.read_csv(path_to_test)
    val_set = pd.read_csv(path_to_val)


    bid_prices, pctr = predict_bid_price(training_set, test_set, val_set)
    bidids = np.array(test_set.bidid.values)
    bidprices = np.array(bid_prices)
    bids = pd.DataFrame(np.column_stack((bidids, bidprices)),
                        columns=['bidid', 'bidprice'])

    # Calculate the total spend
    total = get_total_spend(val_set.payprice.values, bid_prices, 12500)
    return (bids, pctr, total)


def predict_bid_price(train, test, val, test_size=0.1):
    """Predict the bid price using gradient boost classifier."""
    size = len(train.df)
    train.df = train.df.loc[np.random.choice(train.df.index, int(size // 100), replace=False)]

    y_train = train.df.click.values
    X_train = train.df.drop(['click', 'bidprice', 'payprice', 'bidid', 'logtype', 'IP', 'userid'], axis=1)._get_numeric_data().values

    size_val = len(val)
    y_test = val.click.values
    X_test = val.drop(['click', 'bidprice', 'payprice', 'bidid', 'logtype', 'IP', 'userid'],
                             axis=1)._get_numeric_data().values

    # base bids is the bid price for the average CTR cases.
    prices = train.df.bidprice.values
    base_bids = prices * 0.25 / prices.max(axis=0)

    # retrain classifier with best parameters
    clf = svm.SVC(kernel='linear', C=0.1, probability=True, max_iter=200).fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    print('AUC score: %f,\nLog loss: %f,\nRMSE: %f' % (auc_score, logloss, rmse))

    # use classifier to determine pCTR and bid_prices
    mdl = test.drop(['bidid', 'logtype'], axis=1)._get_numeric_data().values
    pctr = clf.predict_proba(mdl)[:, 1]
    print(pctr)
    avgctr = train.metrics.CTR

    bid_price = np.linalg.norm(base_bids) * pctr / avgctr  # bid price estimate
    return (bid_price, pctr)


# def tune_model(data, target):
#     """Implement parameter validation."""
#     param_grid = {'kernel': ['linear', 'poly', 'rbf'],
#                   }
#     clf = svm.SVC()
#     gs_cv = GridSearchCV(clf, param_grid).fit(data, target)
#     return gs_cv.best_params_


def get_total_spend(payprices, bidprices, budget=6250):
    """ Calculate the total spend, cap at the budget price."""
    total = 0
    for pp, bp in zip(payprices, bidprices):
        if bp > pp:
            total += pp
        if total > budget*1000:
            break
    return total // 1000


if __name__ == '__main__':
    path_to_data = '../data'
    path_to_train = os.path.join(path_to_data, 'train.csv')
    path_to_test = os.path.join(path_to_data, 'test.csv')
    path_to_val = os.path.join(path_to_data, 'validation.csv')

    results, pctr, total = svm_strategy(path_to_train, path_to_test, path_to_val)
    results.to_csv('../out/svm_results.csv', mode='w+', index=False)

    print("""Predicted CTR: %f,\nNumber clicks: %d,\nTotal cost: %d CNY fen,"""
          % (pctr.mean(), sum(pctr), total)
          )

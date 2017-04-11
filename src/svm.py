
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


def gradientboost_strategy(path_to_training, path_to_test):
    """Implement GBRT strategy."""
    training_set = dataloader(path_to_training)
    test_set = pd.read_csv(path_to_test)
    bid_prices, pctr = predict_bid_price(training_set, test_set,
                                         test_size=0.3)

    bidids = np.array(test_set.bidid.values)
    bidprices = np.array(bid_prices)
    bids = pd.DataFrame(np.column_stack((bidids, bidprices)),
                        columns=['bidid', 'bidprice'])

    return (bids, pctr)


def predict_bid_price(train, test, predictors=[], test_size=0.1, cv=True):
    """Predict the bid price using gradient boost classifier."""
    # feature selection
    # p = 0.8
    # vt = VarianceThreshold(threshold=(p * (1 - p)))

    y = train.df.click.values
    X = train.df.drop(['click', 'bidprice', 'payprice', 'bidid', 'logtype'],
                      axis=1)._get_numeric_data().values
    # X = vt.fit_transform(X_)
    X = X[:15000]
    y = y[:15000]

    size = len(X)
    split = int(round(size * (1 - test_size)))

    # split into train and test sets
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # base bids is the bid price for the average CTR cases.
    prices = train.df.bidprice.values
    base_bids = prices * 1e-4 / prices.max(axis=0)

    # select the best parameters
    best_params = {}
    if cv:
        best_params = tune_model(X_train, y_train)
        print('Optimal parameter values: %s' % (str(best_params)))

    # retrain classifier with best parameters
    clf = svm.SVC(probability=True, **best_params).fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    auc_score = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    print('AUC score: %f,\nLog loss: %f,\nRMSE: %f' % (auc_score,
                                                       logloss, rmse))

    # use classifier to determine pCTR and bid_prices
    mdl = test.drop(['bidid', 'logtype'], axis=1)._get_numeric_data().values
    pctr = clf.predict_proba(mdl)[:, 1]
    avgctr = train.metrics.CTR

    bid_price = np.linalg.norm(base_bids) * pctr / avgctr  # bid price estimate
    return (bid_price, pctr)


def tune_model(data, target):
    """Implement parameter validation."""
    param_grid = {'kernel': ['linear', 'poly', 'rbf'],
                  'C': [1, 10],
                  # 'degree': [3, 4, 6, 7],
                  # 'gamma': [0.1, 0.2, 0.5, 1.0]
                  }
    clf = svm.SVC()
    gs_cv = GridSearchCV(clf, param_grid).fit(data, target)
    return gs_cv.best_params_


if __name__ == '__main__':
    path_to_data = '../data'
    path_to_train = os.path.join(path_to_data, 'train.csv')
    path_to_test = os.path.join(path_to_data, 'test.csv')
    # path_to_val = os.path.join(path_to_data, 'validation.csv')

    results, pctr = gradientboost_strategy(path_to_train, path_to_test)

    with open('../out/gbdt_results.csv', 'w+') as res:
        results.to_csv(res)

    print("""Predicted CTR: %f,\nNumber clicks: %d,\nTotal cost: %d,"""
          % (np.mean(pctr), sum(pctr), sum(results.bidprice.values) / 1000)
          )

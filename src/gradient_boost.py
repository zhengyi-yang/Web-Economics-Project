
# title:  gradient_boost.py
# description:  Gradient Boosting Model for CTR prediction
# username: NAAK254
# author:  Neema K
# date:  04/04/2017
# usage:  python gradient_boost.py
# python_version: 3.4.3
# ==============================================================================

from utils import dataloader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd
import os


def gradientboost_strategy(path_to_training, path_to_test):
    """Implement GBRT strategy."""
    training_set = dataloader(path_to_training)
    test_set = pd.read_csv(path_to_test)
    len_test_set = len(test_set)
    # val_set = pd.read_csv(path_to_val)

    # create bidprice column for test set, this will be added to the dataframe
    bid_price_col = np.zeros(len_test_set)

    # predictors for the model:
    predictors = ['weekday', 'hour', 'userid', 'useragent', 'IP', 'region',
                  'city', 'adexchange', 'slotid', 'slotwidth',
                  'slotheight', 'slotvisibility', 'slotformat', 'slotprice',
                  'creative', 'advertiser', 'usertag\n']

    bid_prices, pctr = predict_bid_price(training_set, test_set,
                                         predictors=predictors, cv=True)

    current_cost = 0
    for i in range(len_test_set):
        bid_price_col[i] = bid_prices[i]
        current_cost += bid_prices[i]
        # maximum budget is 6,250 fen
        if current_cost >= 6250:
            break

    res = test_set
    res['bidprice'] = pd.DataFrame(bid_price_col, columns=['bidprice'])
    return (res, pctr)


def predict_bid_price(train, test, predictors, test_size=0.1, cv=True):
    """Predict the bid price using gradient boost classifier."""
    # feature selection
    p = 0.8
    vt = VarianceThreshold(threshold=(p * (1 - p)))

    y = train.df.click.values
    X = train.df.drop(['click', 'bidprice', 'payprice'],
                      axis=1)._get_numeric_data().values

    # X = vt.fit_transform(X)

    size = len(X)
    split = int(round(size * (1 - test_size)))

    # split into train and test sets
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # base bids is the bid price for the average CTR cases.
    base_bids = train.df.bidprice.values * 0.1

    # select the best parameters
    best_params = {}
    if cv:
        best_params = tune_model(X_train, y_train)
        print('Optimal parameter values: %s' % (str(best_params)))

    # retrain classifier with best parameters
    clf_2 = GradientBoostingClassifier(n_estimators=100,
                                       **best_params).fit(X_train, y_train)

    print(clf_2.score(X_test, y_test))

    # use classifier to determine pCTR and bid_prices
    mdl = test.drop([], axis=1)._get_numeric_data().values
    pctr = clf_2.predict_proba(mdl)[:, 1]     # predicted CTR
    avgctr = train.metrics.CTR     # calculate average CTR from training set

    bid_price = base_bids * np.mean(pctr) / avgctr     # bid price estimate
    return (bid_price, pctr)


def tune_model(data, target):
    """Implement parameter validation."""
    param_grid = {'learning_rate': [0.1, 0.08, 0.02],
                  'max_depth': [3, 4, 5],
                  'min_samples_leaf': [3, 5, 9, 13, 17, 21],
                  'subsample': [0.2, 0.4, 0.5, 0.6, 0.7]
                  }
    clf = GradientBoostingClassifier(n_estimators=10)
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

    print('Predicted CTR: %d,\nTotal cost: %d,' % (np.mean(pctr),
                                                   sum(results.bidprice.values))
            )

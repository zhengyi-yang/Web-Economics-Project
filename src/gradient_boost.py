
# title:  gradient_boost.py
# description:  Gradient Boosting Model for CTR prediction
# username: NAAK254
# author:  Neema K
# date:  04/04/2017
# usage:  python gradient_boost.py
# python_version: 3.4.3
# ==============================================================================

from utils import dataloader, get_successful_bid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
import numpy as np
import pandas as pd
import os


def gradientboost_strategy(path_to_training, path_to_test, path_to_val):
    """Implement GBRT strategy."""
    training_set = dataloader(path_to_training)
    test_set = pd.read_csv(path_to_test)
    val_set = pd.read_csv(path_to_val)


    bid_prices, pctr = predict_bid_price(training_set, test_set, val_set, cv=False)
    bidids = np.array(test_set.bidid.values)
    bidprices = np.array(bid_prices)
    bids = pd.DataFrame(np.column_stack((bidids, bidprices)),
                        columns=['bidid', 'bidprice'])

    # Calculate the total spend
    total = get_total_spend(val_set.payprice.values, bid_prices, 6250)
    return (bids, pctr, total)


def predict_bid_price(train, test, val, predictors=[], test_size=0.1, cv=True):
    """Predict the bid price using gradient boost classifier."""
    size = len(train.df)
    print(size)
    train.df = train.df.loc[np.random.choice(train.df.index, int(size // 50), replace=False)]

    y_train = train.df.click.values
    X_train = train.df.drop(['click', 'bidprice', 'payprice', 'bidid', 'logtype', 'IP', 'userid'], axis=1)._get_numeric_data().values

    size_val = len(val)
    y_test = val.click.values
    X_test = val.drop(['click', 'bidprice', 'payprice', 'bidid', 'logtype', 'IP', 'userid'],
                      axis=1)._get_numeric_data().values

    # base bids is the bid price for the average CTR cases.
    prices = train.df.bidprice.values
    base_bids = prices * 0.4 / prices.max(axis=0)

    # select the best parameters
    best_params = {'subsample': 0.5, 'min_samples_leaf': 13, 'n_estimators': 10}
    if cv:
        best_params = tune_model(X_train, y_train)
        print('Optimal parameter values: %s' % (str(best_params)))

    # validation error
    vclf = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, **best_params).fit(X_test, y_test)
    y_pred = vclf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)

    # retrain classifier with best parameters
    # then calculate the training error
    clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, **best_params).fit(X_train, y_train)
    y_predv = clf.predict_proba(X_train)[:, 1]
    auc_scorev = roc_auc_score(y_train, y_predv)
    loglossv = log_loss(y_train, y_predv)
    rmsev = mean_squared_error(y_train, y_predv)


    print('Train Set:\n AUC score: %f,\nLog loss: %f,\nRMSE: %f' % (auc_scorev, loglossv, rmsev))
    print('Test Set:\n AUC score: %f,\nLog loss: %f,\nRMSE: %f' % (auc_score, logloss, rmse))
    headers = train.df.drop(['click', 'bidprice', 'payprice', 'bidid', 'logtype', 'IP', 'userid'],
                            axis=1).columns.values
    print('Most important features: \n%s' % str(dict(zip(headers,
                                                         clf.feature_importances_))))

    # use classifier to determine pCTR and bid_prices
    mdl = test.drop(['bidid', 'logtype', 'IP', 'userid'], axis=1)._get_numeric_data().values

    # predicted CTR
    pctr = clf.predict_proba(mdl)[:, 1]
    # calculate average CTR from training set
    avgctr = train.metrics.CTR

    bid_price = np.linalg.norm(base_bids) * pctr / avgctr  # bid price estimate
    # bid_price = prices.mean() * pctr / avgctr
    return (bid_price, pctr)


def tune_model(data, target):
    """Implement hyperparameter validation."""
    param_grid = {'n_estimators': [10, 20, 30],
                  'min_samples_leaf': [9, 13, 17,],
                  'subsample': [0.5, 0.75, 1.0],
                  }
    clf = GradientBoostingClassifier(n_estimators=2)
    gs_cv = GridSearchCV(clf, param_grid).fit(data, target)
    return gs_cv.best_params_

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

    results, pctr, total = gradientboost_strategy(path_to_train, path_to_test, path_to_val)

    with open('../out/gbdt_results.csv', 'w+') as res:
        results.to_csv(res, index=False)

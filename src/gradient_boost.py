
"""@author: NAAK254."""

from utils import dataloader
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd


def gradientboost_strategy(path_to_training, path_to_test):
    """Implements GB-Decision Tree strategy."""
    training_set = dataloader(path_to_training)
    test_set = dataloader(path_to_test)
    len_test_set = len(test_set.df)

    # create bidprice column for test set, this will be added to the dataframe
    bid_price_col = np.zeros(len_test_set)

    # still need to define budget and base prices properly in this function
    #
    base_price = 300
    budget = 25000
    bid_prices = predict_bid_price(training_set, test_set, base_price, budget)

    for i, row in enumerate(test_set.df.values):
        bid_price_col[i] = bid_prices[i]

    res = test_set.df
    res['bidprice'] = pd.DataFrame(bid_price_col, columns=['bidprice'])

    return res

def predict_bid_price(train, test, base_price, budget):
    """Predict the bid price using gradient boost classifier."""
    train_y = train.df.click.values
    train_x = train.df.drop(['click', 'bidprice', 'payprice'],
                            axis=1)._get_numeric_data().values

    classifier = GradientBoostingClassifier().fit(train_x, train_y)

    test_x = test.df.drop([], axis=1)._get_numeric_data().values
    pctr = classifier.predict_proba(test_x)[:, 1]

    ctr = train.metrics.CTR
    bid_price = base_price * pctr / ctr

    return bid_price


if __name__ == '__main__':
    path_to_training = '../data/train.csv'
    path_to_test = '../data/test.csv'
    results = gradientboost_strategy(path_to_training, path_to_test)

    with open('../out/gbdt_results.csv', 'w+') as res:
        results.to_csv(res)

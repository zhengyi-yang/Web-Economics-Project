"""@author: NAAK254."""

from sklearn.linear_model import LogisticRegression
from utils import dataloader
import pandas as pd
import numpy as np

""" Bidding below max eCPC (Mcpc). The goal of bid optimisation is to reduce the eCPC. In [12], given the advertiser's goal on max eCPC, which is the
    upper bound of expected cost per click, the bid price on an impression is obtained by multiplying the max eCPC and the pCTR. Here we calculate the
    max eCPC for each campaign by dividing its cost and achieved number of clicks in the training data. No parameter for this bidding strategy.
    - Zhang et al, 2015.
"""


def mcpc_strategy_lin(path_to_training, path_to_test):
    """Bidding below max eCPC."""
    training_set = dataloader(path_to_training)
    test_set = dataloader(path_to_test)
    len_test_set = len(test_set.df)

    # get max eCPC for each advertiser
    max_by_adv = get_maxecpc(training_set.group())

    # create bidprice column for test set, this will be added to the dataframe
    bid_price_col = np.zeros(len_test_set)

    pctr = get_predicted_ctr(test_set, training_set)

    for i, row in enumerate(test_set.df.values):
        max_ecpc = max_by_adv[row[-2]]
        pred_ctr = pctr[i][1]
        bid_price_col[i] = int(round(max_ecpc * pred_ctr * 1000, 0))

    res = test_set.df
    res['bidprice'] = pd.DataFrame(bid_price_col, columns=['bidprice'])

    return res


def get_maxecpc(training):
    """Get maximum expected CPC for each advertiser."""
    max_ecpc = {}
    for adv, vals in training:
        costs = vals.payprice
        clicks = vals.click
        maxecpc_adv = -1
        for cost, click in zip(costs, clicks):
            if click == 1:
                cpc = cost / click
                if cpc > maxecpc_adv:
                    maxecpc_adv = cpc

        max_ecpc[adv] = maxecpc_adv
    return max_ecpc


def get_predicted_ctr(test, train):
    """Get predicted CTR. Adaptation of bid.py by Zhengyi."""
    train_y = train.df.click.values
    train_x = train.df.drop(['click', 'bidprice', 'payprice'],
                            axis=1)._get_numeric_data().values

    logreg = LogisticRegression().fit(train_x, train_y)

    test_x = test.df.drop([],
                          axis=1)._get_numeric_data().values
    pctr = logreg.predict_proba(test_x)
    return pctr


if __name__ == '__main__':
    path_to_training = '../data/train.csv'
    path_to_test = '../data/test.csv'
    results = mcpc_strategy_lin(path_to_training, path_to_test)

    with open('../out/mcpc_results.csv', 'w+') as res:
        results.to_csv(res)

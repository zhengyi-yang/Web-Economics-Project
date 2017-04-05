
"""@author: NAAK254."""

import utils
from sklearn.ensemble import GradientBoostingClassifier

def gbdt_strategy(path_to_training, path_to_test):
	"""Implements GB-Decision Tree strategy."""
	#TODO: finish this function

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
    results = gbdt_stategy(path_to_training, path_to_test)

    with open('../out/gbdt_results.csv', 'w+') as res:
        results.to_csv(res)
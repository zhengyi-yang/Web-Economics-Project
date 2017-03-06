
from pandas import read_csv
import math
import random
import numpy as np

class BasicBiddingStr:

    def __init__(self):
        self.const_val = 0
        self.random_upper_bound = 0
        self.tr_data = read_csv(path_to_trset)
        self.val_data = read_csv(path_to_valset)

    def const_bidding(self, context):
        """Constant bidding."""
        return self.const_val

    def rand_bidding(self, context):
        """Random bidding."""
        pass

    # def get_error(actual, expected):
    #     error = 0
    #     for i in len(actual):
    #         error += np.sqrt(((actual_ - expected_) ** 2).mean())

    def assign_bid_prices_const(self, dataset):
        """Assign bid prices."""
        size = len(dataset)
        bid_prices = [0] * size
        for i in range(size):
            bid_prices[0] = self.const_bidding()
        return bid_prices


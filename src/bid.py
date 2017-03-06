# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 16:04:27 2017

@author: Zhengyi
"""

import pandas as pd
from random import randint


def const_bid(df, price, budget):
    result = df[price > df.payprice]
    return result[:budget * 1000 // price]


def rand_bid(df, upper, budget):
    spend = 0
    budget *= 1000
    result = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        bid_price = randint(0, upper)
        if bid_price > row['payprice']:
            spend += bid_price
            if spend > budget:
                break
            result.loc[len(result)] = row
    return result

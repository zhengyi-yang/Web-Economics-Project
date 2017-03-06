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
    result=[]
    for idx,payprice in enumerate(df.payprice):
        bid_price = randint(0, upper)
        if bid_price > payprice:
            spend += bid_price
            if spend > budget:
                break
            result.append(idx)
    return df.iloc[result]
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 12:39:08 2017

@author: Zhengyi
"""
from __future__ import division

import pandas as pd
from scipy.sparse import csc_matrix


def get_successful_bid(dataloader, bidprice, budget):
    df = dataloader.df[['click', 'bidprice', 'payprice']]
    df.bidprice = bidprice
    df = df.loc[df.payprice < df.bidprice]
    df['spend'] = df.bidprice.cumsum()

    budget *= 1000
    df = df[df.spend <= budget]

    return df.drop(['spend'], axis=1)


class dataloader(object):

    def __init__(self, path, to_binary=False):
        df = pd.read_csv(path)
        if 'bidprice' in df.columns and 'payprice' in df.columns:
            df = df[df.bidprice > df.payprice]
        self.df = df
        self.metrics = metrics(self.df)
        if to_binary:
            self._to_binary()

    def _to_binary(self):
        cols = ['useragent', 'region', 'advertiser', 'city', 'slotvisibility']
        dummies = []
        for name in cols:
            dummy = pd.get_dummies(self.df[name], prefix=name)
            del self.df[name]
            dummies.append(dummy)

        self.df.usertags = self.df.usertag.apply(
            lambda x: set(x.split(',')) if x != 'null' else set())

        tags = reduce(set.union, self.df.usertags)
        dummy = pd.DataFrame(0, index=self.df.index, columns=tags)
        for idx, usertag in self.df.usertags.iteritems():
            dummy.loc[idx, usertag] = 1

        del self.df.usertags
        dummy.columns = ['usertags_' + str(col) for col in dummy.columns]
        dummies.append(dummy)

        self.df = pd.concat([self.df] + dummies, axis=1)

    def to_value(self, sparse=False, classes=[0, 1]):
        y = self.df.click.values
        X = self.df.drop(['click', 'bidprice', 'payprice'],
                         axis=1)._get_numeric_data().values

        if sparse:
            X = csc_matrix(X)
        if classes != [0, 1]:
            classes.sort()
            y[y == 0] = classes[0]
            y[y == 1] = classes[1]
        return X, y

    def __len__(self):
        return len(self.df.index)


class metrics(object):

    def __init__(self, df):
        self.df = df
        self.impressions = len(self.df)
        self.clicks = len(self.df[self.df.click == 1])
        self.cost = df.bidprice.sum() / 1000
        self.CTR = self.df.click.mean()
        self.CPM = df.bidprice.mean()

        if self.clicks == 0:
            self.CPC = float('NaN')
        else:
            self.CPC = self.cost / self.clicks

    def to_dict(self):
        return {'impressions': self.impressions,
                'clicks': self.clicks, 'cost': self.cost,
                'CTR': self.CTR, 'CPM': self.CPM, 'CPC': self.CPC}

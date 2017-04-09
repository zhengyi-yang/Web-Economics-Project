# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 12:39:08 2017

@author: Zhengyi
"""
from __future__ import division
import os
import json

import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file


def get_successful_bid(dataloader, bidprice, budget):
    df = dataloader.df[['click', 'bidprice', 'payprice']]
    df.bidprice = bidprice
    df = df[df.payprice < df.bidprice]
    df.loc[:, 'spend'] = df.bidprice.cumsum()

    budget *= 1000
    df = df[df.spend <= budget]

    return df.drop(['spend'], axis=1)


def to_csv(json_path):
    if os.path.isfile(json_path):
        json_paths = [json_path]
    elif os.path.isdir(json_path):
        json_paths = [os.path.join(json_path, filename)
                      for filename in os.listdir(json_path)]

    for json_path in json_paths:
        path, ext = os.path.splitext(json_path)
        if ext != '.json':
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df.T
        df.index = map(int, df.index)
        df.sort_index(inplace=True)
        df.to_csv(path + '.csv')


class dataloader(object):

    def __init__(self, path, to_binary=False, test=False):
        df = pd.read_csv(path)
        self.test = test
        self.path = os.path.abspath(path)

        if not test:
            self.df = df[df.bidprice > df.payprice]
            self.metrics = metrics(self.df)
        else:
            self.df = df
            self.metrics = None

        if to_binary:
            self._to_binary()

    def _to_binary(self):
        cols = ('useragent', 'region', 'advertiser', 'city', 'slotvisibility',
                'logtype', 'adexchange', 'slotformat')
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

    def _get_numeric_data(self):
        if not self.test:
            data = self.df.drop(['click', 'bidprice', 'payprice'],
                                axis=1)._get_numeric_data()
        else:
            data = self.df._get_numeric_data()

        return data

    def to_value(self):
        X = self._get_numeric_data().values

        if not self.test:
            y = self.df.click.values
        else:
            y = np.zeros(len(self))

        return X, y

    def dump_libfm(self, out_path):
        X, y = self.to_value()
        dump_svmlight_file(X, y, out_path)

    def get_fields_dict(self):
        '''
        return a dict of <index>:<field id>
        '''
        cols = self._get_numeric_data().columns
        fields = {}  # <field name>:<field id>
        fields_dict = {}  # <index>:<field id>

        for i, col_name in enumerate(cols):
            field = col_name.split('_')[0]
            if field not in fields:
                fields[field] = len(fields)
            fields_dict[i] = fields[field]
        return fields_dict

    def __len__(self):
        return len(self.df.index)

    def __repr__(self):
        return 'utils.dataloader at {}'.format(self.path)

    def __str__(self):
        return repr(self)


class metrics(object):

    def __init__(self, df):
        self.df = df
        self.impressions = len(self.df)
        self.clicks = len(self.df[self.df.click == 1])
        self.cost = df.bidprice.sum() / 1000
        self.CTR = self.df.click.mean()
        self.CPM = df.bidprice.mean()

        if self.clicks == 0:
            self.CPC = np.nan
        else:
            self.CPC = self.cost / self.clicks

    def to_dict(self):
        return {'impressions': self.impressions,
                'clicks': self.clicks, 'cost': self.cost,
                'CTR': self.CTR, 'CPM': self.CPM, 'CPC': self.CPC}

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 12:39:08 2017

@author: Zhengyi
"""
from __future__ import division
import os
import json
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file

# column names that will be expanded
cols = ('useragent', 'region', 'advertiser', 'city', 'slotvisibility',
        'adexchange', 'slotformat', 'usertag')


def gen_feature_data(train_path, validation_path, test_path):

    attrs = _get_all_attrs(train_path, validation_path, test_path)

    expand_features(dataloader(train_path).df,
                    attrs).to_csv(train_path + '.data', index=False)
    expand_features(dataloader(validation_path).df,
                    attrs).to_csv(validation_path + '.data', index=False)
    expand_features(dataloader(test_path, test=True).df,
                    attrs).to_csv(test_path + '.data', index=False)


def expand_features(df, attrs_dict):
    dummies = []
    df_len = len(df.index)

    with tqdm(total=df_len * len(cols)) as bar:
        for col in cols:
            dummy = pd.DataFrame(0, index=df.index, columns=attrs_dict[col])
            if col != 'usertag':
                temp = pd.get_dummies(df[col])
                col_len = len(dummy.columns)
                for column_name in dummy.columns:
                    if column_name in temp.columns:
                        dummy[column_name] = temp[column_name]
                        del temp[column_name]
                    bar.update(df_len / col_len)
            else:
                for idx, value in df['usertag'].iteritems():
                    if value != 'null':
                        for tag in value.split(','):
                            dummy.loc[idx, tag.strip()] = 1
                    bar.update()

            dummy.columns = [col + '_' + str(name) for name in dummy.columns]
            dummies.append(dummy)
            del df[col]

        return pd.concat([df] + dummies, axis=1)


def _get_all_attrs(train, validation, test, out_path=None):

    attrs = defaultdict(set)

    def add(df):
        for col in cols:
            if col != 'usertag':
                attrs[col] = set.union(attrs[col], set(df[col].unique()))
            else:
                df.usertags = df.usertag.apply(
                    lambda x: set(x.split(',')) if x != 'null' else set())
                attrs['usertag'] = reduce(set.union, df.usertags)

    add(dataloader(train).df)
    add(dataloader(validation).df)
    add(dataloader(test, test=True).df)

    attrs = {k: map(str, v) for k, v in attrs.items()}

    if out_path is not None:
        with open(out_path, 'w') as f:
            json.dump(attrs, f)

    return attrs


def get_successful_bid(dataloader, bidprice, budget):
    df = dataloader.df[['click', 'bidprice', 'payprice']].copy()
    df.bidprice = bidprice
    df = df[df.payprice < df.bidprice]
    df['spend'] = df.payprice.cumsum()

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

    def __init__(self, path, test=False):

        df = pd.read_csv(path)
        self.test = test
        self.path = os.path.abspath(path)

        if not test:
            self.df = df[df.bidprice > df.payprice]
            self.metrics = metrics(self.df)
        else:
            self.df = df
            self.metrics = None

    def _get_numeric_data(self, cols=None):
        if not self.test:
            data = self.df.drop(['click', 'bidprice', 'payprice'],
                                axis=1)._get_numeric_data()
        else:
            data = self.df._get_numeric_data()

        if cols is not None:
            data = data[cols]

        return data

    def to_value(self, cols=None):
        X = self._get_numeric_data(cols).values

        if not self.test:
            y = self.df.click.values
        else:
            y = np.zeros(len(self))

        return X, y

    def dump_libfm(self, out_path, cols=None):
        X, y = self.to_value(cols)
        dump_svmlight_file(X, y, out_path)
        return os.path.abspath(out_path)

    def get_fields_dict(self, cols=None):
        '''
        return a dict of <index>:<field id>
        '''
        cols = self._get_numeric_data(cols).columns
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
        self.impressions = len(self.df.index)
        self.clicks = len(self.df[self.df.click == 1])
        self.cost = df.payprice.sum() / 1000
        self.CTR = self.df.click.mean()
        self.CPM = df.payprice.mean()

        if self.clicks == 0:
            self.eCPC = np.nan
        else:
            self.eCPC = self.cost / self.clicks

    def to_dict(self):
        return {'impressions': self.impressions,
                'clicks': self.clicks, 'cost': self.cost,
                'CTR': self.CTR, 'CPM': self.CPM, 'eCPC': self.eCPC}


if __name__ == '__main__':
    train = '../data/train.csv'
    validation = '../data/validation.csv'
    test = '../data/test.csv'
#
#    tr = dataloader(train)
#
#    df = tr.df

#    r = {}
#    for adv in sorted(df.advertiser.unique()):
#        df_adv = df[df.advertiser == adv]
#        r[adv] = metrics(df_adv).to_dict()
#        r[adv]['Bids'] = len(df_adv.index)
#
#    r['All'] = metrics(df).to_dict()
#    r['All']['Bids'] = len(df.index)
    

#    with open('../data/attrs.json') as f:
#        r = json.load(f)

#    gen_feature_data(train, validation, test)

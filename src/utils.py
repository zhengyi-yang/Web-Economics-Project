# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 12:39:08 2017

@author: Zhengyi
"""
from __future__ import division
import pandas as pd


class dataloader(object):

    def __init__(self, path):
        df = pd.read_csv(path)
        df = df[df.bidprice > df.payprice]
        self.df = df

    def group(self):
        for advertiser_id in set(self.df.advertiser):
            yield advertiser_id, self.df[self.df.advertiser == advertiser_id]


class metrics(object):

    def __init__(self, df):
        self.df = df
        self.impressions = len(self.df)
        self.clicks = len(self.df[self.df.click == 1])
        self.cost = df.bidprice.sum() / 1000
        self.CTR = self.df.click.mean()
        self.CPM = df.bidprice.mean()

        if self.clicks == 0:
            self.CPC = 0.
        else:
            self.CPC = self.cost / self.clicks

    def to_dict(self):
        return {'impressions': self.impressions,
                'clicks': self.clicks, 'cost': self.cost,
                'CTR': self.CTR, 'CPM': self.CPM, 'CPC': self.CPC}

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 01:15:05 2017

@author: Zhengyi
"""

from __future__ import division

import numpy as np
from scipy.optimize import curve_fit


TRAIN_CTR = 0.0007444460603546209


def linear_price(pCTR, base_price, CTR=TRAIN_CTR):
    return base_price * pCTR / CTR


def ORTB1(pCTR, lambda_, c=40.490654563012058):
    return np.sqrt(c / lambda_ * pCTR + c * c) - c


def ORTB2(pCTR, lambda_, c=59.754632867436122):
    term = (pCTR + np.sqrt(c * c * lambda_ *
                           lambda_ + pCTR * pCTR)) / (c * lambda_)

    return c * (term**(1 / 3) - term**(-1 / 3))


def fit_c_ORTB1(dataloader, bounds=(20, 100)):
    x, y = _get_p_win_curve(dataloader)
    popt, pcov = curve_fit(ORTB1_w, x, y, bounds=bounds)
    return popt[0], pcov[0, 0]


def ORTB1_w(b, c):
    return b / (c + b)


def fit_c_ORTB2(dataloader, bounds=(20, 100)):
    x, y = _get_p_win_curve(dataloader)
    popt, pcov = curve_fit(ORTB2_w, x, y, bounds=bounds)
    return popt[0], pcov[0, 0]


def ORTB2_w(b, c):
    return b * b / (c * c + b * b)


def _get_p_win_curve(dataloader):
    payprice = dataloader.df.payprice.values

    bidprice = np.arange(0, np.max(payprice) + 1)
    p_win = np.empty(bidprice.shape, dtype=np.float64)

    for price in bidprice:
        p_win[price] = (payprice < price).mean()

    return bidprice, p_win

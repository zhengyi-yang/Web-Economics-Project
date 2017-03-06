import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from bid import const_bid, rand_bid
from utils import dataloader, metrics


def get_results_for_budget_limit(const_price, rand_upper, budget):
    """Get results for budget limit."""
    budget_limit = [budget // 2 ** i for i in range(0, 5)]
    yc_metrics = []
    yr_metrics = []

    for bl in budget_limit:
        result_const = const_bid(dl.df, const_price, budget)
        result_rand = rand_bid(dl.df, 300, budget)

        mtc = metrics(result_const)
        mtr = metrics(result_rand)
        yc_metrics.append([bl, mtc.impressions, mtc.clicks, mtc.cost,
                           mtc.CTR, mtc.CPM, mtc.CPC])
        yr_metrics.append([bl, mtr.impressions, mtr.clicks, mtr.cost,
                           mtr.CTR, mtr.CPM, mtr.CPC])

    headers = ['Budget Limit', 'Imps', 'Clicks', 'Cost', 'CTR', 'CPM']
    with open('../output/constant_bidding_budget_limit.txt', 'w+') as fd:
        fd.write(''.join(tabulate(yc_metrics, headers=headers,
                                  tablefmt='latex')))

    with open('../output/random_bidding_budget_limit.txt', 'w+') as fd:
        fd.write(''.join(tabulate(yr_metrics, headers=headers,
                         tablefmt='latex')))


def get_results_for_advertisers(const_price, rand_upper, budget):
    """Get results for different advertisers."""
    result_const = const_bid(dl.df, const_price, budget)
    result_rand = rand_bid(dl.df, rand_upper, budget)

if __name__ == '__main__':
    dl = dataloader('../data/validation.csv')
    mt = metrics(dl.df)

    # just chose arbitrary values to see if it was working
    get_results_for_budget_limit(100, 300, 25)

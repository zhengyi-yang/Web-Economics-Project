import pandas as pd
import json
import re
import os, sys
import numpy as np
from tabulate import tabulate
from utils import dataloader, metrics
import matplotlib.pyplot as plt


def produce_table(file_name, vis):
    """Produce tables of results for constant and random bidding strategies."""

    path_to_json = os.path.join('..', 'out', file_name)
    with open(path_to_json) as f:
        blob = f.read()
    results = json.loads(blob)

    results_table = []
    for adv in results.keys():
        row = [adv, 0, 0, 0, 0, 0]
        vals = results[adv]
        for v in vals.values():
            row[1] += v['impressions']
            row[2] += v['clicks']
            row[3] += v['cost']
            if not np.isnan(v['CTR']):
                row[4] += v['CTR']
            if not np.isnan(v['CPM']):
                row[5] += v['CPM']
        results_table.append(row)

    headers = ['Adv', 'Imps', 'Clicks', 'Cost', 'CTR', 'CPM']
    name = re.sub(r"(.json)", '', file_name)

    tables = os.path.join(vis, 'tables')
    if not os.path.exists(tables):
        os.mkdir(tables)

    with open('%s/%s' % (tables, name), 'w+') as fd:
        fd.write(''.join(tabulate(results_table, headers=headers,
                                  tablefmt='latex')))


if __name__ == '__main__':
    dl = dataloader('../data/validation.csv')
    mt = metrics(dl.df)

    # this is where all plots and tables will go
    vis = os.path.join('..', 'vis')

    if not os.path.exists(vis):
        os.mkdir(vis)

    path = '../out'
    out_files = ['const_bid_10000.json', 'const_bid_20000.json', 'const_bid_25000.json',
                 'rand_bid_10000.json', 'rand_bid_20000.json', 'rand_bid_25000.json']

    for out_file in out_files:
        produce_table(out_file, vis)

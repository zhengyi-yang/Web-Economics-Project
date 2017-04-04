import json
import re
import os
import numpy as np
from tabulate import tabulate
from utils import dataloader, metrics
import matplotlib.pyplot as plt
import pandas as pd


def to_csv(json_path):
    if os.path.isfile(json_path):
        json_paths = [json_path]
    elif os.path.isdir(json_path):
        json_paths = [os.path.join(json_path, filename)
                      for filename in os.listdir(json_path)]

    for json_path in json_paths:
        path, ext = os.path.splitext(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df.T
        df.index = map(int, df.index)
        df.sort_index(inplace=True)
        df.to_csv(path + '.csv')


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
        # take an average of the values
        for v in vals.values():
            row[1] += v['impressions']
            row[2] += v['clicks']
            row[3] += v['cost']
            if not np.isnan(v['CTR']):
                row[4] += v['CTR']
            if not np.isnan(v['CPM']):
                row[5] += v['CPM']

        size_vals = len(vals.values())

        for i in range(1, 3):
            row[i] //= size_vals

        for i in range(3, 6):
            row[i] /= size_vals

        results_table.append(row)

    headers = ['Adv', 'Imps', 'Clicks', 'Cost', 'CTR', 'CPM']
    name = re.sub(r"(.json)", '', file_name)

    tables = os.path.join(vis, 'tables')
    if not os.path.exists(tables):
        os.mkdir(tables)

    with open('%s/%s' % (tables, name), 'w+') as fd:
        fd.write(''.join(tabulate(results_table, headers=headers,
                                  # tablefmt='latex'
                                  )))
    return results_table


def produce_error_table(results_table, expected_vals, out_file):
    """Produce error table."""
    err = error(results_table, expected_vals)
    error_file_name = re.sub(r"(.json)", '', out_file)
    error_file_name = error_file_name + '_error_table'
    tables = os.path.join(vis, 'tables')

    with open('%s/%s' % (tables, error_file_name), 'w+') as fd:
        fd.write(''.join(tabulate(err,
                                  headers=['Adv', 'Imps', 'Clicks',
                                           'Cost', 'CTR', 'CPM'],
                                  # tablefmt='latex'
                                  )))


def get_expected_values(path_to_valset):
    """Get true metrics for the validation set, grouped by advertiser."""
    true_vals = []
    dl = dataloader(path_to_valset)
    group = dl.group()
    for adv, vals in group:
        mt = metrics(vals)
        mt_dict = mt.to_dict()
        true_vals.append([adv, mt_dict['impressions'], mt_dict['clicks'],
                          mt_dict['cost'], mt_dict['CTR'], mt_dict['CPM']
                          ])
    return true_vals


def error(predicted, expected):
    """Calculate error."""
    len_expected = len(expected)
    if len(predicted) != len_expected:
        return -1

    row = len(expected[0])
    error = [[0] * row for i in range(len_expected)]
    for i in range(len_expected):
        for j in range(row):
            if j == 0:
                error[i][j] = expected[i][j]
            else:
                error[i][j] = abs(predicted[i][j] -
                                  expected[i][j]) / expected[i][j]

    return error


# def plot_errorbars(results):
#     """Produce graphs for the results."""
#     charts = os.path.join('..', 'vis', 'charts')

#     if not os.path.exists(charts):
#         os.mkdir(charts)

#     fig, ax = plt.subplots()
#     budgets = [10000, 20000, 25000]
#     ax.plot(budgets, results[:3], 'ro-')
#     ax.plot(budgets, results[3:], 'go-')
#     plt.ylabel('CTR')
#     plt.xlabel('Budget')
#     plt.title('')
#     plt.show()


if __name__ == '__main__':
    path_to_valset = '../data/validation.csv'
    expected_vals = get_expected_values(path_to_valset)

    # fist number - bid price

    # this is where all plots and tables will go
    vis = os.path.join('..', 'vis')

    if not os.path.exists(vis):
        os.mkdir(vis)

    path = '../out'
    out_files = ['const_bid_10000.json', 'const_bid_20000.json', 'const_bid_25000.json',
                 'rand_bid_10000.json', 'rand_bid_20000.json', 'rand_bid_25000.json']
    ctr_values = []

    for out_file in out_files:
        results_table = produce_table(out_file, vis)
        produce_error_table(results_table, expected_vals, out_file)
        ctr_values.append(results_table[6][4])

    # plot the error bars for 2261
    # (the advertiser with the smallest error
    # plot_errorbars(ctr_values)

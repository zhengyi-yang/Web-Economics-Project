#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:26:13 2017

@author: zhengyi
"""
import os
import sys
from tempfile import mkstemp
from subprocess import Popen, PIPE

import numpy as np
from sklearn.metrics import log_loss
from tqdm import tqdm

import utils

LIBFM_PATH = os.path.abspath('../libfm')
LIBFFM_PATH = os.path.abspath('../libffm')


def gen_libfm_data(train_path, validation_path, test_path, libffm=False):
    with tqdm(total=6) as pbar:

        train = utils.dataloader(train_path)
        pbar.update()
        train_libfm_path = train_path + '.libfm'
        train.dump_libfm(train_libfm_path)
        pbar.update()
        if libffm:
            to_libffm_format(train_libfm_path,
                             train.get_fields_dict())
        del train

        validation = utils.dataloader(validation_path)
        pbar.update()
        validation_libfm_path = validation_path + '.libfm'
        validation.dump_libfm(validation_libfm_path)
        if libffm:
            to_libffm_format(validation_libfm_path,
                             validation.get_fields_dict())
        pbar.update()
        del validation

        test = utils.dataloader(test_path, test=True)
        pbar.update()
        test_libfm_path = test_path + '.libfm'
        test.dump_libfm(test_libfm_path)
        if libffm:
            to_libffm_format(test_libfm_path,
                             test.get_fields_dict())
        pbar.update()
        del test

    return train_libfm_path, validation_libfm_path, test_libfm_path


def to_libffm_format(libfm_data_path, fields_dict):
    fields_dict_str = {str(k): str(v) for k, v in fields_dict.items()}

    out = open(libfm_data_path + '.libffm', 'w')

    def add_field_id(x): return fields_dict_str[(x.split(':')[0])] + ':' + x

    with open(libfm_data_path, 'r') as f:
        for line in f:
            line = line.split(' ')
            new_line = line[:1] + map(add_field_id, line[1:])
            out.write(' '.join(new_line))
#            out.flush()
    out.close()


def fm_pCTR(train_libfm_path, validation_libfm_path, test_libfm_path,
            n_iter=10, rank=8, learn_rate=1e-7, init_stdev=0.1, out_path=None):
    train = os.path.abspath(train_libfm_path)
    validation = os.path.abspath(validation_libfm_path)
    test = os.path.abspath(test_libfm_path)

    libfm = os.path.join(LIBFM_PATH, 'bin', 'libFM')

    if _is_windows():
        libfm += '.exe'

    if not os.path.exists(libfm):
        raise RuntimeError(
            "libFM can not be found at {} ".format(LIBFM_PATH))

    if out_path is None:
        out_path = mkstemp(suffix='.libfm')[1]
    else:
        out_path = os.path.abspath(out_path)

    cmd = "{libfm} -task c -method sgda -train {train} -validation {validation} "\
          "-test {test} -iter {n_iter} -dim '1,1,{rank}' -learn_rate {learn_rate} "\
          "-init_stdev {init_stdev} -out {out}"

    cmd = cmd.format(libfm=libfm, train=train, validation=validation, test=test,
                     n_iter=n_iter, rank=rank, learn_rate=learn_rate,
                     init_stdev=init_stdev, out=out_path)

    print 'Running:\n {} \n'.format(cmd)

    returncode = _run_with_output(cmd)

    if returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError('Error occured with libFM')

    pCTR = np.loadtxt(out_path)

    return pCTR


def ffm_pCTR(train_libfm_path, validation_libfm_path, test_libfm_path,
             out_path=None, **kwargs):

    model = ffm_train(train_libfm_path, validation_libfm_path, **kwargs)

    pCTR = ffm_pred(test_libfm_path, model, out_path=out_path)

    return pCTR


def ffm_train(train_libffm_path, validation_libffm_path,
              reg=2e-5, factor=4, iteration=15, learn_rate=0.1, threads=1,
              model_path=None):
    train = os.path.abspath(train_libffm_path)
    validation = os.path.abspath(validation_libffm_path)

    if _is_windows():
        libffm_train = os.path.join(LIBFFM_PATH, 'windows', 'ffm-train.exe')
    else:
        libffm_train = os.path.join(LIBFFM_PATH, 'ffm-train.exe')

    if not os.path.exists(libffm_train):
        raise RuntimeError(
            "libFFM can not be found at {} ".format(LIBFM_PATH))

    if model_path is None:
        model_path = mkstemp(suffix='.model')[1]
    else:
        model_path = os.path.abspath(model_path)

    cmd = "{libffm} -l {reg} -k {factor} -t {iteration} -r {learn_rate} " \
          "-s {threads} -p {validation} --auto-stop {train} {model} "

    cmd = cmd.format(libffm=libffm_train, reg=reg, factor=factor, iteration=iteration, learn_rate=learn_rate, threads=threads,
                     validation=validation, train=train, model=model_path)

    print 'Running:\n {} \n'.format(cmd)

    returncode = _run_with_output(cmd)

    if returncode != 0 or not os.path.exists(model_path):
        raise RuntimeError('Error occured with libFM')

    return model_path


def ffm_pred(test_libffm_path, model_path, out_path=None):
    test = os.path.abspath(test_libffm_path)

    if _is_windows():
        libffm_pred = os.path.join(LIBFFM_PATH, 'windows', 'ffm-predict.exe')
    else:
        libffm_pred = os.path.join(LIBFFM_PATH, 'ffm-predict.exe')

    if not os.path.exists(libffm_pred):
        raise RuntimeError(
            "libFFM can not be found at {} ".format(LIBFM_PATH))

    if out_path is None:
        out_path = mkstemp(suffix='.libffm')[1]
    else:
        out_path = os.path.abspath(out_path)

    cmd = "{libffm} {test} {model} {out}"

    cmd = cmd.format(libffm=libffm_pred, test=test,
                     model=model_path, out=out_path)

    print 'Running:\n {} \n'.format(cmd)

    returncode = _run_with_output(cmd)

    if returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError('Error occured with libFM')

    pCTR = np.loadtxt(out_path)

    return pCTR


def get_log_loss(path, pCTR):
    y = utils.dataloader(path).to_value()[1]
    return log_loss(y_true=y, y_pred=pCTR)


def _is_windows():
    return sys.platform == 'win32'


def _run_with_output(cmd):

    process = Popen(cmd, stdout=PIPE, shell=True)

    while 1:
        out = process.stdout.read(1)
        if out == '' and process.poll() != None:
            break
        if out != '':
            sys.stdout.write(out)
            sys.stdout.flush()

    return process.returncode


def fm_strategy(train_path, validation_path, test_path, base_price, **kwargs):
    train_libfm = train_path + '.libfm'
    validation_libfm = validation_path + '.libfm'
    test_libfm = test_path + '.libfm'

    if not (os.path.exists(train_libfm) and
            os.path.exists(validation_libfm) and
            os.path.exists(test_libfm)):
        train_libfm, validation_libfm, test_libfm = gen_libfm_data(
            train_path, validation_path, test_path)

    pCTR = fm_pCTR(train_libfm, validation_libfm, test_libfm, **kwargs)

    CTR = utils.dataloader(train_path).metrics.CTR

    bid_prices = base_price * pCTR / CTR

    return bid_prices


if __name__ == '__main__':
    train = '../data/train.csv'
    validation = '../data/validation.csv'
    test = '../data/test.csv'

    train_libfm = '../data/train.csv.libfm'
    validation_libfm = '../data/validation.csv.libfm'
    test_libfm = '../data/test.csv.libfm'

    train_libffm = '../data/train.csv.libfm.libffm'
    validation_libffm = '../data/validation.csv.libfm.libffm'
    test_libffm = '../data/test.csv.libfm.libffm'

    out = '../out/pCTR_FM.txt'

#    bid_prices = fm_strategy(train, validation, test, 200, out_path=out)

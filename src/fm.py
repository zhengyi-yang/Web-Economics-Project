#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:26:13 2017

@author: zhengyi
"""
import os
import sys
from tempfile import mkdtemp
from subprocess import Popen, PIPE

import numpy as np
from tqdm import tqdm

import utils

LIBFM_PATH = os.path.abspath('../libfm')


def libfm_data_gen(train_path, validation_path, test_path, out_dir):
    print 'Generating libFM format data...'
    with tqdm(total=6) as pbar:

        train = utils.dataloader(train_path, to_binary=True)
        pbar.update(1)
        train_libfm_path = train_path + '.libfm'
        train.dump_libfm(train_libfm_path)
        pbar.update(2)
        del train

        validation = utils.dataloader(validation_path, to_binary=True)
        pbar.update(3)
        validation_libfm_path = validation_path + '.libfm'
        validation.dump_libfm(validation_libfm_path)
        pbar.update(4)
        del validation

        test = utils.dataloader(test_path, test=True, to_binary=True)
        pbar.update(5)
        test_libfm_path = test_path + '.libfm'
        test.dump_libfm(test_libfm_path)
        pbar.update(6)
        del test

    return train_libfm_path, validation_libfm_path, test_libfm_path


def fm_pCTR(train_libfm_path, validation_libfm_path, test_libfm_path,
            n_iter=10, rank=8, learn_rate=1e-7, init_stdev=0.1, out_path=None):
    train = os.path.abspath(train_libfm_path)
    validation = os.path.abspath(validation_libfm_path)
    test = os.path.abspath(test_libfm_path)

    libfm = os.path.join(LIBFM_PATH, 'bin', 'libFM')

    if sys.platform == 'win32':
        libfm += '.exe'
        
    if not os.path.exists(libfm):
        raise RuntimeError(
            "libFM not found, run 'make' in {} first".format(LIBFM_PATH))

    if out_path is None:
        temp_dir = mkdtemp()
        out_path = os.path.join(temp_dir, 'out.libfm')

    cmd = "{libfm} -task c -method sgda -train {train} -validation {validation} "\
          "-test {test} -iter {n_iter} -dim '1,1,{rank}' -learn_rate {learn_rate} "\
          "-init_stdev {init_stdev} -out {out}"

    cmd = cmd.format(libfm=libfm, train=train, validation=validation, test=test,
                     n_iter=n_iter, rank=rank, learn_rate=learn_rate,
                     init_stdev=init_stdev, out=out_path)

    print 'Running:\n {} \n'.format(cmd)

    returncode = _run_with_output(cmd)

    if returncode != 0:
        raise RuntimeError('Error occured with libFM')

    pCTR = np.loadtxt(out_path)

    return pCTR


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


if __name__ == '__main__':
    train = '../data/train.csv'
    validation = '../data/validation.csv'
    test = '../data/test.csv'

    out = '../out/pCTR_FM.txt'

    fm_pCTR(*libfm_data_gen(train, validation, test, '../data/'), out_path=out)
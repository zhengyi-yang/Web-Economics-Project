#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 18:26:13 2017

@author: zhengyi
"""
import os
from tempfile import mkdtemp
from subprocess import call

import numpy as np

import utils

LIBFM_PATH = os.path.abspath('../libfm')


def libfm_data_gen(train_path, validation_path, test_path, out_dir):
    print 'generating the training set...'
    train = utils.dataloader(train_path, to_binary=True)
    train_libfm_path = train_path + '.libfm'
    train.dump_libfm(train_libfm_path)

    print 'generating the validation set...'
    validation = utils.dataloader(validation_path, to_binary=True)
    validation_libfm_path = validation_path + '.libfm'
    validation.dump_libfm(validation_libfm_path)

    print 'generating the test set...'
    test = utils.dataloader(test_path, test=True, to_binary=True)
    test_libfm_path = test_path + '.libfm'
    test.dump_libfm(test_libfm_path)

    return train_libfm_path, validation_libfm_path, test_libfm_path


def fm_pCTR(train_libfm_path, validation_libfm_path, test_libfm_path,
            n_iter=10, rank=8, learn_rate=1e-7, init_stdev=0.1, out_path=None):
    train = os.path.abspath(train_libfm_path)
    validation = os.path.abspath(validation_libfm_path)
    test = os.path.abspath(test_libfm_path)

    libfm = os.path.join(LIBFM_PATH, 'bin', 'libFM')
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

    returncode = call(cmd, shell=True)
    if returncode != 0:
        raise RuntimeError('Error occured with libFM')

    pCTR = np.loadtxt(out_path)

    return pCTR

if __name__=='__main__':
    train='../data/train.csv'
    validation='../data/validation.csv'
    test='../data/test.csv'
    
    out = '../out/pCTR_FM.txt'
    
    fm_pCTR(*libfm_data_gen(train,validation,test),out_path=out)
    
    
    

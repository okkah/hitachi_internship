from __future__ import print_function
import argparse
import random
import math
import pickle
import cv2
import sys

import numpy as np
import pickle

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer.backends import cuda
from chainer import training
from chainer import serializers
from chainer.training import extensions

from chainer.functions.loss import softmax_cross_entropy

import unittest

from chainer import gradient_check

import models.vgg
from train_cub import PreprocessedDataset

from sklearn.model_selection import train_test_split
from collections import Counter


import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check
from chainer import variable


class LossLWF(chainer.Function):
    def __init__(self, t, A, idx, T=2, lambdan=1):
        super(LossLWF, self).__init__()
        self._t = t.reshape(t.size, 1)
        self._T = T
        self._batchsize = t.size
        self._lambda = lambdan
        
        self._iT = 1 / T
        self._curA = A
        
        self._vins = None
        self._yk = None
        self._c = None
    
    def forward(self, inputs):

        y, = inputs
        xp = cuda.get_array_module(y)
        
        
        
        
        S = F.softmax(y[:, :1000]).data
        self._vins = F.softmax(y[:, 1000:]).data

        loss_new = F.softmax_cross_entropy(y[:, 1000:], self._t.ravel()).data
        #loss_new = softmax_cross_entropy.softmax_cross_entropy(y[:, 1000:], self._t.ravel()).data
        
        SiT = xp.power(S, self._iT)
        X = SiT.sum(axis=1).reshape(self._batchsize, 1)
        self._yk = SiT / X
        self._c = (self._yk * self._curA).sum(axis=1).reshape(self._batchsize, 1)
        loss_old = - self._c.sum() / self._batchsize

        loss = loss_old + self._lambda * loss_new
        return loss,

    def backward(self, inputs, grad_outputs):
        y, = inputs
        gy, = grad_outputs
        
        xp = cuda.get_array_module(y)

        
        g_loss_new = self._vins.copy()
        g_loss_new[xp.arange(self._batchsize), xp.maximum(self._t.ravel(), 0)] -= 1
        
        #print('g loss new max: ', g_loss_new.max())
        
        yk = self._yk.copy()
        c = self._c.copy()
        g_loss_old = (yk * (c - self._curA)) * self._iT
        #g_loss_old *= 0
        #print('g loss old max: ', g_loss_old.max())
        
        g_loss = xp.hstack((g_loss_old, g_loss_new))
        
        return g_loss * gy / self._batchsize,


def loss_LWF(y, t, A, T=2, lambdan=1):
    return LossLWF(t, A, T, lambdan)(y)


def main():
    y = cuda.cupy.random.rand(32, 1020)    
    t = cuda.cupy.random.randint(20, size=(32,))
    with open('lwf/A.pkl', 'rb') as rf:
        A = pickle.load(rf)
    A = chainer.cuda.to_gpu(A)
    idx = 0
    
    
    
    #loss = forward_gpu(y, t, A, idx)
    #g_loss_new, g_loss_old = backward_gpu(y, t, A, idx)
    
    y_grad = cuda.cupy.random.randn(32, 1020).astype(np.float32)
    
    gradient_check.check_backward(loss_LWF, (y, t, A, cuda.cupy.asarray(idx, cuda.cupy.int32)), (y_grad), atol=1e-4, rtol=1e-4)
    

if __name__ == '__main__':
    main()


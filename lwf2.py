from __future__ import print_function
import argparse
import random
import math
import pickle
import cv2
import sys

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from chainer.backends import cuda
from chainer import training
from chainer import serializers
from chainer.training import extensions

import models.vgg
from train_cub import PreprocessedDataset

from sklearn.model_selection import train_test_split
from collections import Counter


def compute_A():
    with open('image-label.pkl', 'rb') as rf:
        image_label = pickle.load(rf)    
    root = '.'
    mean = cuda.cupy.load('mean.npy')
    image = PreprocessedDataset(image_label, root, mean, 234)
    
    model = L.Classifier(models.vgg.VGG16())
    model.to_gpu()

    A = []

    with chainer.using_config('train', False):
        for i in range(len(image)):
            print("image", i)
            x = image[i][0]
            x = x[np.newaxis, :, :, :]
            xgpu = chainer.cuda.to_gpu(x)
            ygpu = model.predictor(xgpu)
            ycpu = chainer.cuda.to_cpu(ygpu.data)
            A.append(ycpu[0])
    
    A = np.asarray(A, np.float32)
    Ap = np.power(A, 1/2)
    s = Ap.sum(axis=1)
    s = s.reshape(s.size, 1)
    A = Ap / s
    A = np.log(A)

    with open('lwf/A.pkl', 'wb') as wf:
        pickle.dump(A, wf)

    return 0


def compute_loss(y, t, A, idx):
    def loss_old_func(y, t, A, idx):
        viop = chainer.functions.softmax(y[:, 0:1000])
        T = 2

        xp = cuda.get_array_module(y)
        viopp = xp.power(viop.data, 1/T)
        s = viopp.sum(axis=1)
        s2 = s.reshape(s.size, 1)
        yoi1 = viopp / s2
        c = yoi1 * A[idx:idx+t.size]
        loss_old = - c.sum() / t.size

        #print("loss_old", loss_old)

        return loss_old

    l = 1
    loss_new = F.softmax_cross_entropy(y[:, 1000:], t)
    loss_old = loss_old_func(y, t, A, idx)
    loss = loss_old + l * loss_new
    print("loss", loss)

    return loss


def main():
    #compute_A()

    y = cuda.cupy.random.rand(32, 1020)    
    t = cuda.cupy.random.randint(20, size=32)
    with open('lwf/A.pkl', 'rb') as rf:
        A = pickle.load(rf)
    A = chainer.cuda.to_gpu(A)
    idx = 0
     
    loss = compute_loss(y, t, A, idx)


if __name__ == '__main__':
    main()

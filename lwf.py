from __future__ import print_function
import argparse
import random
import pickle
import cv2
import sys

import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.training import extensions

import models.vgg
from train_cub import PreprocessedDataset

from sklearn.model_selection import train_test_split
from collections import Counter


def compute_ai():
    print("compute_ai")
    with open('image-label.pkl', 'rb') as rf:
        image_label = pickle.load(rf)    
    root = '.'
    mean = np.load('mean.npy')
    image = PreprocessedDataset(image_label, root, mean, 234)
    
    model = L.Classifier(models.vgg.VGG16())
    model.to_gpu()

    ai = []

    with chainer.using_config('train', False):
        for i in range(len(image)):
            x = image[i][0]
            x = x[np.newaxis, :, :, :]
            xgpu = chainer.cuda.to_gpu(x)
            ygpu = model.predictor(xgpu)
            #yp = chainer.functions.softmax(ygpu)
            #p = yp.data
            #pcpu = chainer.cuda.to_cpu(p)
            ai.extend(ygpu)

    with open('lwf/ai.pkl', 'wb') as wf:
        pickle.dump(ai, wf)

    return 0


def compute_li():
    print("compute_li")
    with open('image-label.pkl', 'rb') as rf:
        image_label = pickle.load(rf)

    li = []

    for i in range(len(image_label)):
        p = [0] * 20
        p[image_label[i][1]] = 1
        li.append(p)

    with open('lwf/li.pkl', 'wb') as wf:
        pickle.dump(li, wf)

    return 0


def compute_vio_vin():
    print("compute_vio_vin")
    with open('image-label.pkl', 'rb') as rf:
        image_label = pickle.load(rf)
    root = '.'
    mean = np.load('mean.npy')
    image = PreprocessedDataset(image_label, root, mean, 234)

    model = L.Classifier(models.vgg.VGG16_LWF(1020))
    chainer.serializers.load_npz("./result/lwf_cub_model2/model.npz", model)
    model.to_gpu()

    vi = []

    with chainer.using_config('train', False):
        for i in range(len(image)):
            print("image", i)
            x = image[i][0]
            x = x[np.newaxis, :, :, :]
            xgpu = chainer.cuda.to_gpu(x)
            ygpu = model.predictor(xgpu)
            #yp = chainer.functions.softmax(ygpu)
            #p = yp.data
            #pcpu = chainer.cuda.to_cpu(p)
            vi.extend(ygpu)

    print(vi)
    vi = np.array(vi)
    print(vi.shape)
    vio, vin = np.split(vi, [1000], axis=1)
    print(vio.shape)
    print(vin.shape)
    vio = list(vio)
    vin = list(vin)

    with open('lwf/vio.pkl', 'wb') as wf:
        pickle.dump(vio, wf)
    
    with open('lwf/vin.pkl', 'wb') as wf:
        pickle.dump(vin, wf)

    return 0


def compute_lossnew():
    print("compute_lossnew")
    with open('lwf/vin.pkl', 'rb') as rf:
        vin = pickle.load(rf)
    with open('lwf/li.pkl', 'rb') as rf:
        li = pickle.load(rf)
    
    vin = np.array(vin)
    li = np.array(li)
    vinp = chainer.functions.softmax(vin)
    vinp = vinp.data
    vinplog = np.log(vinp)
    lossnew = -1 * li * vinplog
    lossnew = list(lossnew)
    
    for i in range(1115):
        for j in range(20):
            if lossnew[i][j] == -0:
                lossnew[i][j] = 0

    with open('lwf/lossnew.pkl', 'wb') as wf:
        pickle.dump(lossnew, wf)

    return 0


def compute_lossold(t):
    print("compute_lossold")
    with open('lwf/vio.pkl', 'rb') as rf:
        vio = pickle.load(rf)
    with open('lwf/ai.pkl', 'rb') as rf:
        ai = pickle.load(rf)
    T = t

    vio = np.array(vio)
    ai = np.array(ai)
    viop = chainer.functions.softmax(vio)
    viop = viop.data
    print("hoge")

    numerator = []
    numerator2 = []
    vioy = []
    aiy = []

    for i in range(1115):
        print("image", i)
        numerator.clear()
        numerator2.clear()
        denominator = 0
        denominator2 = 0

        for j in range(1000):
            numerator.append(viop[i][j] ** (1 / T))
            denominator = denominator + numerator[j]
            numerator2.append(ai[i][j] ** (1 / T))
            denominator2 = denominator2 + numerator2[j]
        
        vioy.append(numerator / denominator)
        aiy.append(numerator2 / denominator2)

    vioy = np.array(vioy)
    aiy = np.array(aiy)
    vioplog = np.log(vioy)
    lossold = -1 * aiy * vioplog
    lossold = list(lossold)
    
    for i in range(1115):
        for j in range(20):
            if lossold[i][j] == -0:
                lossold[i][j] = 0


    with open('lwf/lossold.pkl', 'wb') as wf:
        pickle.dump(lossold, wf)

    return 0


def compute_loss(l):
    print("compute_loss")
    with open('lwf/lossold.pkl', 'rb') as rf:
        lossold = pickle.load(rf)
    with open('lwf/lossnew.pkl', 'rb') as rf:
        lossnew = pickle.load(rf)

    lossold = np.array(lossold)
    lossnew = np.array(lossnew)
    L = l
    lossold = L * lossold
    loss = np.hstack((lossold, lossnew))
    loss = list(loss)

    with open('lwf/loss.pkl', 'wb') as wf:
        pickle.dump(loss, wf)

    return 0


def main():
    #compute_ai()
    #compute_li()
    #compute_vio_vin()
    #compute_lossnew()
    t = 2
    compute_lossold(t)
    l = 1
    #compute_loss(l)


if __name__ == '__main__':
    main()

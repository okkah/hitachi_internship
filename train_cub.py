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

import lwf3


#import os 
#os.chdir('/home/s2user/okamoto/1/models')


import models.vgg

from sklearn.model_selection import train_test_split
from collections import Counter


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            #if random.randint(0, 1):
            #    image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        if image.shape[0] == 1:
            image = np.vstack((image, image, image))

        image = image[:, top:bottom, left:right]
        image = cv2.resize(image.transpose(1,2,0), (224, 224))
        image = image.transpose(2,0,1)
        #image -= self.mean[:, top:bottom, left:right]
        #image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.00001,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    
    
    with open('lwf/A.pkl', 'rb') as rf:
        A = pickle.load(rf)
    A = chainer.cuda.to_gpu(A)
    A = A[:int(A.shape[0]/args.batchsize)*args.batchsize]
    
    class_labels = 20
    with open('image-label2.pkl', 'rb') as rf:
        image_label = pickle.load(rf)

    train_image_label, val_image_label = train_test_split(image_label, test_size=0.2)
 
    model = L.Classifier(models.vgg.VGG16_LWF(class_labels), A=A, lossfun=lwf3.loss_LWF)
    #model = L.Classifier(models.vgg.VGG16_LWF(class_labels), A=A, lossfun=lwf3.loss_LWF)
    #model = L.Classifier(models.vgg.VGG16_LWF(class_labels), A=A)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    #optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer = chainer.optimizers.Adam(alpha=args.learnrate)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    root = '.'
    mean = np.load('mean2.npy')
    train = PreprocessedDataset(train_image_label, root, mean, 234)
    val = PreprocessedDataset(val_image_label, root, mean, 234, False)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(val, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate
    #trainer.extend(extensions.ExponentialShift('lr', 0.5),
    #               trigger=(10, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    #trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                  'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Resume from a snapshot
    #chainer.serializers.load_npz("./result/", trainer)

    
    # Run the training
    #trainer.run()
    
    #chainer.serializers.save_npz("./result/model.npz", model) 
    
    #chainer.serializers.load_npz("./result/lwf_cub/modelLWF.npz", model)
    

    # Test mode
    def test_func(x):
        true_count = 0
        test = []
    
        with chainer.using_config('train', False):
            for i in range(len(val)):
                x = val[i][0]
                x = x[np.newaxis, :, :, :]
                xgpu = chainer.cuda.to_gpu(x)
                ygpu = model.predictor(xgpu)
                yp = chainer.functions.softmax(ygpu[:, :1000])
                p = yp.data
                pcpu = chainer.cuda.to_cpu(p)
                pcpumax = np.argmax(pcpu)
                #print(pcpu)
                #print(pcpumax)
                #print(val[i][1])
                if pcpumax == val[i][1]:
                    true_count = true_count + 1
                test.append(pcpumax)
                print(val[i][1], '  ', pcpumax)

        a = true_count
        b = len(val)
        c = true_count / len(val)
        counter = Counter(test)
        i = j = 0
        for word, cnt in counter.most_common():
            if i < 20:
                #print(word, cnt)
                j = j + cnt
                i = i + 1
        print(j, j/len(val))

        return a, b, c
    
    a, b, c = test_func(val)
    #print(a, b, c)


if __name__ == '__main__':
    main()

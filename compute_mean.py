import argparse
import sys

import numpy as np

import chainer
import pickle

def compute_mean(dataset):
    print('compute mean image')
    sum_image = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')
    return sum_image / N


def main():
    parser = argparse.ArgumentParser(description='Compute images mean array')
    with open('image-label2.pkl', 'rb') as rf:
        image_label = pickle.load(rf)
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument('--output', '-o', default='mean2.npy', help='path to output mean array')
    args = parser.parse_args()

    dataset = chainer.datasets.LabeledImageDataset(image_label, args.root)
    mean = compute_mean(dataset)
    np.save(args.output, mean)


if __name__ == '__main__':
    main()

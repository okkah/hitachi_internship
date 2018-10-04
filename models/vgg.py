import chainer
import chainer.functions as F
import chainer.links as L
import pickle

import numpy as np


class VGG16(chainer.Chain):

    def __init__(self):
        super(VGG16, self).__init__(
            base = L.VGG16Layers(),
        )

    def __call__(self, x):
        with chainer.using_config('enable_backprop', False):
            y = self.base(x, layers=['prob'])['prob']
        return y


class VGG16_Finetune(chainer.Chain):

    def __init__(self, n_class):
        super(VGG16_Finetune, self).__init__(
            base = L.VGG16Layers(),
            fc8 = L.Linear(None, n_class)
        )

    def __call__(self, x):
        with chainer.using_config('enable_backprop', False):
            h = self.base(x, layers=['pool5'])['pool5']
        h = F.dropout(F.relu(self.base.fc6(h)))
        h = F.dropout(F.relu(self.base.fc7(h)))
        y = self.fc8(h)
        return y


class VGG16_Finetune2(chainer.Chain):

    def __init__(self, n_class):
        super(VGG16_Finetune2, self).__init__()
        with self.init_scope():
            self.m1 = L.VGG16Layers()
            self.m2 = L.Classifier(VGG16_Finetune(n_class))
            chainer.serializers.load_npz("./result/ft_cub/model.npz", self.m2)
            #chainer.serializers.load_npz("./result/lwf_cub/modelLWF.npz", self.m2)
            self.m4 = self.m2.predictor
            self.m4.fc8 = self.m1.fc8

    def __call__(self, x):
         y = self.m4(x)
         return y


class VGG16_LWF(chainer.Chain):

    def __init__(self, n_class):
        super(VGG16_LWF, self).__init__(
            base = L.VGG16Layers(),
            fc8 = L.Linear(4096, 1020)
        )
        self.fc8.W.data[:1000,:] = self.base.fc8.W.data
        self.fc8.b.data[:1000] = self.base.fc8.b.data
        #self.fc8 = self.base.fc8

    def __call__(self, x):
        with chainer.using_config('enable_backprop', False):
            h = self.base(x, layers=['pool5'])['pool5']
        h = F.dropout(F.relu(self.base.fc6(h)))
        h = F.dropout(F.relu(self.base.fc7(h)))
        y = self.fc8(h)
        return y


class VGG16_LWF2(chainer.Chain):

    def __init__(self, n_class):
        super(VGG16_LWF2, self).__init__()
        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc8 = L.Linear(None, 1020)

    def __call__(self, x):
        with chainer.using_config('enable_backprop', False):
            h = self.base(x, layers=['fc7'])['fc7']
            with open('lwf/loss.pkl', 'rb') as rf:
                data = pickle.load(rf)
                data = np.array(data)
            #print(self.base.fc7.W.data.shape)
            #self.fc8.W.data = data
            y = self.fc8(h)
        return y

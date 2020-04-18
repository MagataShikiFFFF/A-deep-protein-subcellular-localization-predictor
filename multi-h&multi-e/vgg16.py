from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import function_hooks

class Linear(chainer.Chain):
    def __init__(self, n_in, n_out):
        initializer = chainer.initializers.GlorotUniform()
        super(Linear, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_in, n_out, initialW=initializer)
            self.bn = L.BatchNormalization(n_out)

    def __call__(self, x):
        h = self.linear(x)
        h = self.bn(h)

        return h

class Block(chainer.Chain):

    def __init__(self, out_channels, ksize, pad=1):
        initializer = chainer.initializers.GlorotNormal()
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True, initialW=initializer)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(x)
        

class vgg_16(chainer.Chain):

    def __init__(self):
        super(vgg_16, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, ksize=(1, 3))
            self.block1_2 = Block(64, ksize=(1, 3))
            self.block2_1 = Block(128, ksize=(1, 3))
            self.block2_2 = Block(128, ksize=(1, 3))
            self.block3_1 = Block(256, ksize=(1, 3))
            self.block3_2 = Block(256, ksize=(1, 3))
            self.block3_3 = Block(256, ksize=(1, 3))
            self.block4_1 = Block(512, ksize=(1, 3))
            self.block4_2 = Block(512, ksize=(1, 3))
            self.block4_3 = Block(512, ksize=(1, 3))
            self.block5_1 = Block(512, ksize=(1, 3))
            self.block5_2 = Block(512, ksize=(1, 3))
            self.block5_3 = Block(512, ksize=(1, 3))
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)

        return h 

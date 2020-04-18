import time
import numpy as np
import chainer

import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, Variable, function_hooks

from mh_trainer import c_gpu, g_cpu

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

        return F.relu(h)

class Autoencoder(chainer.Chain):
    def __init__(self, n_in, n_out):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.l1 = Linear(n_in, n_out)
            self.l2 = Linear(n_out, n_in)

    def __call__(self, x):

        h = self.l1(x)
        y = self.l2(h)

        return F.mean_squared_error(y, x)

class ae_Trainer(object):
    """docstring for ae_Trainer"""
    def __init__(self, epoch, data, batchsize, ae, optimizer):
        super(ae_Trainer, self).__init__()
        self.batchsize = batchsize
        self.epoch = epoch
        self.data = data
        self.ae = ae
        self.optimizer = optimizer
        self.data_length = data.shape[0]

    def run(self):      
        for epoch in range(self.epoch):
            start = time.time()
            print ('epoch', epoch)
            indexes = np.random.permutation(self.data_length)
            for i in range(0, self.data_length , self.batchsize):
                x_batch = c_gpu(self.data[indexes[i : i + self.batchsize]])
                self.ae.cleargrads()
                loss = self.ae(x_batch)
                loss.backward()
                self.optimizer.update()
            print (str(epoch) + ':' + str(time.time()-start))
            print ('loss:' + str(loss.data))   	

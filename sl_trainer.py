import time
import numpy as np
import chainer

import chainer.functions as F
import chainer.links as L
from chainer import function_hooks, cuda, Variable, function_hooks

from vgg16 import vgg_16, Linear

from evaluation import prfga, prfga_

def c_gpu(x):
    data = Variable(cuda.to_gpu(x))

    return data

def g_cpu(x):
    data = cuda.to_cpu(x)

    return data

def reshapei(x):
    data = x.reshape(x.shape[0], 1, 1, x.shape[1])
    return data

class tr_Classifier(chainer.Chain):
    def __init__(self, hid, n_classes2):
        super(tr_Classifier, self).__init__()
        with self.init_scope():
            self.FE = vgg_16()
            self.o = Linear(hid, n_classes2)

    def __call__(self, x, t):

        h = self.FE(reshapei(x))
        h = F.sigmoid(self.o(h))
        loss = F.sigmoid_cross_entropy(h, t)

        return h, loss


class Trainer_trans(object):
    """docstring for Trainer"""
    def __init__(self, epoch, data, label, model, optimizer, batchsize):
        super(Trainer_trans, self).__init__()
        self.batchsize = batchsize
        self.epoch = epoch
        self.data = data
        self.label = label
        self.model = model
        self.optimizer = optimizer
        self.data_length = data.shape[0]
        self.n_classes = label.shape[1]

    def run(self, test_data=None, test_label=None):      
        for epoch in range(self.epoch):
            start = time.time()
            print ('epoch', epoch)
            hid1 = []
            indexes = np.random.permutation(self.data_length)
            for i in range(0, self.data_length , self.batchsize):
                x_batch = c_gpu(self.data[indexes[i : i + self.batchsize]])
                y_batch = c_gpu(self.label[indexes[i : i + self.batchsize]])
                self.model.cleargrads()
                h, loss = self.model(x_batch, y_batch)
                loss.backward()
                self.optimizer.update()
                hid1.append(h.data)
            print (str(epoch) + ':' + str(time.time()-start))
            print ('loss_train:' + str(loss.data)) 
            hid1 = np.array(hid1).reshape(self.data_length, self.n_classes)
            p, r, f, g, a = prfga(hid1, self.label)
            
            print ('precision:' + str(p))
            print ('recall:' + str(r))
            print ('f1-score:' + str(f))
            print('G-mean:' + str(g))
            print ('Acc:' + str(a))

            if test_data is not None:
                final_loss = 0
                hid2 = []
                indexes = np.random.permutation(test_data.shape[0])
                for i in range(0, test_data.shape[0], self.batchsize):
                    batch_x = c_gpu(test_data[indexes[i : i + self.batchsize]])
                    batch_y = c_gpu(test_label[indexes[i : i + self.batchsize]])
                    h, loss_t = self.model(batch_x, batch_y)
                    hid2.append(h.data)
                    final_loss += loss_t.data 
                print ('loss_test:' + str(final_loss/float(self.batchsize)))
                hid2 = np.array(hid2).reshape(test_data.shape[0], self.n_classes)
                p, r, f, g, a = prfga(hid2, test_label)
                
                print ('precision:' + str(p))
                print ('recall:' + str(r))
                print ('f1-score:' + str(f))
                print ('G-mean:' + str(g))
                print ('Acc:' + str(a))


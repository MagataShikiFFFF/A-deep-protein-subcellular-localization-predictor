import time
import numpy as np
import chainer

import chainer.functions as F
import chainer.links as L
from chainer import function_hooks, cuda, Variable, function_hooks

from vgg16 import vgg_16, Linear

#from fe_cnn import FE, Linear
from evaluation import prfga, prfga_


class tr_Classifier_lgl(chainer.Chain):
    def __init__(self, hid, n_classes2):
        super(tr_Classifier_lgl, self).__init__()
        with self.init_scope():
            self.FE = vgg_16()
            self.o1 = Linear(hid, n_classes2)
            self.o2 = Linear(hid, n_classes2)

    def __call__(self, x, x1, t):

        h1 = self.FE(reshapei(x))
        h1 = F.sigmoid(self.o1(h1))
        loss1 = F.sigmoid_cross_entropy(h1, t)

        h2 = self.FE(reshapei(x1))
        h2 = F.sigmoid(self.o2(h2))
        loss2 = F.sigmoid_cross_entropy(h2, t)

        return h1, h2, loss1, loss2


class Trainer_trans_lgl(object):
    """docstring for Trainer"""
    def __init__(self, epoch, data1, data2, label, model, optimizer, batchsize):
        super(Trainer_trans_lgl, self).__init__()
        self.batchsize = batchsize
        self.epoch = epoch
        self.data1 = data1
        self.data2 = data2
        self.label = label
        self.model = model
        self.optimizer = optimizer
        self.data_length = data1.shape[0]
        self.n_classes = label.shape[1]

    def run(self, test_data1=None, test_data2=None, test_label=None):      
        for epoch in range(self.epoch):
            start = time.time()
            print ('epoch', epoch)
            hid1 = []
            indexes = np.random.permutation(self.data_length)
            for i in range(0, self.data_length , self.batchsize):
                x_batch1 = c_gpu(self.data1[indexes[i : i + self.batchsize]])
                x_batch2 = c_gpu(self.data2[indexes[i : i + self.batchsize]])
                y_batch = c_gpu(self.label[indexes[i : i + self.batchsize]])
                self.model.cleargrads()
                _, h2, loss1, loss2 = self.model(x_batch1, x_batch2, y_batch)
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.update()
                hid1.append(h2.data)
            print (str(epoch) + ':' + str(time.time()-start))
            print ('loss_train:' + str(loss.data)) 
            hid1 = np.array(hid1).reshape(self.data_length, self.n_classes)
            p, r, f, g, a = prfga(hid1, self.label)
            
            print ('precision:' + str(p))
            print ('recall:' + str(r))
            print ('f1-score:' + str(f))
            print('G-mean:' + str(g))
            print ('Acc:' + str(a))

            if test_data1 is not None:
                final_loss = 0
                hid2 = []
                indexes = np.random.permutation(test_data1.shape[0])
                for i in range(0, test_data1.shape[0], self.batchsize):
                    batch_x1 = c_gpu(test_data1[indexes[i : i + self.batchsize]])
                    batch_x2 = c_gpu(test_data2[indexes[i : i + self.batchsize]])
                    batch_y = c_gpu(test_label[indexes[i : i + self.batchsize]])
                    _, h2, loss_t1, loss_t2 = self.model(batch_x1, batch_x2, batch_y)
                    hid2.append(h2.data)
                    loss_t = loss_t1 + loss_t2
                    final_loss += loss_t.data 
                print ('loss_test:' + str(final_loss/float(self.batchsize)))
                hid2 = np.array(hid2).reshape(test_data1.shape[0], self.n_classes)
                p, r, f, g, a = prfga(hid2, test_label)
                
                print ('precision:' + str(p))
                print ('recall:' + str(r))
                print ('f1-score:' + str(f))
                print ('G-mean:' + str(g))
                print ('Acc:' + str(a))


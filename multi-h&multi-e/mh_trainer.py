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

class Classifier(chainer.Chain):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.l1 = Linear(512, n_classes)

    def __call__(self, x, t):

        y = F.sigmoid(self.l1(x))

        return y, F.sigmoid_cross_entropy(y, t)


class multi_head(chainer.Chain):
    def __init__(self, hid, n1_classes, n2_classes, n3_classes, n4_classes, n5_classes, n_classes):
        super(multi_head, self).__init__()
        with self.init_scope():
            self.FE = vgg_16()
            self.n1_classes = n1_classes
            self.o1 = Linear(hid, n1_classes)

            self.n2_classes = n2_classes
            self.o2 = Linear(hid, n2_classes)

            self.n3_classes = n3_classes
            self.o3 = Linear(hid, n3_classes)

            self.n4_classes = n4_classes
            self.o4 = Linear(hid, n4_classes)

            self.n5_classes = n5_classes
            self.o5 = Linear(hid, n5_classes)

            self.o = Linear(hid, n_classes)

#            self.o_s = Linear(hid, n_classes2)

    def __call__(self, x, t):

        h1 = self.FE(reshapei(x))
        h1 = F.sigmoid(self.o1(h1))
        loss1 = F.sigmoid_cross_entropy(h1, t[:, :self.n1_classes])
      
        h2 = self.FE(reshapei(x))
        h2 = F.sigmoid(self.o2(h2))
        loss2 = F.sigmoid_cross_entropy(h2, t[:, :self.n2_classes])

        h3 = self.FE(reshapei(x))
        h3 = F.sigmoid(self.o3(h3))       
        loss3 = F.sigmoid_cross_entropy(h3, t[:, :self.n3_classes]) 

        h4 = self.FE(reshapei(x))
        h4 = F.sigmoid(self.o4(h4))
        loss4 = F.sigmoid_cross_entropy(h4, t[:, :self.n4_classes])

        h5 = self.FE(reshapei(x))
        h5 = F.sigmoid(self.o5(h5))
        loss5 = F.sigmoid_cross_entropy(h5, t[:, :self.n5_classes]) 

        h6 = self.FE(reshapei(x))
        h6 = F.sigmoid(self.o(h6))
        loss6 = F.sigmoid_cross_entropy(h6, t)

#        h7 = self.FE(reshapei(x))
#        h7 = F.sigmoid(self.o_s(h7))
#        loss7 = F.sigmoid_cross_entropy(h7, t2)

        return h1, h2, h3, h4, h5, h6, loss1, loss2, loss3, loss4, loss5, loss6 


class Trainer_one(object):
    """docstring for Trainer"""
    def __init__(self, epoch, data, label, model, optimizer, batchsize):
        super(Trainer_one, self).__init__()
        self.batchsize = batchsize
        self.epoch = epoch
        self.data = data
        self.label = label
        #self.label2 = label2
        self.model = model
        self.optimizer = optimizer
        self.data_length = data.shape[0]
        self.n_classes = label.shape[1]

    def run(self, test_data=None, test_label=None):
        hid1 = []
        hid2 = []     
        for epoch in range(self.epoch):
            start = time.time()
            print ('epoch', epoch)
            indexes1 = np.random.permutation(self.data_length)
            for i in range(0, self.data_length , self.batchsize):
                x_batch = c_gpu(self.data[indexes1[i : i + self.batchsize]])
                y_batch = c_gpu(self.label[indexes1[i : i + self.batchsize]])
                #y_batch2 = c_gpu(self.label2[indexes1[i : i + self.batchsize]])
                self.model.cleargrads()
                _, _, _, _, _, _, loss1, loss2, loss3, loss4, loss5, loss6 = self.model(x_batch, y_batch)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 
                loss.backward()
                self.optimizer.update()
            print (str(epoch) + ':' + str(time.time()-start))
            print ('loss_train:' + str(loss.data)) 

            if test_data is not None:
                final_loss = 0
                indexes2 = np.random.permutation(test_data.shape[0])
                for i in range(0, test_data.shape[0], self.batchsize):
                    batch_x = c_gpu(test_data[indexes2[i : i + self.batchsize]])
                    batch_y = c_gpu(test_label[indexes2[i : i + self.batchsize]])
                    #batch_y2 = c_gpu(test_label2[indexes2[i : i + self.batchsize]])
                    _, _, _, _, _, _, loss_t1, loss_t2, loss_t3, loss_t4, loss_t5, loss_t6 = self.model(batch_x, batch_y)
                    loss_t = loss_t1 + loss_t2 + loss_t3 + loss_t4 + loss_t5 + loss_t6
                    final_loss += loss_t.data 
                print ('loss_test:' + str(final_loss/float(self.batchsize)))
                
        for i in range(0, self.data_length , self.batchsize):
            x_batch = c_gpu(self.data[indexes1[i : i + self.batchsize]])
            y_batch = c_gpu(self.label[indexes1[i : i + self.batchsize]])
            _, _, _, _, _, h6, _, _, _, _, _, _ = self.model(x_batch, y_batch)
            hid1.append(h6.data)
        
        hid1 = np.array(hid1).reshape(self.data_length, self.n_classes)
        p, r, f, g, a = prfga(hid1, self.label)
        print ('train')
        print ('precision:' + str(p))
        print ('recall:' + str(r))
        print ('f1-score:' + str(f))
        print('G-mean:' + str(g))
        print ('Acc:' + str(a))                
        
        for i in range(0, test_data.shape[0], self.batchsize):
            batch_x = c_gpu(test_data[indexes2[i : i + self.batchsize]])
            batch_y = c_gpu(test_label[indexes2[i : i + self.batchsize]])
            _, _, _, _, _, h6, _, _, _, _, _, _ = self.model(batch_x, batch_y)
            hid2.append(h6.data)            
        
        hid2 = np.array(hid2).reshape(test_data.shape[0], self.n_classes)
        p, r, f, g, a = prfga(hid2, test_label) 
        print ('test')           
        print ('precision:' + str(p))
        print ('recall:' + str(r))
        print ('f1-score:' + str(f))
        print ('G-mean:' + str(g))
        print ('Acc:' + str(a))


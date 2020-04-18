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

class mh_me(chainer.Chain):
    def __init__(self, hid, n1_classes, n2_classes, n3_classes, n4_classes, n5_classes, n_classes):
        super(mh_me, self).__init__()
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

            #self.o_s = Linear(hid, n_classes2)
    def __call__(self, x, x1, x2, x3, x4, x5, x6, t):

        h1 = self.FE(reshapei(x))
        h1 = F.sigmoid(self.o1(h1))
        loss1 = F.sigmoid_cross_entropy(h1, t[:, :self.n1_classes])
      
        h2 = self.FE(reshapei(x1))
        h2 = F.sigmoid(self.o2(h2))
        loss2 = F.sigmoid_cross_entropy(h2, t[:, :self.n2_classes])

        h3 = self.FE(reshapei(x2))
        h3 = F.sigmoid(self.o3(h3))       
        loss3 = F.sigmoid_cross_entropy(h3, t[:, :self.n3_classes]) 

        h4 = self.FE(reshapei(x3))
        h4 = F.sigmoid(self.o4(h4))
        loss4 = F.sigmoid_cross_entropy(h4, t[:, :self.n4_classes])

        h5 = self.FE(reshapei(x4))
        h5 = F.sigmoid(self.o5(h5))
        loss5 = F.sigmoid_cross_entropy(h5, t[:, :self.n5_classes]) 

        h6 = self.FE(reshapei(x5))
        h6 = F.sigmoid(self.o(h6))
        loss6 = F.sigmoid_cross_entropy(h6, t) 

        return h1, h2, h3, h4, h5, h6, loss1, loss2, loss3, loss4, loss5, loss6

class Trainer_all(object):
    """docstring for Trainer"""
    def __init__(self, epoch, data, data1, data2, data3, data4, data5, data6, label, model, optimizer, batchsize):
        super(Trainer_all, self).__init__()
        self.batchsize = batchsize
        self.epoch = epoch
        self.data = data
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        self.data5 = data5
        self.data6 = data6
        self.label = label
        #self.label2 = label2
        self.model = model
        self.optimizer = optimizer
        self.data_length = data.shape[0]
        self.n_classes = label.shape[1]

    def run(self, test_data=None, test_data1=None, test_data2=None, test_data3=None, test_data4=None, test_data5=None, test_data6=None, test_label=None):      
        hid1 = []
        hid2 = []         
        for epoch in range(self.epoch):
            start = time.time()
            print ('epoch', epoch)
            indexes1 = np.random.permutation(self.data_length)
            for i in range(0, self.data_length , self.batchsize):
                x_batch = c_gpu(self.data[indexes1[i : i + self.batchsize]])
                x_batch1 = c_gpu(self.data1[indexes1[i : i + self.batchsize]])
                x_batch2 = c_gpu(self.data2[indexes1[i : i + self.batchsize]])
                x_batch3 = c_gpu(self.data3[indexes1[i : i + self.batchsize]])
                x_batch4 = c_gpu(self.data4[indexes1[i : i + self.batchsize]])
                x_batch5 = c_gpu(self.data5[indexes1[i : i + self.batchsize]])
                x_batch6 = c_gpu(self.data6[indexes1[i : i + self.batchsize]])
                y_batch = c_gpu(self.label[indexes1[i : i + self.batchsize]])
                #y_batch2 = c_gpu(self.label2[indexes[i : i + self.batchsize]])
                self.model.cleargrads()
                _, _, _, _, _, _, loss1, loss2, loss3, loss4, loss5, loss6 = self.model(x_batch, x_batch1, x_batch2, x_batch3, x_batch4, x_batch5, x_batch6, y_batch)
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
                    batch_x1 = c_gpu(test_data1[indexes2[i : i + self.batchsize]])
                    batch_x2 = c_gpu(test_data2[indexes2[i : i + self.batchsize]])
                    batch_x3 = c_gpu(test_data3[indexes2[i : i + self.batchsize]])
                    batch_x4 = c_gpu(test_data4[indexes2[i : i + self.batchsize]])
                    batch_x5 = c_gpu(test_data5[indexes2[i : i + self.batchsize]])
                    batch_x6 = c_gpu(test_data6[indexes2[i : i + self.batchsize]])
                    batch_y = c_gpu(test_label[indexes2[i : i + self.batchsize]])
                    #batch_y2 = c_gpu(test_label2[indexes[i : i + self.batchsize]])
                    _, _, _, _, _, _, loss_t1, loss_t2, loss_t3, loss_t4, loss_t5, loss_t6 = self.model(batch_x, batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_y)
                    loss_t = loss_t1 + loss_t2 + loss_t3 + loss_t4 + loss_t5 + loss_t6
                    final_loss += loss_t.data 
                print ('loss_test:' + str(final_loss/float(self.batchsize)))
        
        for i in range(0, self.data_length , self.batchsize):
            x_batch = c_gpu(self.data[indexes1[i : i + self.batchsize]])
            x_batch1 = c_gpu(self.data1[indexes1[i : i + self.batchsize]])
            x_batch2 = c_gpu(self.data2[indexes1[i : i + self.batchsize]])
            x_batch3 = c_gpu(self.data3[indexes1[i : i + self.batchsize]])
            x_batch4 = c_gpu(self.data4[indexes1[i : i + self.batchsize]])
            x_batch5 = c_gpu(self.data5[indexes1[i : i + self.batchsize]])
            x_batch6 = c_gpu(self.data6[indexes1[i : i + self.batchsize]])
            y_batch = c_gpu(self.label[indexes1[i : i + self.batchsize]])            
            _, _, _, _, _, h6, _, _, _, _, _, _ = self.model(x_batch, x_batch1, x_batch2, x_batch3, x_batch4, x_batch5, x_batch6, y_batch)
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
            batch_x1 = c_gpu(test_data1[indexes2[i : i + self.batchsize]])
            batch_x2 = c_gpu(test_data2[indexes2[i : i + self.batchsize]])
            batch_x3 = c_gpu(test_data3[indexes2[i : i + self.batchsize]])
            batch_x4 = c_gpu(test_data4[indexes2[i : i + self.batchsize]])
            batch_x5 = c_gpu(test_data5[indexes2[i : i + self.batchsize]])
            batch_x6 = c_gpu(test_data6[indexes2[i : i + self.batchsize]])
            batch_y = c_gpu(test_label[indexes2[i : i + self.batchsize]])
            _, _, _, _, _, h6, _, _, _, _, _, _ = self.model(batch_x, batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_y)
            hid2.append(h6.data)                
        hid2 = np.array(hid2).reshape(test_data.shape[0], self.n_classes)
        p, r, f, g, a = prfga(hid2, test_label) 
        print ('test')           
        print ('precision:' + str(p))
        print ('recall:' + str(r))
        print ('f1-score:' + str(f))
        print ('G-mean:' + str(g))
        print ('Acc:' + str(a))


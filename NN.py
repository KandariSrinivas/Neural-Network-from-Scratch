# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:22:38 2019

@author: home
"""

import numpy as np


class Net():
    def __init__(self, x, y):
        self.dims = [5,10,2]
        self.x =x
        self.y = y
        
        self.lrate = 0.01
        self.loss = []
        self.params = {}
        self.store ={}
        self.grad = {}
        
    def init_weights(self):
        #np.random.seed(1)
        self.params['w1'] = np.random.randn(self.dims[0], self.dims[1]) ## / np.sqrt(self.dims[1])
        self.params['b1'] = np.random.randn(1, self.dims[1])
        self.params['w2'] = np.random.randn(self.dims[1], self.dims[2]) ## / np.sqrt(self.dims[2])
        self.params['b2'] = np.random.randn(1, self.dims[2])
        
    def relu(self,x):
        x[x<0] =0
        return x
    
    def sigmoid(self,x):
       return 1/(1+ np.exp(-x))
    
    def Drelu(self,x):
        x[x<0] = 0
        x[x>0] = 1
        
        return x
    
    def crossLoss(self, y_):
        return ((-np.dot(y_.T, np.log(self.y)) - np.dot(1- y_.T, np.log(1-self.y))) * 1/self.y.shape[0]).sum()
        
    def Dsigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s) 
    
    def forward(self):
        z1 = self.x.dot(self.params['w1']) + self.params['b1']         ## N,5 . 5,10 -> N,10
        self.store['z1'] = z1
        
        A1 = self.relu(z1)
        self.store['A1'] = A1
        
        z2 = A1.dot(self.params['w2']) + self.params['b2']            ## N,10 . 10,2 -> N,2
        self.store['z2'] = z2
        
        A2 = self.sigmoid(z2)
        self.store['A2'] = A2
        
        self.y_ = A2
        loss = self.crossLoss(A2)
        self.loss.append(loss)
        
        return self.y_, loss
    
    def backward(self):
        Dloss_A2 = -( self.store['A2'] / self.y  - (1- self.store['A2']) / (1 - self.y) )                               ## dLoss_Yh = — (Y/Yh — (1-Y)/(1-Yh))
        Dloss_z2 = Dloss_A2 * self.Dsigmoid(self.store['z2'])    ##N,2
        
        Dloss_w2 = self.store['A1'].T.dot(Dloss_z2)  * 1 / Dloss_z2.shape[0]                     ## 10,2
        Dloss_A1 = Dloss_z2.dot(Dloss_w2.T)     ## N,10
        Dloss_b2 = np.ones([1,Dloss_z2.shape[0]]).dot(Dloss_z2)  * 1 / Dloss_z2.shape[0] ##1,2
        
        Dloss_z1 = Dloss_A1 * self.Drelu(self.store['z1'])   ##N,10
        
        Dloss_w1 = self.x.T.dot(Dloss_z1) * 1 / Dloss_z1.shape[0]   ## 5,10
        Dloss_b1 = np.ones([1, Dloss_z1.shape[0]]).dot(Dloss_z1) * 1 / Dloss_z1.shape[0]        ##  ->1,10
       ## print(Dloss_w1, Dloss_b1, Dloss_w2, Dloss_b2)
        
        self.params['w1'] = self.params['w1'] - self.lrate * Dloss_w1
        self.params['b1'] = self.params['b1'] - self.lrate * Dloss_b1
        self.params['w2'] = self.params['w2'] - self.lrate * Dloss_w2
        self.params['b2'] = self.params['b2'] - self.lrate * Dloss_b2
    
        
        
    def gd(self, iter = 3000):
        np.random.seed(1)                         
    
        self.init_weights()
    
        for i in range(0, iter):
            Yh, loss=self.forward()
            self.backward()
            
        
            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, loss))
                print("############ Gradient ##########")
                print(gradient)
                print("############ Gradient ##########")
                self.loss.append(loss)
    
        return


        
    #### data is completely random don't expect params to converge
x = np.random.randint(low = 0, high = 100, size = [200,5])
coeffs = np.random.randint(low=1, high=100, size=[5,2]) / 100
y = np.ones((2,200))
y[0][::2]= 0.3
y[0][1::2]= 0.7
y[1][1::2]= 0.3
y[1][::2]= 0.7
y=y.T

net = Net(x,y)
net.gd()

        
        
        
        
        
        
        
        
        

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 08:48:40 2019

@author: hyang1985
"""

from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.constraints import max_norm
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



shIndex=pd.read_csv("C:/Users/yang/Downloads/shIndex.csv")
timeLen=5 #如果分析周期为11，前10个交易日预测下一个交易日
pos=0
X=np.zeros((len(shIndex['Close'])-timeLen,timeLen,1),dtype=np.float)
Y=np.zeros((len(shIndex['Close'])-timeLen),dtype=np.float)
while pos<len(shIndex['Close'])-timeLen:
    Y[pos]=shIndex['Close'][pos+timeLen]
    for j in range(0,timeLen):
        X[pos,j,0]=shIndex['Close'][pos:pos+timeLen][j+pos]
        #X[pos,j,1]=shIndex['ChgPct'][pos:pos+timeLen][j+pos]
        #X[pos,j,2]=shIndex['TurnoverValue'][pos:pos+timeLen][j+pos]
        #X[pos,j,3]=shIndex['TurnoverVol'][pos:pos+timeLen][j+pos]
    pos=pos+1 
    
#X[np.isnan(X[...,1])]=np.mean(X[~np.isnan(X[...,1])])
min_max_scaler=MinMaxScaler()
X[...,0]=min_max_scaler.fit_transform(X[...,0])
#X[...,1]=min_max_scaler.fit_transform(X[...,1])
#X[...,2]=min_max_scaler.fit_transform(X[...,2])
#X[...,3]=min_max_scaler.fit_transform(X[...,3])
Y = min_max_scaler.fit_transform(Y.reshape(-1,1))


model = Sequential()
model.add(LSTM(120,input_dim=1,stateful=False,return_sequences=False,kernel_initializer='random_uniform'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("tanh"))
model.compile(loss="mse", optimizer="Adam")
model.fit(X,Y,epochs=200,batch_size=80,validation_split=0.1)
Y=min_max_scaler.inverse_transform(Y)
Y_pred=min_max_scaler.inverse_transform(model.predict(X))

plt.figure()
plt.title("stock price")
plt.plot(Y,'b',label='real')
plt.plot(Y_pred,'r',label='predict')
plt.legend()
plt.show()

ma5=np.mean(X,axis=1)
ma5=min_max_scaler.inverse_transform(ma5)
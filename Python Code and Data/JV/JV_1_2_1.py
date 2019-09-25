# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:48:24 2019
@author: Depp Pc
"""


import os
exp_str = "dataout/JVtest_concentration1_1"
os.makedirs(exp_str, exist_ok=True)
import pandas as pd
import numpy as np
import time
import copy
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import sklearn.metrics as skm
#Read data file
df_original = pd.read_csv("fig7acsv_1_2.csv")
import math


df = copy.deepcopy(df_original) 

from utility2 import make_norm, make_norm_given_lim, make_unnorm, my_r2_score, split_tt, plot_true_vs_pred
#df_normed, norm_lims = make_norm(df, "V","concentration")
df_normed, norm_lims = make_norm(df, *df.columns.tolist())


#np.random.seed(3) ; np.random.rand(len(df)) <0.6



    
data = {'V':df_original.loc[0:25,"V"]}
data2= pd.DataFrame(data)

test_concentration=15
conc_value = pd.DataFrame({'concentration' : np.tile([test_concentration], 26)})  
target_value = pd.DataFrame({'target' : np.tile([0], 26)})

dftest = pd.concat([data2, conc_value, target_value], axis=1, sort=False)

dftest_normed = make_norm_given_lim(dftest, norm_lims, "V", "concentration")





xdftrain, ydftrain, xdftest, ydftest, xtrain, ytrain, xtest, ytest = split_tt(df_normed,100,102,'V','concentration')


xtest=dftest.dropna(axis=0)
xtest=xtest.drop(columns="target")
xtest= xtest.values
xtest.shape
#ytest is empty


dim_output = 1
dim_input = 2
len_train_input = ytrain.shape[0]

mymodel = Sequential([
        Dense(units = 12, input_dim = dim_input, activation='tanh'),
        #Dense(10, activation='tanh'),
        #Dense(5, activation='tanh'),
        Dense(1, activation='linear'),
])
#RMSprop
#Adadelta
#Nadam
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
mymodel.compile(optimizer='RMSprop',
          loss='mse')

start=time.time()
myhist = mymodel.fit(xtrain,ytrain, batch_size=len_train_input, epochs=1000, verbose=0)
end=time.time()
train_time= [end-start]

#Fit the model to our training data


dftest_normed= dftest_normed.drop(["target"],axis=1)
dftest_normed= dftest_normed.values
#Predict target values with our test data
yhat = mymodel.predict(x=dftest_normed)
yhat= pd.DataFrame(yhat,columns=["target"])
yhat_unnormed= make_unnorm(yhat, norm_lims)

yhat_unnormed=np.array(yhat_unnormed)
yhat_unnormed = yhat_unnormed.reshape(len(yhat_unnormed))

ytrain_out = mymodel.predict(x=xtrain)
ytrain=np.array(ytrain)
ytrain_out = ytrain_out.reshape(len(ytrain_out))
    
#entire_predicted= np.concatenate((ytrain_out,yhat))
#entire_true = np.concatenate((ytrain,ytest))

import matplotlib.pyplot as plt

data0=df_original.loc[df_original.concentration == 0]
data1=df_original.loc[df_original.concentration == 1]
data13_4=df_original.loc[df_original.concentration == 13.4]
data30=df_original.loc[df_original.concentration == 30]


data1.index = range(len(data1))
data13_4.index = range(len(data13_4))
data30.index = range(len(data30))


#not really needed
import pickle
pklfile = open(file=exp_str + "/test_concentration" + str(test_concentration) + ".pkl", mode="wb")
toPickle = {'xtrain': xtrain, 'ytrain':ytrain , 'xtest':xtest , 'ytrain_out': ytrain_out, 'ytest':ytest, 'yhat': yhat_unnormed
            ,'norm_lims' : norm_lims, 'loss_iter':myhist.history['loss'],
            'data0':data0,'data1':data1,'data13_4':data13_4,'data30':data30}
pickle.dump(obj = toPickle, file=pklfile)
pklfile.close()


axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([-9,9])

plt.plot(yhat_unnormed, 'm^', label='Predicted Value') 
plt.plot(data0['target'], 'k', label= 'conc=0')
plt.plot(data1['target'], 'r', label= 'conc=1')
plt.plot(data13_4['target'], 'g', label= 'conc=13.4')
plt.plot(data30['target'], 'b', label= 'conc=30')
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([-9,9])
plt.legend()
plt.xlabel('Samples' ,fontsize=20)
plt.ylabel('J(mA/cm^2)', fontsize=20)
plt.suptitle('J Prediction for new conc=' + str(test_concentration),fontsize=20)
plt.savefig(fname=exp_str + "/test_concentration" + str(test_concentration) + ".pdf")
plt.show()



pklfile = open(file=exp_str + "/test_concentration" + str(test_concentration) + ".pkl", mode="rb")
obj=pickle.load(pklfile)
pklfile.close()


# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:11:45 2019

@author: Depp Pc
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random


def plot_true_vs_pred(ytest, yhat, ytrain, ytrain_out, norm_lims, exp_str, j,which_conc_feature, optimizer = ""):
    ytest_df=pd.DataFrame({'target':ytest})
    yhat_df=pd.DataFrame({'target':yhat})
    ytrain_df=pd.DataFrame({'target':ytrain})
    ytrain_out_df=pd.DataFrame({'target':ytrain_out})
    
    ytest_unnormed=make_unnorm(ytest_df,norm_lims)
    yhat_unnormed=make_unnorm(yhat_df,norm_lims)
    ytrain_unnormed=make_unnorm(ytrain_df,norm_lims)
    ytrain_out_unnormed=make_unnorm(ytrain_out_df,norm_lims)
    
    plt.plot(ytest_unnormed, 'bs', label='True Value') 
    plt.plot(yhat_unnormed, 'r^', label='Predicted Value') 
    plt.legend()
    t = np.arange(0, len(ytest_unnormed), 1)
    plt.xticks(t)
    plt.grid(which='both', axis = 'both')
    plt.xlabel('Test Samples',fontsize=20)
    plt.ylabel('J(mA/cm^2)', fontsize=20)
    plt.suptitle('J Prediction of Test Samples',fontsize=20)
    plt.savefig(fname=exp_str + "/figures/test_learned_nh_" + str(j) + which_conc_feature + optimizer + ".pdf")
    plt.show()
    
    #get_ipython().run_line_magic('matplotlib', 'qt')
    plt.plot(ytrain_unnormed, 'bs', label='True Value') 
    plt.plot(ytrain_out_unnormed, 'r^',label='Predicted Value') 
    plt.legend()
    t = np.arange(0, len(ytrain_unnormed), 1)
    plt.xticks(t)
    plt.grid(which='both', axis = 'both')
    plt.xlabel('Training Samples',fontsize=20)
    plt.ylabel('J(mA/cm^2)', fontsize=20)
    plt.suptitle('J Prediction of Training Samples',fontsize=15)
    plt.savefig(fname=exp_str + "/figures/train_learned_nh_" + str(j) + which_conc_feature + optimizer + ".pdf")
    plt.show()


def make_norm(data,*restcol):
    from copy import deepcopy
    data_n = deepcopy(data)
    
    norm_limits = {}
    for colname in restcol:
        meanthis= (np.mean(data[colname]))
        stdthis= (np.std(data[colname]))
        norm_limits[colname] = {'mean': meanthis, 'std': stdthis}
        data_n[colname] = (data[colname] - meanthis)/stdthis
#        maxthis = max(data[colname])
#        minthis = min(data[colname])
        #norm_limits[colname] = {'max':maxthis, 'min':minthis}
        #data_n[colname]=(2*((data[colname]-minthis)/(maxthis-minthis))) - 1;
    return data_n, norm_limits

def make_norm_given_lim(data, norm_lims, *restcol):
    from copy import deepcopy
    data_n = deepcopy(data)
    
    for colname in norm_lims.keys():
        if colname not in data.columns.tolist():
            continue
        else:
            data_n[colname] = (data[colname] - norm_lims[colname]['mean'])/norm_lims[colname]['std']
    return data_n


def make_unnorm(data, normlims):
    from copy import deepcopy
    data_unnorm = deepcopy(data)
    
    for colname in normlims.keys():
        if colname not in data.columns.tolist():
            continue
        else:
            data_unnorm[colname] = data[colname]*normlims[colname]['std'] + normlims[colname]['mean']
            #data_unnorm[colname]= normlims[colname]['min'] + data[colname]*(normlims[colname]['max'] - normlims[colname]['min'])
    return data_unnorm


def my_r2_score(v_true, v_pred):
    ssres = np.sum(np.square(v_true - v_pred))
    sstot = np.sum(np.square(v_true - np.mean(v_true)))
    return 1 - ssres / sstot
    


#np.random.seed(3) ; np.random.rand(len(df)) <0.6
#for j in range(5): 
#    #a = [[0] * len(df)] * j#iteration
#    for i in range(100):
#        np.random.seed(j+i+1); 
#        msk = np.random.rand(len(df)) < 0.7
#        dftrain=df[msk]
#        dftest=df[~msk]
#        if (len(dftrain) == math.trunc(0.7*len(df))):
#            print(msk)
#            break
#
#np.random.rand(len(df)) <0.6
#To split DataFrame into train and test, make a boolean array with prob<0.6 as True 

def split_tt(df_normed, train_percent,iteration=3, *featurename):
    n=train_percent/100    
    dftrain = df_normed.sample(frac =n, random_state=iteration)
    
    # checking if sample is 0.25 times data or not 
    dftrain.sort_index(inplace=True)
        
    dftest=df_normed.drop(dftrain.index)
    actual_perc=(len(dftrain)/len(df_normed))*100
    print(actual_perc)
    
    
#    for m in range(100):
#        n=train_percent/100
#        np.random.seed(iteration+m+1)
#        msk = np.random.rand(len(df_normed)) < n
#        dftrain=df_normed[msk]
#        dftest=df_normed[~msk]
#        if (len(dftrain) == math.trunc(n*len(df_normed))):
#            break
        #Check their length

 #Put all the features in a list (required for the next for loop where target_name is chosen)
    tempp=[]
    for j in featurename:
        temp_list = []
        for k in featurename:
            temp_list.append(k)
        tempp.append(temp_list)    
    #Use the right target for given feature. e.g. if feature13_4, then we want target13_4
    for j in temp_list:
        if j[0:7] == 'feature':
            target_name= 'target' + j[7:]
        else:
            target_name= 'target'
    
    xdftrain=dftrain
    ydftrain=dftrain
    xdftest=dftest
    ydftest=dftest
    xdftrain=xdftrain.filter(items=featurename, axis=1)
    ydftrain=ydftrain.filter(items=[target_name], axis=1)
    xdftest=xdftest.filter(items=featurename,axis=1)
    ydftest=ydftest.filter(items=[target_name],axis=1)
    
    xdftrain=xdftrain.dropna(axis=0)
    ydftrain=ydftrain.dropna(axis=0)
    xdftest=xdftest.dropna(axis=0)
    ydftest=ydftest.dropna(axis=0)
    
    xtrain = xdftrain.values
    xtrain=xtrain.reshape((len(xtrain),len(featurename)))
    
    ytrain = ydftrain.values
    ytrain=ytrain.reshape(len(ytrain),)
    
    xtest = xdftest.values
    xtest=xtest.reshape((len(xtest),len(featurename)))
    
    ytest = ydftest.values
    ytest=ytest.reshape(len(ytest),)
    return xdftrain, ydftrain, xdftest, ydftest, xtrain, ytrain, xtest, ytest,
#xdftrain, ydftrain, xtrain, ytrain, xtest, ytest = split_tt(df,70,30,'wavelength','concentration')










#Check the numpy arrays shape and dimension
#print("Shape of Training feature matrix as array: ",np.shape(xtrain))
#print("Shape of Target array: ",np.shape(ytrain))
#print("Dimension of training data array: ",np.ndim(xtrain))
#print("Dimension of target array: ",np.ndim(ytrain))
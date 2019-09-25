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


def plot_true_vs_pred(ytest, yhat, ytrain, ytrain_out, norm_lims, str_path, 
                      exp_str, j,which_conc_feature, optimizer = ""):
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

    plt.grid(which='both', axis = 'both')
    plt.xlabel('Test Samples',fontsize=20)
    plt.ylabel('T%', fontsize=20)
    plt.suptitle('Transmittance Prediction of Test Samples',fontsize=20)
    plt.savefig(fname=str_path + "/"+ exp_str + "/figures/test_learned_nh_"
                + str(j) + which_conc_feature + optimizer + ".pdf")
    plt.show()

    plt.plot(ytrain_unnormed, 'bs', label='True Value') 
    plt.plot(ytrain_out_unnormed, 'r^',label='Predicted Value') 
    plt.legend()

    plt.grid(which='both', axis = 'both')
    plt.xlabel('Training Samples',fontsize=20)
    plt.ylabel('T%', fontsize=20)
    plt.suptitle('Transmittance Prediction of Training Samples',fontsize=20)
    plt.savefig(fname=str_path + "/"+ exp_str + "/figures/train_learned_nh_"
                + str(j) + which_conc_feature + optimizer + ".pdf")
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

    return data_n, norm_limits

def make_norm_given_lim(data, norm_lims, *restcol):
    from copy import deepcopy
    data_n = deepcopy(data)
    
    for colname in norm_lims.keys():
        if colname not in data.columns.tolist():
            continue
        else:
            data_n[colname] = (data[colname] - 
                  norm_lims[colname]['mean'])/norm_lims[colname]['std']
    return data_n

def make_unnorm(data, normlims):
    from copy import deepcopy
    data_unnorm = deepcopy(data)
    
    for colname in normlims.keys():
        if colname not in data.columns.tolist():
            continue
        else:
            data_unnorm[colname] =(data[colname] + 
                       normlims[colname]['std']) + normlims[colname]['mean']
    return data_unnorm

def my_r2_score(v_true, v_pred):
    ssres = np.sum(np.square(v_true - v_pred))
    sstot = np.sum(np.square(v_true - np.mean(v_true)))
    return 1 - ssres / sstot

def split_tt(df_normed, train_percent,iteration=3, *featurename):
    n=train_percent/100    
    dftrain = df_normed.sample(frac =n, random_state=iteration)
    dftrain.sort_index(inplace=True) 
    dftest=df_normed.drop(dftrain.index)
    actual_perc=(len(dftrain)/len(df_normed))*100
    print(actual_perc)
    tempp=[]
    for j in featurename:
        temp_list = []
        for k in featurename:
            temp_list.append(k)
        tempp.append(temp_list)    

    for j in temp_list:
        if j[0:7] == 'feature':
            target_name= 'target' + j[7:]
        elif j[0:10] == 'wavelength' and len(j)>10:  
            continue
        else:
            target_name= 'target'
    
    
    xdftrain=dftrain
    ydftrain=dftrain
    xdftest=dftest
    ydftest=dftest
    xdftrain=xdftrain.filter(items=featurename, axis=1)
    ydftrain=ydftrain.filter(like=target_name, axis=1)
    xdftest=xdftest.filter(items=featurename,axis=1)
    ydftest=ydftest.filter(like=target_name,axis=1)  
    xdftrain=xdftrain.dropna(axis=1)
    ydftrain=ydftrain.dropna(axis=1)
    xdftest=xdftest.dropna(axis=1)
    ydftest=ydftest.dropna(axis=1) 
    xtrain = xdftrain.values
    xtrain=xtrain.reshape((len(xtrain),len(featurename)))  
    ytrain = ydftrain.values
    ytrain=ytrain.reshape(len(ytrain),)    
    xtest = xdftest.values
    xtest=xtest.reshape((len(xtest),len(featurename)))    
    ytest = ydftest.values
    ytest=ytest.reshape(len(ytest),)
    return xdftrain, ydftrain, xdftest, ydftest, xtrain, ytrain, xtest, ytest

















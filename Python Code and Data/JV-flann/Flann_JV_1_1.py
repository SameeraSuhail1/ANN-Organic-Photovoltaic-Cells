# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:48:24 2019

@author: Depp Pc
"""

import os
exp_str = "dataout/JV1_1"
import pandas as pd
import numpy as np
import time
import scipy.stats
from numpy.random import seed
from tensorflow import set_random_seed

from utility_flann import make_norm, make_unnorm, my_r2_score, split_tt, plot_true_vs_pred
#Read data file
df = pd.read_csv("fig7acsv.csv")
import math

df_normed, norm_lims = make_norm(df, *df.columns.tolist())



which_conc_feature= "feature13_4"
concentration= which_conc_feature[7:]

df_normed=df_normed.loc[:, df_normed.columns.str.endswith(concentration)]
df_normed=df_normed.dropna(axis=0)

which_target = 'target' + which_conc_feature[7:]
df_normed = df_normed[np.isfinite(df_normed[which_conc_feature])]
df_normed = df_normed[np.isfinite(df_normed[which_target])]

def which_flann(df, flann_polynomial):
    if flann_polynomial=='chebyshev':
        str_path="chebyshev"
        df['V2']=1
        df['V3']=2*df[which_conc_feature]**2-1
        df['V4']=2*df[which_conc_feature]**3-3*df[which_conc_feature]
        df['V5']=8*df[which_conc_feature]**4-8*df[which_conc_feature]**2+1
        
    elif flann_polynomial=='legendre':
        str_path="legendre"
        df['V2']=1
        df['V3']=(3*df[which_conc_feature]**2-1)/2
        df['V4']=(5*df[which_conc_feature]**3-3*df[which_conc_feature])/2
        df['V5']=(35*df[which_conc_feature]**4-30*df[which_conc_feature]**2+3)/8

        
    elif flann_polynomial=='trigonometric':
        str_path="trigonometric"
        df['V2']=np.cos(df[which_conc_feature])
        df['V3']=np.sin(df[which_conc_feature])
        df['V4']=np.cos(2*math.pi*df[which_conc_feature])
        df['V5']=np.sin(2*math.pi*df[which_conc_feature])
     
    elif flann_polynomial=='power':
        str_path="power"
        df['V2']=1
        df['V3']=df[which_conc_feature]**2
        df['V4']=df[which_conc_feature]**3
        df['V5']=df[which_conc_feature]**4

    else:
        print("invalid flann. check spelling")
    return str_path, df

        

str_path, df_normed_expanded = which_flann(df_normed, 'power')
#e.g. power/dataout/JV1_1
os.makedirs(str_path + "/"+ exp_str +"/figures", exist_ok=True)




feature_list=[which_conc_feature ,'V2','V3','V4','V5']
xdftrain, ydftrain, xdftest, ydftest, xtrain, ytrain, xtest, ytest  = split_tt(df_normed_expanded,70,102, *feature_list)

print("Shape of Training feature matrix as array: ",np.shape(xtrain))
print("Shape of Target array: ",np.shape(ytrain))
print("Dimension of training data array: ",np.ndim(xtrain))
print("Dimension of target array: ",np.ndim(ytrain))
#import multi-layer perceptron regressor from sklearn
from keras.models import Sequential
from keras.layers import Dense, Activation
import sklearn.metrics as skm


dim_output = 1
dim_input = 5
len_train_input = ytrain.shape[0]


#import multi-layer perceptron regressor from sklearn
performance_out=pd.DataFrame()

for j in range(1):
    num_hidden_neuron= j+1
    #make a model with specified hyper parameters
    mymodel = Sequential([
            Dense(units = num_hidden_neuron, input_dim = dim_input, activation='tanh'),
            Dense(1, activation='linear'),
    ])
    
    mymodel.compile(optimizer='RMSprop',
              loss='mse')

    xdftrain, ydftrain, xdftest, ydftest, xtrain, ytrain, xtest, ytest  = split_tt(df_normed_expanded,70,102,*feature_list)
    #-------------seed------------------
    seed(102)
    set_random_seed(102)
    #-----------------------------------
    #Fit the model to our training data
    start=time.time()
    myhist = mymodel.fit(xtrain,ytrain, batch_size=len_train_input, epochs=500, verbose=0)
    end=time.time()
    train_time= [end-start]
    
    
    
    #Predict target values with our test data
    yhat = mymodel.predict(x=xtest)
    yhat = yhat.reshape(len(yhat))
    
    ytrain_out = mymodel.predict(x=xtrain)
    ytrain_out = ytrain_out.reshape(len(ytrain_out))
    
    
    
        
        
    entire_predicted= np.concatenate((ytrain_out,yhat))
    entire_true = np.concatenate((ytrain,ytest))
    
    #Calculate train, test and overall mean squared error
    test_mse = [skm.mean_squared_error(ytest,yhat)]
    train_mse= [skm.mean_squared_error(ytrain,ytrain_out)]
    overall_mse= [skm.mean_squared_error(entire_true,entire_predicted)]
    
    test_mae= [skm.mean_absolute_error(ytest,yhat)]
    
    myRsq = [my_r2_score(v_true=ytest, v_pred=yhat)]
    myRsq_training = [my_r2_score(v_true=ytrain, v_pred=ytrain_out)]
    
    no_iter=myhist.epoch[-1]+1
    useless_modelcount=0
    useless_modelcount_training=0
    
    if myRsq[0] < 0:
        useless_modelcount= useless_modelcount + 1
    if myRsq_training[0] < 0:
        useless_modelcount_training= useless_modelcount_training + 1
    
    n_ensemble=100
    for i in range(n_ensemble-1):
        xdftrain, ydftrain, xdftest, ydftest, xtrain, ytrain, xtest, ytest = split_tt(df_normed_expanded,70,102, *feature_list)
        #-------------seed------------------
        seed(i+1)
        set_random_seed(i+1)
        #-----------------------------------
        start=time.time()
        myhist = mymodel.fit(xtrain,ytrain, batch_size=len_train_input, epochs=500, verbose=0) # this has hardcoded epochs. For stopping on loss tolerance see https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
        end=time.time()
        print("Training time is:", round(end-start,3))
        train_time= train_time + [(end-start)]
        no_iter = no_iter + (myhist.epoch[-1]+1)
        print("The number of iterations ran was: ",myhist.epoch[-1]+1)
        yhat = mymodel.predict(xtest); yhat = yhat.reshape(len(yhat))
        ytrain_out= mymodel.predict(xtrain); ytrain_out = ytrain_out.reshape(len(ytrain_out))
        entire_predicted= np.concatenate((ytrain_out,yhat))
        entire_true = np.concatenate((ytrain,ytest))
        #performance metrics
            #MSE
        test_mse= test_mse + [skm.mean_squared_error(ytest,yhat)]
        train_mse= train_mse + [skm.mean_squared_error(ytrain,ytrain_out)]
        overall_mse= overall_mse + [skm.mean_squared_error(entire_true,entire_predicted)]
        #MAE
        test_mae= test_mae + [skm.mean_absolute_error(ytest,yhat)]
        #R squared
        current_Rsq = my_r2_score(v_pred=yhat, v_true=ytest)
        current_Rsq_training = my_r2_score(v_pred=ytrain_out, v_true=ytrain)
        
#        if (current_Rsq_training < 0):
#            import pdb; pdb.set_trace()
        myRsq = myRsq + [current_Rsq]
        myRsq_training = myRsq_training + [current_Rsq_training]
        
        if current_Rsq < 0:
            useless_modelcount= useless_modelcount + 1
        if current_Rsq_training < 0:
            useless_modelcount_training= useless_modelcount_training + 1

            #print(i,' mse ',round(mymse,4),' fmse ',round(myfmse,4))
    
    # calculate means and standard error of the means of the various measures
    avg_train_time_mean = np.mean(train_time)
    avg_train_time_sem = scipy.stats.sem(train_time)
    

    test_fmse_mean = np.mean(test_mse)             #find average mse over 100 ensembles
    test_fmse_sem = scipy.stats.sem(test_mse)
    train_fmse_mean = np.mean(train_mse)
    train_fmse_sem = scipy.stats.sem(train_mse)
    
    overall_fmse_mean = np.mean(overall_mse)
    overall_fmse_sem = scipy.stats.sem(overall_mse)
    
    test_fmae_mean = np.mean(test_mae)
    test_fmae_sem = scipy.stats.sem(test_mae)
    
    my_fRsq_mean = np.mean(myRsq)
    my_fRsq_sem = scipy.stats.sem(myRsq)
    
    my_fRsq_training_mean = np.mean(myRsq_training)
    my_fRsq_training_sem = scipy.stats.sem(myRsq_training)
    

    overall_MSEdB= round(10*math.log10(overall_fmse_mean),3)
    test_MSEdB= round(10*math.log10(test_fmse_mean),3)
    train_MSEdB= round(10*math.log10(train_fmse_mean),3)
    
    print("")
    print("number of models with negative R squared (test data)", useless_modelcount)
    print("")
    print("number of models with negative R squared (training data)", useless_modelcount_training)
    print("")
    print("average mean squared error of entire dataset", round(10*math.log10(overall_fmse_mean),3),"dB")
    print ("average mean squared error of testing set is: ", round(10*math.log10(test_fmse_mean),3),"dB") 
    print("average mean squared error for training set",round(10*math.log10(train_fmse_mean),3),"dB")
    
    print ("average mean absolute error of testing set is: ", round(10*math.log10(test_fmae_mean),3),"dB") 
    
    print("average R squared (test data) for the model is: ", round(my_fRsq_mean,4))
    print("")
    
    print("average R squared (training data) for the model is: ", round(my_fRsq_training_mean,4))
    print("")

    no_fiter=no_iter/n_ensemble # this is meaningless now since we're not stopping on tolerance
    print("avg iterations",no_fiter)
    print("")
    # intialise data of lists. 
    temp = pd.DataFrame({'Nh':j, 'Overall dB': overall_MSEdB, 'Test dB': test_MSEdB, 'Train dB': train_MSEdB,
                         'Overall MSE mean': overall_fmse_mean, 'Test MSE mean': test_fmse_mean, 'Train MSE mean': train_fmse_mean,
                         'Overall MSE sem': overall_fmse_sem, 'Test MSE sem': test_fmse_sem, 'Train MSE sem': train_fmse_sem,
                         'Avg. R Squared (test) mean': my_fRsq_mean,'Avg. R Squared (training) mean': my_fRsq_training_mean,
                         'Avg. R Squared (test) sem': my_fRsq_sem, 'Avg. R Squared (training) sem': my_fRsq_training_sem,
                         'Avg. Train Time mean':avg_train_time_mean, 'Avg. Train Time sem':avg_train_time_sem, 'Avg. no of iterations':no_fiter, 
                         'No. of useless models (test)': useless_modelcount,
                         'No. of useless models (training)': useless_modelcount_training}, index=[0])
    performance_out = pd.concat([performance_out, temp])
    #performance_out.append({'Nh':j, 'Overall': overall_MSEdB, 'Test': test_MSEdB, 'Train': train_MSEdB,'Avg. R Squared': round(my_fRsq,4),'Avg. Train Time':avg_train_time, 'Avg. no of iterations':no_fiter, 'No. of useless models': useless_modelcount },ignore_index=True)
    
    #Find the final yhat
    #yhat=myfmodel.predict(xtest)
    import pickle 

    pklfile = open(file=str_path + "/"+ exp_str + "/" + which_conc_feature + 'FLANN_JV_1_1.pkl', mode="wb")
    toPickle = {'xtrain': xtrain, 'ytrain':ytrain ,'ytrain_out': ytrain_out, 'xtest':xtest , 'ytest':ytest , 'yhat':yhat, 'entire_predicted':entire_predicted , 'entire_true': entire_true
                ,'norm_lims':norm_lims, 'test_mse':test_mse , 'train_mse':train_mse , 'overall_mse':overall_mse , 'test_mae': test_mae, 'current_Rsq': current_Rsq 
                ,'current_Rsq_training':current_Rsq_training, 'loss_iter':myhist.history['loss'] }
    pickle.dump(obj = toPickle, file=pklfile)
    pklfile.close()
    
    
    
    #Percentage errors in each test data sample
    perror = 100.0*(np.abs(ytest-yhat)/ytest)
    #find maximum percentage error
    maxperror = np.max(100.0*(np.abs(ytest-yhat)/ytest))
    print("Maximum percet error = ",maxperror)

    plot_true_vs_pred(ytest,yhat,ytrain,ytrain_out, norm_lims,str_path, exp_str,j,which_conc_feature, "")
    
#exp_str = "dataout/exp1_1"
#str_path="power"
#"D:\python\SampleANN\JV_FLANN"
os.makedirs(str_path + "/"+ exp_str, exist_ok=True)
export_csv = performance_out.to_csv(str_path + "/"+ exp_str +"/"+  r'FLANN_performances_1_1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
#performance will be stored in 'power/dataout/exp1_1/FLANN_performances_1_1.csv'


#str_path="power"
pklfile = open(file=str_path + "/"+ exp_str + "/" + which_conc_feature + 'FLANN_JV_1_1.pkl', mode="rb")
obj = pickle.load(pklfile)
pklfile.close()



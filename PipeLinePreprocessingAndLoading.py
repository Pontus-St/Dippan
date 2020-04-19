# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:03:05 2020

@author: Pontus
"""

from DataHolder import DataHolder
import smallDataset
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np



def normalize(X_train,X_test):
    normalizer = StandardScaler()
    
    Resized = False
    n_Train,x_shape,y_shape,z_shape = X_train.shape
    n_Test ,x_shape,y_shape,z_shape = X_test.shape
    if len(X_train.shape) > 2:
        X_train = resizeForNormalization(X_train)
        X_test = resizeForNormalization(X_test)
        Resized = True
        
    normalizer.fit(X_train)

    X_train=normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)
    
    if Resized:
        X_train = resizeFromNormalization(X_train,n_Train,x_shape,y_shape,z_shape)
        X_test = resizeFromNormalization(X_test,n_Test,x_shape,y_shape,z_shape)
    return X_train , X_test

def loadDataSet(useBrainData):
    
    
    if useBrainData: 
        DH = DataHolder("C:/Users/Pontus/Desktop/Dippa/dataset") # delning med / ska användas 
        subs = list(DH.subjects.keys())
        D,i = DH.subjects.get(subs[3]).getDataAndInfoForSubject()
        DS = smallDataset.SmallDataset(D,i)
        
        
    else:
        print("downloading data...")
        X_d, y_d = fetch_openml('mnist_784', version=1, return_X_y=True)
        randInd = np.random.choice(np.arange(0,69000,1),(1000))
    
        X = X_d[randInd,:]
        y = y_d[randInd]
        X = X/255
        X = np.reshape(X,(1000,28,28,1))
     
        DS = smallDataset.SmallDataset(X,y)
    
    
    return DS



def resizeForNormalization(X):
    if len(X.shape) > 2:
        n_shape,x_shape,y_shape,z_shape = X.shape
        print(type(X))
        X = np.reshape(X,(n_shape,x_shape*y_shape*z_shape))
    return X


def resizeFromNormalization(X,n_shape,x_shape,y_shape,z_shape):
    
    X = np.reshape(X,( n_shape,x_shape,y_shape,z_shape))
    return X
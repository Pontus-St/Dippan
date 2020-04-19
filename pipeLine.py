# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:21:45 2020

@author: Pontus
"""


import torch.utils.data 
from torch.utils.data import DataLoader
from DataHolder import DataHolder
import smallDataset
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import DRmethods 
import TestEvaluvationTools

import Networks
#%%
'''
Construct the dataSet

'''
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

def plotLearntClasses(DR,DRY):
    col = DRY
    plt.scatter(DR[:,1],DR[:,0],c = col/10)
    plt.colorbar()
    plt.show()

def InitSiamase(digitData):
    net = Networks.Network("siamase",digitData)
    return net

def trainSiamase(net,DS):
    net.train_custom(DS)
    return net

def useSiamase(DS_Train,digitData):
    sia_net = InitSiamase(digitData)
    print("training sia")
    sia_net = trainSiamase(sia_net,DS_Train)
    #%%
    saveString = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    print("saving model to: " , saveString)
    torch.save(sia_net,saveString)
    
    return sia_net

def useTriplet(DS_Train,digitData):
    tripnet = Networks.Network("triplet",digitData)
    print("training triplet")
    tripnet.train_custom(DS_Train)
    saveString = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    saveString = "triplet" + saveString
    print("saving model to: " , saveString)
    torch.save(tripnet,saveString)
    return tripnet


def resizeForNormalization(X):
    if len(X.shape) > 2:
        n_shape,x_shape,y_shape,z_shape = X.shape
        print(type(X))
        X = np.reshape(X,(n_shape,x_shape*y_shape*z_shape))
    return X


def resizeFromNormalization(X,n_shape,x_shape,y_shape,z_shape):
    
    X = np.reshape(X,( n_shape,x_shape,y_shape,z_shape))
    return X


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

#%%
'''
=========
param and load data
'''
useBrainData = True
DS = loadDataSet(useBrainData)

X_train, data_train, X_test, data_test = DS.split(int(len(DS)*0.8))

#%%
'''load network'''

loadNetworkSiapath = "DEBUGsiamase2020-04-19-19-00-38" 

loadNetworkTripletpath = "DEBUGtriplet2020-04-19-18-59-53" 
#normalize 
X_train , X_test = normalize(X_train,X_test)

print("datacollected")
DS_Train = smallDataset.SmallDataset(X_train,data_train)
DS_Test = smallDataset.SmallDataset(X_test,data_test)
print("test and train set constructed")


sia_net = torch.load(loadNetworkSiapath)
sia_net.eval()
 

triplet_net = torch.load(loadNetworkTripletpath)
triplet_net.eval()
#%%
DS_Train.setNormal()
PCA = DRmethods.getPCA()
NCA = DRmethods.getNCA()
LDA = DRmethods.getLDA()
UMAP = DRmethods.getUmap()
tSNE = DRmethods.getTSNE()
#models = [("Siamese",sia_net),("PCA",PCA),("NCA",NCA),("LDA",LDA),("UMAP",UMAP),("tSNE",tSNE)]
#models = [("triplet",triplet_net),("PCA",PCA),("NCA",NCA),("LDA",LDA),("UMAP",UMAP),("tSNE",tSNE)]
models = [("Siamese",sia_net),("triplet",triplet_net),("PCA",PCA),("NCA",NCA),("LDA",LDA),("UMAP",UMAP),("tSNE",tSNE)]
#%%
DREmbbeddigns = []
DREmbbeddignsTrained = []

for method in models:
    print("Transformin for model " , method[0])
    if method[0] == "Siamese" or method[0] == "triplet" :
        
        drx,dry = method[1].getAllDRCoordinatesFromDataset(DS_Test)
        DREmbbeddigns.append(drx)
        drx,dry = method[1].getAllDRCoordinatesFromDataset(DS_Train)
        DREmbbeddignsTrained.append(drx)
        
    elif method[0] == "UMAP":
        
        method[1].fit(resizeForNormalization(DS_Train.ORGx),DS_Train.ORGy)
        buffer = method[1].transform(resizeForNormalization(DS_Test.ORGx))
        DREmbbeddigns.append(buffer)
        buffer = method[1].transform(resizeForNormalization(DS_Train.ORGx))
        DREmbbeddignsTrained.append(buffer)
        
    elif method[0] == "tSNE":
        
        buffer = method[1].fit_transform(resizeForNormalization(DS_Test.ORGx))
        DREmbbeddigns.append(buffer)
        buffer = method[1].fit_transform(resizeForNormalization(DS_Train.ORGx))
        DREmbbeddignsTrained.append(buffer)
        
    else:
        method[1].fit(resizeForNormalization(DS_Train.ORGx),DS_Train.ORGy)
        buffer = method[1].transform(resizeForNormalization(DS_Test.ORGx))
        DREmbbeddigns.append(buffer)
        buffer = method[1].transform(resizeForNormalization(DS_Train.ORGx))
        DREmbbeddignsTrained.append(buffer)
        
    
#%%
#print(DS_Test.info.columns)
#%%
numberOfResults = len(DREmbbeddigns)+1

fig,axs = plt.subplots(2,int(np.ceil(numberOfResults/2)))
i = 0
print(axs)
for res in DREmbbeddigns:
    print("sess_num:",models[i][0])
    #c  = pd.factorize( DS_Test.info['seq_train'])[0] 
    c  = DS_Test.ORGy.astype(int)/10
    axs[i%2,int(np.floor(i/2))].scatter(res[:, 0], res[:, 1], c=c)  
    axs[i%2,int(np.floor(i/2))].set_title(models[i][0])
    
    plt.show()
    #(dataOrg,dataDR,label,ContAndThrustNumOfNeigh,trainData,trainLabel):
    sil,cont,thrust,acc = TestEvaluvationTools.evaluateDR(resizeForNormalization(DS_Test.ORGx),res,DS_Test.ORGy.astype(int),10,DREmbbeddignsTrained[i],DS_Train.ORGy)
    i += 1
    print("sil:{},cont:{} ,thrust:{},acc{}".format(sil,cont,thrust,acc))
#%%
'''
print(DREmbbeddigns[-1].shape)
drx,dry = triplet_net.getAllDRCoordinatesFromDataset(DS_Train)
res = drx
c  = DS_Train.ORGy.astype(int)/10
#c  = pd.factorize( DS_Train.info['sess_num'])[0] 
plt.figure()
plt.scatter(res[:, 0], res[:, 1], c=c)
plt.show()

'''
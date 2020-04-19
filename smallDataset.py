# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:33:29 2020

@author: Pontus
"""

from torch.utils.data import Dataset, DataLoader
from DataHolder import DataHolder
import pandas as pd
from DataHolder import DataHolder
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import sleep
import torch
#%%
class SmallDataset(Dataset):

    def __init__(self,data,info):
        
        self.ORGx = data
        self.info = info
        if type(info) == type(pd.core.frame.DataFrame()):
            self.ORGy = pd.factorize( self.info['true_sequence'])[0] 
        else:
            print("nope")
            self.ORGy = info
        self.type = "normal"
        
        self.ClassBuckets = []
    def setAsPairwise(self):
        nf = np.math.factorial( len(self.ORGy)) 
        kf = np.math.factorial(2)
        numberOfPairs = int(nf/(kf*np.math.factorial(len(self.ORGy)-2)))
        #print(numberOfPairs)
        
        self.pairIndex = np.zeros((2,int(numberOfPairs)))
        self.pairLabel = np.zeros((int(numberOfPairs)))
        counter = 0
        
        self.posExamples = np.zeros((int(numberOfPairs))) 
        posCounter = 0
        
        self.negExamples = np.zeros((int(numberOfPairs))) 
        negCounter = 0
        
        for i in range(len(self.ORGy)):
            for j in range(i+1,len(self.ORGy)):
                self.pairIndex[0,counter] = i
                self.pairIndex[1,counter] = j
                if self.ORGy[i] == self.ORGy[j]:
                    self.pairLabel[counter] = 0
                    self.posExamples[posCounter] = counter
                    posCounter +=1
                else:
                    self.pairLabel[counter] = 1
                    self.negExamples[negCounter] = counter
                    negCounter +=1
                    
                counter +=1
        self.posExamples = self.posExamples[0:posCounter]
        self.negExamples = self.negExamples[0:negCounter]
        
        #print("len pre ",len( self.negExamples))
        
        #balanced list, drop some negativity 
        self.negExamples = np.random.choice(self.negExamples,size = np.size(self.posExamples),replace = False)
        
        self.pairIndexBalanced = np.append(self.posExamples,self.negExamples)
        np.random.shuffle(self.pairIndexBalanced)
        
        self.pairIndexBalanced =self.pairIndexBalanced.astype(int)
        self.pairIndex = self.pairIndex.astype(int)
        self.pairLabel = self.pairLabel.astype(int)
        #print(len(self.pairIndexBalanced))
        
        self.type = "pairWise"
        #print(pairIndex[:,numberOfPairs-10:numberOfPairs-0])
        #print(pairLabel[numberOfPairs-10:numberOfPairs-0])
        #print("len post ",len( self.negExamples))
        #print("len post ",len( self.posExamples))
    def getTripletLenght(self):
        return 50000
    def setAsTriplet(self):
        if self.type != "triplet":
            self.type = "triplet"
        
            maxClass = len(np.unique(self.ORGy))
            print("max:",maxClass)
            for i in range(maxClass):
                buffer = []
                self.ClassBuckets.append(buffer)
                
            #add index of data to correct BUCket! 
            for i in range(len(self.ORGy)):
                self.ClassBuckets[int(self.ORGy[i])].append(i)
        
            
    def getTripletIndex(self):
        #randomly select classes
        [classRef, classNeg] = np.random.choice(len(self.ClassBuckets),2,replace=False )
        #from the referens class choose a ref datapoint and a positiv match
        [index_ref,indexPos] =  np.random.choice(self.ClassBuckets[classRef],2,replace=False )
        #from neg class get one 
        [indexNeg]  =  np.random.choice(self.ClassBuckets[classNeg],1,replace=False )
        
        return index_ref , indexPos, indexNeg
        
        print(classRef,classNeg, index_ref,indexPos,indexNeg)
    def getpairWiseIndex(self,ind):
        labelIndex = self.pairIndexBalanced[ind]
        
        dataIndex1 =self.pairIndex[0,labelIndex]
        
        dataIndex2 =self.pairIndex[1,labelIndex]
        
        return dataIndex1,dataIndex2,labelIndex
    
    def __len__(self):
        if self.type == "normal":
            return len(self.ORGy)
        elif self.type == "pairWise":
            return 2*len(self.posExamples)
        elif self.type == "triplet":
            return self.getTripletLenght() 
        else:
            print("no lenght type specified in small dataset")
            return 0
   

    def __getitem__(self,ind):
        #print("ind;:",ind)
        if self.type == "normal":
            return np.expand_dims(self.ORGx[ind,:,:,:],0) , self.ORGy[ind]
        elif self.type == "pairWise":
            dataIndex1,dataIndex2,labelIndex = self.getpairWiseIndex(ind)
            return np.expand_dims(self.ORGx[dataIndex1,:,:,:],0),np.expand_dims(self.ORGx[dataIndex2,:,:,:],0), self.pairLabel[labelIndex]
        
        elif self.type == "triplet":
            index_ref , indexPos, indexNeg = self.getTripletIndex()
            x_ref =  np.expand_dims(self.ORGx[index_ref,:,:,:],0)
            x_pos =  np.expand_dims(self.ORGx[indexPos,:,:,:],0)
            x_neg =  np.expand_dims(self.ORGx[indexNeg,:,:,:],0)
            
            return x_ref, x_pos, x_neg
        else:
            return 0 
        
    def  setNormal(self):
        self.type = "normal"
        
    def  split(self,lenTrain):
        indexForSplit = np.random.choice(np.arange(0,len(self.ORGy)),size = lenTrain,replace = False)
        print(len(self)," " ,len(indexForSplit))
        X_train = self.ORGx[indexForSplit]
        
        indexRemaining = []
        for i in range(len(self)):
            if not i in indexForSplit:
                indexRemaining.append(i)
        X_test =  self.ORGx[indexRemaining]
        
        if type(self.info) == type(pd.core.frame.DataFrame()):
            data_test = self.info.iloc[indexRemaining]
            data_train = self.info.iloc[indexForSplit]
        else:
            data_test = self.ORGy[indexRemaining]
            data_train = self.ORGy[indexForSplit]
        #print(X_train)
        return X_train, data_train , X_test, data_test
#%%
if __name__ == "__main__" :
    pass
    #DH = DataHolder("C:/Users/Pontus/Desktop/Dippa/dataset")
    #subs = list(DH.subjects.keys())
    #D,i = DH.subjects.get(subs[3]).getDataAndInfoForSubject()
    
    #dataset = SmallDataset(D,i)
    #dataset.setAsPairwise()
    #print(len(dataset))
    #x1,x2,y  = dataset[0:10]
    #print(y)
    #%%
    #print(np.shape(D))
    #print(D[0,0,0,0])
    #print(np.expand_dims(D,1).shape)
    #print(type(i))
    #print(type(pd.core.frame.DataFrame()))
    X_d, y_d = fetch_openml('mnist_784', version=1, return_X_y=True)
    #%%
    randInd = np.random.choice(np.arange(0,69000,1),(1000))
    print(randInd)
    #%%
    X = X_d[randInd,:]
    y = y_d[randInd]
    
    #print(np.max(X))
    #print(X.shape," ", y.shape)
    X = np.reshape(X,(1000,28,28,1))
    
    dataset2 = SmallDataset(X,y)
    
    #%%
    dataset2.setAsTriplet()
    
    dataloader = DataLoader(dataset2, batch_size=1, shuffle=False)
    
    for i, (x1,x2,x3) in enumerate(dataloader):
        plt.subplot(3,1,1)
        plt.imshow(x1[0,0,:,:,0])
        plt.subplot(3,1,2)
        plt.imshow(x2[0,0,:,:,0])
        plt.subplot(3,1,3)
        plt.imshow(x3[0,0,:,:,0])
        break

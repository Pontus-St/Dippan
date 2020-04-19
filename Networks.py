# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from DataHolder import DataHolder

from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules.distance import PairwiseDistance
import smallDataset
import pandas as pd
from sklearn.datasets import fetch_openml

class Network(nn.Module):
    def __init__(self,networkType,Digits):
        
        super(Network, self).__init__()
        
        if Digits:
            paddingParameter =2
        else:
            paddingParameter = 0
        
        
        self.BaseNetwork = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3,padding=paddingParameter),
            nn.ReLU(),
            nn.Conv3d(16,16,kernel_size = 3,padding=paddingParameter),
            nn.MaxPool3d(3,1),
            nn.BatchNorm3d(16, affine=False),
            nn.ReLU(),
            
            nn.Conv3d(16,32,kernel_size = 3,padding=paddingParameter),
            nn.ReLU(),
            nn.Conv3d(32,32,kernel_size = 3,padding=paddingParameter),
            nn.MaxPool3d(3,2),
            nn.BatchNorm3d(32, affine=False),
            nn.ReLU(),
            
            nn.Conv3d(32,16,kernel_size = 3,padding=paddingParameter),
            nn.MaxPool3d(3,2),
            nn.BatchNorm3d(16, affine=False),
            nn.ReLU(),
            
        
            
            nn.Flatten(),
            nn.Linear(1088,1000),
            nn.BatchNorm1d(1000, affine=False),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.BatchNorm1d(100, affine=False),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.BatchNorm1d(50, affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(50,2),

        )
        self.pD = PairwiseDistance(2)
        
        self.networkType =networkType
        
        self.tripletSoftMax = nn.Softmax(dim = 1)

    def forwardHelper(self, x):
        x = self.BaseNetwork.forward(x)
        return x
    
        
    def getDR(self,x):
        return self.forwardHelper(x)
    
    def forwardSiamase(self,x1,x2):
        g1 = self.forwardHelper(x1)
        g2 =self.forwardHelper(x2)
        

        diff = self.pD.forward(g1,g2)

        return diff

    def forwardTriplet(self,xRef,xPos,xNeg):
        eRef = self.forwardHelper(xRef)
        ePos = self.forwardHelper(xPos)
        eNeg = self.forwardHelper(xNeg)
        
        diffPos = self.pD.forward(eRef,ePos)
        diffNeg = self.pD.forward(eRef,eNeg)
        
        #print(type(diffPos))
        return torch.t(torch.stack((diffPos,diffNeg)))

    
    def lossSiamase(self,out,Y):
        margin =2 
        return  torch.mean( (1-Y)*torch.max(torch.pow( out,2),torch.abs(out))  +  (Y)*torch.pow( torch.clamp(margin-out,min=0.0),2))

    def loss_triplet(self,out):
        #print(out.shape)
        out=self.tripletSoftMax(out)
        toCompare = torch.tensor([[0.], [1]],requires_grad = True)
        toCompare = torch.t(toCompare.expand(2,out.shape[0]))
        #print(toCompare)
        toCompare= Variable(toCompare.float().cuda())
        #print(out-toCompare)
        buffer = out-toCompare 
        zeroes = torch.zeros_like(buffer[:,1])

            
        buffer[:,1] = torch.where(buffer[:,1] > 0, zeroes,  buffer[:,1])
        #print(buffer)
        out = torch.mean(torch.pow(buffer,2))
        return out
    def getLossSiamase(self,data):
        (x1,x2,y) = data
        x1 = Variable(x1.float().cuda())
        x2 = Variable(x2.float().cuda())
        y = Variable(y.cuda())
        
        out = self.forwardSiamase( x1, x2)
        out = out.squeeze()
        y = y.squeeze().float()
        lossValue = self.lossSiamase(out, y)
        
        return lossValue
    
    def getLossTriplet(self,data):
        ( x_ref, x_pos, x_neg) = data
        x_ref = Variable(x_ref.float().cuda())
        x_pos = Variable(x_pos.float().cuda())
        x_neg = Variable(x_neg.float().cuda())
        

        out= self.forwardTriplet(x_ref,x_pos,x_neg)

        lossValue = self.loss_triplet(out)
        return lossValue
    
    def getloss(self,data):
        if self.networkType == "siamase": 
                lossValue = self.getLossSiamase(data)
        elif self.networkType == "triplet": 
                lossValue = self.getLossTriplet(data)
        
        return lossValue
    @torch.no_grad()
    def getValLoss(self,DS_val,margin):
        
        self.eval()
        dataloader = DataLoader(DS_val, batch_size=16, shuffle=True)
        lossValue = 0
        maxInter = 10000
        for i, data in enumerate(dataloader):
            if self.networkType == "siamase": 
                lossValue = self.getLossSiamase(data)
            if i > maxInter:
                break
        self.train()
        return lossValue/(i*1)
    
    
    def train_custom(self,DS,parameters):
        #split train val
        X_train, data_train, X_val, data_val = DS.split(int(len(DS)*0.9))
        DS = smallDataset.SmallDataset(X_train,data_train)
        DS_val = smallDataset.SmallDataset(X_val,data_val)
        if self.networkType == "siamase": 
                DS.setAsPairwise()
                DS_val.setAsPairwise()
        elif self.networkType == "triplet": 
                DS.setAsTriplet()
                DS_val.setAsTriplet()
        
        #parameters
        epochs=parameters.get("epochs")
        
        if "maxIterations" in parameters:
            maxIter =parameters.get("maxIterations")
        else:
            maxIter =len(DS)
            
        learningRate =parameters.get("lr")
        
        print("init training for {}, epoch {}, lenght {}, lr {}".format(self.networkType,epochs,maxIter,learningRate) )
        
        #utility
        torch.cuda.empty_cache()
        self.cuda()
        self.train()
        batch_size = 16
        optimizer = torch.optim.Adam(self.parameters(),lr = learningRate )
        dataloader = DataLoader(DS, batch_size=batch_size, shuffle=True)
        
        #loss plotting
        intervall = 100
        lossGraph = np.zeros(int(np.ceil(maxIter/intervall)))
        buffer=0
        
        #validation set        
        ValpreformingworseCounter = 0
        bestVal = np.inf
        Valpreformingworseallowed = 20
        currentValLoss = 0
        currentbatchLoss=0
        
        for e in range(epochs):
            for i, data in enumerate(dataloader):
                optimizer.zero_grad()
                
                
                lossValue = self.getloss(data)
                
                lossValue.backward()
                
                optimizer.step()
                buffer += lossValue.item()

                
                if i % intervall == 0 and i != 0: 
                    lossGraph[int(np.floor(i/intervall))] = buffer / intervall
                    currentbatchLoss = buffer / intervall
                    print(i*batch_size,"/", maxIter," loss",buffer / intervall)
                    buffer=0
                    if e >= 4:
                        
                        with torch.no_grad():
                            currentValLoss = self.getValLoss(DS_val,2)
                        
                        
                        if currentValLoss > bestVal:
                            ValpreformingworseCounter +=1
                            
                        else:
                            ValpreformingworseCounter = 0
                            bestVal = currentValLoss
                        if ValpreformingworseCounter > Valpreformingworseallowed:
                            
                            break
                        print("currentValLoss " ,currentValLoss, "  ValpreformingworseCounter ",ValpreformingworseCounter)
                if i*batch_size >= maxIter :
                    break
                    pass 
                    
            if ValpreformingworseCounter > Valpreformingworseallowed:
                    break
        
        plt.subplot(1,2, 1)
        plt.plot(lossGraph)
        plt.show()
        self.eval()
        return currentValLoss , currentbatchLoss
    def getAllDRCoordinatesFromDataset(self,DS):
        DS.setNormal()
        maxViz = len(DS)
        DR = np.zeros((maxViz,2))
        DR_y =np.zeros((maxViz))

        
        dataloader_notshuffle = DataLoader(DS, batch_size=1, shuffle=False)
        counter = 0
        for i, (x1,y) in enumerate(dataloader_notshuffle):
            #print(i)
            #print(x1.shape)
            x1 = Variable(x1.cuda())
        
            X1 = self.getDR(x1.float())
            #print(X1.shape)
            DR[i*1:(i+1)*1,:] = X1.detach().cpu().numpy()
        
            DR_y[i*1:(i+1)*1] = y
            counter += 1
        return DR,DR_y

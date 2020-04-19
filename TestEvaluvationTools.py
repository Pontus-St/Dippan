# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:01:40 2020

@author: Pontus
"""

import sklearn
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

import sklearn.manifold  





def getSilhouettScore(data,label,metricType = 'euclidean'):
    buffer = sklearn.metrics.silhouette_score(data, label, metric=metricType)
    return buffer

def getKNN(data,label,numOfNeighbours,testData,TestLabel):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(data, label)
    return classifier.score(testData,TestLabel)

def getCont(dataOrg,dataDR,numOfNeighbours):
    return sklearn.manifold.t_sne.trustworthiness(dataDR,dataOrg,numOfNeighbours)
    
def getThrust(dataOrg,dataDR,numOfNeighbours):
    return sklearn.manifold.t_sne.trustworthiness(dataOrg,dataDR,numOfNeighbours)

def evaluateDR(dataOrg,dataDR,label,ContAndThrustNumOfNeigh,trainData,trainLabel):
    return  getSilhouettScore(dataDR,label),getCont(dataOrg,dataDR,ContAndThrustNumOfNeigh),getThrust(dataOrg,dataDR,ContAndThrustNumOfNeigh),getKNN(trainData,trainLabel,5,dataDR,label)
    
'''  
ThrustPSA = sklearn.manifold.t_sne.trustworthiness(X,DRX,100)
ThrustSne = sklearn.manifold.t_sne.trustworthiness(X,DRXTsne,100)
print("Thrust psa:" ,ThrustPSA ,"  tSne:",ThrustSne)
contPSA = sklearn.manifold.t_sne.trustworthiness(DRX,X,100)
conttSne = sklearn.manifold.t_sne.trustworthiness(DRXTsne,X,100)
print("CONT psa:" ,contPSA ,"  tSne:",conttSne)

#%%
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print("done")
#%%
numSamples = 1000
X = X[:5000,:]
print(np.shape(X))
#%%
pca = PCA(n_components=2)
pca.fit(X)
DRX = pca.transform(X)
#%%
DRXTsne = TSNE(n_components = 2).fit_transform(X)


#%%
print(np.shape(DRXTsne))
#%%
print(np.shape(DRX))
plt.subplot(211)
plt.scatter(DRX[:,0],DRX[:,1],c = ((y[:5000].astype(int)+1)/10))
plt.subplot(212)
plt.scatter(DRXTsne[:,0],DRXTsne[:,1],c = ((y[:5000].astype(int)+1)/10))
plt.show


#%%
scorePSA = sklearn.metrics.silhouette_score(DRX, y[:5000].astype(int), metric='euclidean')

scoreTSNE = sklearn.metrics.silhouette_score(DRXTsne, y[:5000].astype(int), metric='euclidean')



#%%
print("psa:" ,scorePSA ,"  tSne:",scoreTSNE)


neighPSA = KNeighborsClassifier(n_neighbors=3)
neighPSA.fit(DRX, y[:5000].astype(int))


neightSNE = KNeighborsClassifier(n_neighbors=3)
neightSNE.fit(DRXTsne, y[:5000].astype(int))

accPSA = neighPSA.score(DRX,y[:5000].astype(int))

acctSne = neightSNE.score(DRXTsne,y[:5000].astype(int))

print("ACC psa:" ,accPSA ,"  tSne:",acctSne)

DistanceORG = ThrustAndCont.dummyL2Calc(X)
DistanceDRX = ThrustAndCont.dummyL2Calc(DRX)
DistancetSNE = ThrustAndCont.dummyL2Calc(DRXTsne)
print("done with distance")


NeighORG = ThrustAndCont.getRankedNeighbours(DistanceORG)
NeighPSA = ThrustAndCont.getRankedNeighbours(DistanceDRX)
NeighTsne = ThrustAndCont.getRankedNeighbours(DistancetSNE)
print("done with Neigh")

print(NeighORG[:2,1:6])
print(NeighPSA[:2,1:6])
print(NeighTsne[:2,1:6])

contPSA =  ThrustAndCont.getContinuity(NeighPSA[:,1:],NeighORG[:,1:6])
conttSne =  ThrustAndCont.getContinuity(NeighTsne[:,1:],NeighORG[:,1:6])
print("CONT psa:" ,contPSA ,"  tSne:",conttSne)

ThrustPSA = ThrustAndCont.getThrustworthiness( NeighORG[:,1:],NeighORG[:,1:5])
ThrustSne = ThrustAndCont.getThrustworthiness( NeighORG[:,1:],NeighORG[:,1:5])
print("CONT psa:" ,ThrustPSA ,"  tSne:",ThrustSne)

#%%
ThrustPSA = sklearn.manifold.t_sne.trustworthiness(X,DRX,100)
ThrustSne = sklearn.manifold.t_sne.trustworthiness(X,DRXTsne,100)
print("Thrust psa:" ,ThrustPSA ,"  tSne:",ThrustSne)
contPSA = sklearn.manifold.t_sne.trustworthiness(DRX,X,100)
conttSne = sklearn.manifold.t_sne.trustworthiness(DRXTsne,X,100)
print("CONT psa:" ,contPSA ,"  tSne:",conttSne)


'''






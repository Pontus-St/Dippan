# -*- coding: utf-8 -*-

import Networks
import smallDataset
import PipeLinePreprocessingAndLoading
import numpy as np
from datetime import datetime
import torch
import matplotlib.pyplot as plt
#%%
def useSiamase(DS_Train,digitData,parameters):
    sia_net = Networks.Network("siamase",digitData)
    print("training sia")
    results= sia_net.train_custom(DS_Train,parameters)
    saveString = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    saveString = "siamase" + saveString
    print("saving model to: " , saveString)
    torch.save(sia_net,saveString)
    
    return results

def useTriplet(DS_Train,digitData,parameters):
    tripnet = Networks.Network("triplet",digitData)
    print("training triplet")
    results = tripnet.train_custom(DS_Train,parameters)
    saveString = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    saveString = "triplet" + saveString
    print("saving model to: " , saveString)
    torch.save(tripnet,saveString)
    return results


def parameterSearch():
    useBrainData = True
    digitData = not useBrainData
    DS = PipeLinePreprocessingAndLoading.loadDataSet(useBrainData)

    X_train, data_train, X_test, data_test = DS.split(int(len(DS)*0.8))


    X_train , X_test = PipeLinePreprocessingAndLoading.normalize(X_train,X_test)
    
    print("datacollected")
    DS_Train = smallDataset.SmallDataset(X_train,data_train)
    DS_Test = smallDataset.SmallDataset(X_test,data_test)
    print("test and train set constructed")

    
    numberOfsamples = 5
    results = []
    parameters = []
    net = None
    
    lrMax= -2.5
    lrMin= -4.5
    
    #only for rough search
    maxIter = 100
    
    epochs = 1 
    
    para = {}
    para.update({"epochs":epochs})
    para.update({"maxIterations":maxIter})

    
    for i in range(10):
        print(i)
        para = {}
        para.update({"epochs":epochs})
        para.update({"maxIterations":maxIter})

        lr = np.power(10,np.random.uniform(lrMin,lrMax))
        para.update({"lr":lr})
        #res = useTriplet(DS_Train,digitData,para)
        res = useSiamase(DS_Train,digitData,para)
        results.append(res)
        parameters.append(para)
    return results ,parameters

results ,parameters = parameterSearch()
#%%
for i in range(len(results)):
    print(results[i], " " , parameters[i].get("lr"))

#%%
lrList = []
lossTrain = []
for i in range(len(results)):
    buffer = parameters[i].get("lr")
    lrList.append(np.log10(buffer))
    lossTrain.append(results[i][1])
plt.scatter(lrList,lossTrain)
plt.title("Parameter search, 10 samples, 10000 iterations ")
plt.xlabel("log(learning rate)")
plt.ylabel("Loss")
plt.show()

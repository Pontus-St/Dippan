# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:35:44 2019

@author: Pontus
"""
import os
from pathlib import Path
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataHolder():
    
    def __init__(self, path):
        print("Reading in data...")
        self.filePath =  Path(path)
        self.subjectsPaths = os.listdir(self.filePath / 'func')
        self.subjects = {}
        infoCounter =1
        for s in self.subjectsPaths:
            print("reading in subject", infoCounter, "/" , len(self.subjectsPaths))
            infoCounter += 1
            self.subjects[s] = Subject(self.filePath/ 'func' /s,self.filePath)
        
    

class Subject():
    def __init__(self,path,originalPath):
        self.FmriPath = path
        self.name = path.name
        self.sessionsPaths = os.listdir(self.FmriPath)
        self.sessions = {}
        self.sessionsLenght = []
        self.sessionCount = 0
        for session in self.sessionsPaths:
            self.sessions[session] = Session(self.FmriPath / session,self.name,session,originalPath,False)
            if self.sessions.get(session).exists:
                self.sessionCount += 1
            else:
                print("subject- session does not exist :",self.name, " - ", session)
        print("created subject with name :", self.name, " sessions " ,self.sessionCount )
        self.getSessionLength()
        
    def getSessionLength(self):
        buffer = self.sessions.keys()
        for i in buffer:
            if self.sessions[i].exists:
                self.sessionsLenght.append(len(self.sessions[i].mappedValidSampleIndex))
            else: 
                self.sessionsLenght.append(0)
    def getDataAndInfoForSubject(self):
        D = None
        info = None
        firstpass = True
        for i in  self.sessions:
            sess = self.sessions.get(i)
            if sess.exists:
                
                if firstpass: 
                    D,info = sess.getDataAndLabel()
                    firstpass =False
                else:
                    buffer_D, buffer_i = sess.getDataAndLabel()
                    D = np.concatenate((D,buffer_D),3)
             
                    info=pd.concat([info,buffer_i])
                
                    
        return D.transpose(),info
class Session():
    def __init__(self,path,name,session,originalPath,getShape):
        self.subjectName = name
        self.sessionName = session
        
        #change to big?
        self.filepath = path / 'effects.nii.gz'
        self.filepath = path / 'effects_small.nii.gz'
        
        self.filepathSmall= path / 'effects_small.nii.gz'
        self.exists = os.path.isfile(self.filepath)
        self.data = None
        infoFilePath = Path(originalPath) /'responses'/name/session /'sequences.csv'
        self.info = pd.read_csv(os.path.abspath(infoFilePath), delim_whitespace=True)
        self.dataShape = None
        self.mappedValidSampleIndex = None
        if self.exists: 
            self.createFilter()

            if (getShape):
                self.loadDropGetDataShape()
            
            
        
        #print(self.mappedValidSampleIndex)
        
            
    def dropData(self):
        self.data = None
    
    def loadData(self):
        self.data = nib.load(os.path.abspath(self.filepath))
        self.data = self.data.get_fdata()

        
    def loadDropGetDataShape(self):
        self.loadData()
        self.dataShape =  self.data.shape
        self.dropData()
        return self.dataShape

    def getShape(self):
        if self.data != None:
             self.dataShape =  self.data.shape

    def createFilter(self):
        #self.mappedValidSampleIndex = self.info['accuracy'] == 1
        self.mappedValidSampleIndex =  self.info.index[self.info['accuracy'] == 1].tolist()


    def cropImage(self):
        nim = nib.load(os.path.abspath(self.filepath))
        image = nim.get_data()
        X, Y, Z,N = image.shape[:4]
        
        x1, x2 = 0,X
        y1, y2 = int(Y*(3/8)), int(Y*(5/8))
        z1, z2 = int(Z*(1/2)), Z
        x1, x2 = max(x1, 0), min(x2, X)
        y1, y2 = max(y1, 0), min(y2, Y)
        z1, z2 = max(z1, 0), min(z2, Z)

        # Crop the image
        image = image[x1:x2, y1:y2, z1:z2,:]
    
        # Update the affine matrix
        affine = nim.affine
        affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
        nim2 = nib.Nifti1Image(image, affine)
        #nim2Fdata = nim2.get_fdata()
        #slice_0 = nim2Fdata[int(X/2), :, :,0]
        #slice_1 = nim2Fdata[:, 0, :,0]
        #slice_2 = nim2Fdata[:, :, 0,0]
        #self.show_slices([slice_0, slice_1, slice_2])
        nib.save(nim2, os.path.abspath(self.filepathSmall)) 
    def show_slices(self,slices):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")

    def getDataAndLabel(self):
        if self.data == None: 
            self.loadData()
       
        toDelete =  self.info.index[self.info['accuracy'] != 1].tolist()
       
        newdata = np.delete(self.data,toDelete,3)
        newinfo = self.info.drop(toDelete)
        self.dropData()
        return newdata, newinfo
 
        
        
if __name__ == "__main__":
    DH = DataHolder("C:/Users/Pontus/Desktop/Dippa/dataset") # delning med / ska användas 
    print("Created DataBase")
    DH.subjects.get
    b = []
    buffer1 =[]
    done = False
    for sub in DH.subjects:
        print(sub)
        b = []
        buffer1 =[]
        for ses in DH.subjects.get(sub).sessions:
            #seq_type
            #true_sequence
            if(DH.subjects.get(sub).sessions.get(ses).exists):
                
                DF = DH.subjects.get(sub).sessions.get(ses).info
                dataCut = DF[(DF['seq_train'] =='trained') ]
                print(sorted(dataCut.true_sequence.unique().tolist()))
                print(sorted(dataCut.seq_type.unique().tolist()))
                buffer1.extend(sorted(dataCut.true_sequence.unique().tolist()))
                #b.extend(sorted(dataCut.true_sequence.unique().tolist()))
                #DH.subjects.get(sub).sessions.get(ses).cropImage()
                D,i = DH.subjects.get(sub).sessions.get(ses).getDataAndLabel()
                print("another")
                break
        break
        b = set(buffer1)
        for i in b:
            print(i)

        print("\n")
#%%
    subs = list(DH.subjects.keys())
    
    D,i = DH.subjects.get(subs[3]).getDataAndInfoForSubject()
    
    
    #%%
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    x,y,z,n=D.shape
    print(x,y,z,n)
    D =D.reshape((x*y*z,n))
    print(D.shape)
    #%%
    
    DRXTsne = TSNE(n_components = 2).fit_transform(D[:,:].transpose())
    #%%
    #col = pd.factorize(i['seq_train'])[0] + 1
    col = pd.factorize(i['seq_train'])[0] 
    print(DRXTsne.shape)
    plt.scatter(DRXTsne[:,0],DRXTsne[:,1],c = col/10)
    plt.colorbar()
    plt.show()
    
    #%%
    for h in i.columns:
        print(h)
    print(col)
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:02:55 2020

@author: Pontus
"""
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn import manifold
import umap

def getPCA():
    pca = make_pipeline(PCA(n_components=2))
    return pca
def getLDA():
    lda = make_pipeline(LinearDiscriminantAnalysis(n_components=2))
    return lda
def getNCA():
    nca= make_pipeline(NeighborhoodComponentsAnalysis(n_components=2))
    return nca

def getTSNE():
    return manifold.TSNE()

def getUmap():
    return umap.UMAP(n_neighbors=5)


import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import acos,degrees
from model import TwoLayerNet

'''This class handles all functionalities required for distinctiveness'''
class Distinctiveness(object):
    model = None
    '''Class is initialized with the old model and the curent number of hidden units. It is then pruned and new model is stored'''
    def __init__(self, model, H):
        removed_index = self.getSimilarUnits(model)
        model.update(H,removed_index)
        self.model = model

    '''Function to identify similar units in a network'''
    def getSimilarUnits(self, model):
        removed_index = []
        A = (model.hrelu).data.numpy()
        A = A / A.max(axis=0)
        A= np.transpose(A)
        A_sparse = sparse.csr_matrix(A)
        similarities_sparse = cosine_similarity(A_sparse,dense_output=True)
        for k in range(0,A.shape[0]):
            for index, values in enumerate(similarities_sparse[k]):
                if index> k and (values>.96 or values<-0.96) and index < A.shape[0]:
                    if index not in removed_index:
                        removed_index.append(index)
                        A = np.delete(A,index,0)
        return list(set(removed_index))
    
    '''Function to return the trimmed model'''
    def updatedModel(self):
        return self.model
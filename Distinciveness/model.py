import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import random
from scipy import sparse


##  Defining model, an additional function called update is added. Update function is responsible for updating the 
##  model parameters hen required. 

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu =  torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)
        self.hrelu = None

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        self.hrelu = h_relu
        return y_pred

    def updateHRelu(self,removed_list):
        A = self.hrelu.data.numpy()
        A = np.transpose(A)
        A = np.delete(A,removed_list,0)
        A = np.transpose(A)
        self.hrelu = (torch.from_numpy(np.array(A)))

    def updateLinear1Bias(self, removed_list):
        f = self.linear1.bias.data.numpy()
        f = np.delete(f,removed_list,0)            
        self.linear1.bias=torch.nn.Parameter(torch.from_numpy(np.array(f)))

    def updateLinear1Weight(self, removed_list):
        A = self.linear1.weight.data.numpy()
        A = np.delete(A,removed_list,0)
        A= (torch.from_numpy(np.array(A)))
        self.linear1.weight = torch.nn.Parameter(A.float())

    def updateLinear2Weight(self, removed_list):
        A = self.linear2.weight.data.numpy()
        A = np.transpose(A)
        A = np.delete(A,removed_list,0)
        A = np.transpose(A)
        A = (torch.from_numpy(np.array(A)))
        self.linear2.weight = torch.nn.Parameter(A.float())

    def update(self, H, removed_list):
        self.updateLinear1Weight(removed_list)
        self.updateLinear2Weight(removed_list)
        self.linear1.out_features = H
        self.linear2.in_features = H

        self.updateHRelu(removed_list)
        self.updateLinear1Bias(removed_list)


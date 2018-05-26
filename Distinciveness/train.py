import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import KFold
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import acos,degrees
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import TwoLayerNet
    
## Initilaizing variables

accuracy_theshold = .10
accuracy_theshold = 1- accuracy_theshold
removed_index = []
test_set = None
frames = []
k = 10
no_of_batch = 5
epoch = 1000
validation_index = 0
distinctiveness = True


print("Run with argument 'normal' for normal run, in all other cases code runs with distinctiveness")
if len(sys.argv)>1 and sys.argv[1]=="normal":
    distinctiveness = False

# Function to split data set into k frames 
def splitDataSet(data_frame):
    index = 0 
    mod_size = (len(data_frame)/k)
    for i in range(0,k):
        i_frame = pd.DataFrame()
        while index < ((i+1)*mod_size):
            i_frame = pd.concat([i_frame,data_frame.iloc[[index]]])
            index+=1
        frames.append(i_frame)

#Function to create train and validation sets depending upon the validation index
def trainAndValidationSets():
    global validation_index
    v_i = validation_index
    validation_set = frames[v_i]
    train_set = pd.DataFrame()
    for x, frame in enumerate(frames): 
        if x!=validation_index:
            train_set = pd.concat([train_set,frame])
    train_target = train_set['F20']
    train_features = train_set.drop('F20',axis=1)
    validation_target = validation_set['F20']
    validation_features = validation_set.drop('F20',axis=1)
    validation_index+=1
    return train_features,train_target,validation_features,validation_target


#Function to identify similar units
def getSimilarUnits(model):
    global removed_index
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
    removed_index = list(set(removed_index))

def getBatches(x,y,no_of_batch):
    bs = y.shape[0]
    bs = int(bs/no_of_batch) 
    batches = []
    for t in range(0,no_of_batch):
        start = t*bs
        end = (t+1)*bs
        if(end>y.shape[0]):
            batches.append([x[start:],y[start:]])
        else:
            batches.append([x[start:end],y[start:end]])
    return batches


#Function to plot the confusion matrix
def plot_confusion(y_pred,y):
    y_pred = y_pred.data
    y=y.data
    y_pred = np.reshape(y_pred,y.shape[0])
    y = np.reshape(y,y.shape[0])
    y = pd.Series(y, name='Actual')
    y_pred = pd.Series(y_pred, name='targ')
    res = pd.crosstab(y,y_pred)
    plt.matshow(res)
    plt.colorbar()
    tick_marks = np.arange(len(res.columns))
    plt.xticks(tick_marks, res.columns, rotation=45)
    plt.yticks(tick_marks, res.index)
    plt.ylabel("Values")
    plt.xlabel("Targets")
    plt.show()

#Function to plot accuracy
def plot_accuracy(accuracies):
    x = [1,2,3,4,5,6,7,8,9,10]
    plt.plot(x,accuracies)
    plt.xlabel("Validation indexes")
    plt.ylabel("Accuracy")
    plt.show()

#Adding an additional row for easier indexing
data_frame = pd.read_csv('diabteic_rheno.csv', 
                  names = ['F1', 'F2', 'F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20'])
data_frame = data_frame[data_frame.F1 != 0]

print("Number of instances: "+str(len(data_frame)))
print("Number of features: 19")
print("Learning rate: 0.001")
print("Initial number of Hidden Neurons: 500")

#Normalizing values
for column in data_frame:
    if column != "F20" and column != "F1":
        data_frame[column] = data_frame.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

data_frame.to_csv('normalized.csv', sep='\t')
#Split dataset into k frames
splitDataSet(data_frame)

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 0.001
D_in, D_out =  19, 1
threshold_met = False
total_loss = 0

accuracy_scores = [] 

while threshold_met != True:
    H=500

    model = TwoLayerNet(D_in, H, D_out)
    train_features,train_target,validation_features,validation_target = trainAndValidationSets()

    x = Variable(torch.from_numpy(np.array(train_features))).float()
    y = Variable(torch.from_numpy(np.array(train_target))).float()

    x_v = Variable(torch.from_numpy(np.array(validation_features))).float()
    y_v = Variable(torch.from_numpy(np.array(validation_target))).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batches = getBatches(x,y,5)
    min_val = len(validation_features)
    tmp_model = model
    min_loss_index = 0
    c= 5
    for t in range(epoch):
        for batch in batches:
            x,y = batch[0],batch[1]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_pred = model(x_v)
        y_pred = (y_pred > 0.5).float()
        loss_fn = torch.nn.L1Loss(size_average=False)
        loss_val = (int(loss_fn(y_pred,y_v)))

        ## If loss is greater than min loss occured so far for 5 consecutive times, we exit. Also only if we train the network to a minimum level (epoch 300)
        if loss_val<min_val:
            min_loss_index = t
            min_val=loss_val
            tmp_model = model
        if t > 200 and loss_val>min_val and min_val< len(validation_features) and t<min_loss_index+5:
            c-=1
            if(c==1):
                break
        else:
            c = 5

   
    if(min_val!=len(validation_features)):
        model = tmp_model

    y_pred = model(x_v)
    y_pred = (y_pred > 0.5).float()
    loss_fn = torch.nn.L1Loss(size_average=False)
    total_loss += (int(loss_fn(y_pred,y_v))/len(validation_target))
    
    accuracy_scores.append(1-(int(loss_fn(y_pred,y_v))/len(validation_target)))

    ## Removing similar weights -- Distinctiveness

    if distinctiveness==True:
        getSimilarUnits(model)
        H = H-len(removed_index)
        model.update(H,removed_index)
        y_pred = model(x_v)
        y_pred = (y_pred > 0.5).float()
        loss_fn = torch.nn.L1Loss(size_average=False)
        n_acc = 1-(int(loss_fn(y_pred,y_v))/len(validation_target))

        print("Accuracy of validation set %d is %.2f. After removing %d neurons, accuracy becomes %.2f" %(validation_index,accuracy_scores[-1],len(removed_index),n_acc))
    else:
        print("Accuracy of validation set %d is %.2f" %(validation_index,accuracy_scores[-1]))


    removed_index = []    

    if validation_index >= k:
        loss_fraction = total_loss/(k)
        if loss_fraction < accuracy_theshold:
            threshold_met=True
            print("Validation accuracy > Threshold -- "+str(1-loss_fraction))
        else:
            print("Validation accuracy < Threshold -- "+str(1-loss_fraction))
            total_loss = 0
            validation_index=0





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
import random
import copy
from distinctiveness import Distinctiveness
    

#Function to return a random value for Hidden Units
def getRandomHidden():
    return random.randint(10,800)

#Function to return a random value for number of epochs
def getRandomEpoch():
    return random.randint(20,1000)

#Function to return a random value for number of batches
def getRandomBatchSize():
    return random.randint(1,14)

#Function to return a random value for Learning rate
def getRandomLearningRate():
    return 1/random.randint(10,1000)

#Function to return a random Optimizer
def getRandomOptimizer():
    optims = [torch.optim.Adagrad,torch.optim.Adam, torch.optim.Adamax, torch.optim.SGD]
    return optims[random.randint(0,3)]

#Function to create a random gene. When required index is passed as paramter.
def getRandom(i):
    function_index = {
        0: getRandomHidden(),
        1: getRandomEpoch(),
        2: getRandomBatchSize(),
        3: getRandomLearningRate(),
        4: getRandomOptimizer()
        }
    return function_index[i]

removed_index = []
test_set = None
frames = []
distinctiveness = True
debug = False

print("Run with argument normal for normal run, in all other cases code runs with distinctiveness. Eg. python train.py normal")
print("To see all intermediate values add argument debug. Eg. python train.py debug/ python train.py normal debug")
if len(sys.argv)>1 and sys.argv[1]=="normal":
    distinctiveness = False

if len(sys.argv)>1 and sys.argv[1]=="debug" or len(sys.argv)>1 and sys.argv[2]=="debug" :
    debug = True

#A print utility functions to control values dispalyed in console 
def printf(message):
    if(debug):
        print(message)

# Function to split data set into k frames 
def splitDataSet(data_frame):
    k = 10
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
    validation_set = pd.DataFrame()
    train_set = pd.DataFrame()
    for x, frame in enumerate(frames): 
        if x!=9 and x!=8 and x!=7:
            train_set = pd.concat([train_set,frame])
        else:
            validation_set = pd.concat([validation_set,frame])
    train_target = train_set['F20']
    train_features = train_set.drop('F20',axis=1)
    validation_target = validation_set['F20']
    validation_features = validation_set.drop('F20',axis=1)
    return train_features,train_target,validation_features,validation_target

#Function to split data into batches
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

#Adding an additional row for easier indexing
data_frame = pd.read_csv('diabteic_rheno.csv', 
                  names = ['F1', 'F2', 'F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20'])
data_frame = data_frame[data_frame.F1 != 0]


#Normalizing values
for column in data_frame:
    if column != "F20" and column != "F1":
        data_frame[column] = data_frame.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

data_frame.to_csv('normalized.csv', sep='\t')
#Split dataset into k frames
splitDataSet(data_frame)

#Function to generates a model and returns the acuuracy and also the model if specified
def execute(param, return_model = False):
    H = param[0]
    epoch = param[1]
    batch_size = param[2]
    learning_rate = param[3]
    optim = param[4]
    printf("================= Hyperparameters used ========================")
    printf("Learning rate:"+str(learning_rate))
    printf("Initial number of Hidden Neurons:"+str(H))
    printf("Epoch:"+str(epoch))
    printf("No of batches to be split:"+ str(batch_size))
    
    
    D_in, D_out =  19, 1
    total_loss = 0

    loss_fn = torch.nn.MSELoss(size_average=False)
    model = TwoLayerNet(D_in, H, D_out)
    train_features,train_target,validation_features,validation_target = trainAndValidationSets()

    x = Variable(torch.from_numpy(np.array(train_features))).float()
    y = Variable(torch.from_numpy(np.array(train_target))).float()

    x_v = Variable(torch.from_numpy(np.array(validation_features))).float()
    y_v = Variable(torch.from_numpy(np.array(validation_target))).float()

    optimizer = optim(model.parameters(), lr=learning_rate)

    batches = getBatches(x,y,batch_size)

    for _ in range(epoch):
        for batch in batches:
            x,y = batch[0],batch[1]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #Testing the model     
    y_pred = model(x_v)
    y_pred = (y_pred > 0.5).float()
    loss_fn = torch.nn.L1Loss(size_average=False)
    total_loss = (int(loss_fn(y_pred,y_v))/len(validation_target))
    loss_fraction = total_loss
    printf("Accuracy "+str(1-loss_fraction))
    if(return_model):
        return (1-loss_fraction),model
    return (1-loss_fraction)

#Function which implements the slection procedure in genetic algorithm. Limit determines the number of values retunred. 
#Values are sorted based on fitness values, accuracy.
def selection(scores , limit = 2):
    top_keys = sorted(scores,reverse=True)
    top_keys = top_keys[0:limit]
    selected_params = []
    for key in top_keys:
        selected_params.append(scores[key][0])
    return selected_params

#Function which implements the mutation procedure in genetic algorithm. 
def mutate(params):
    _params= copy.deepcopy(params)
    for param in _params:
        if random.randint(0,4) > 2:
            mutated = copy.deepcopy(param)
            index  = random.randint(0,4)
            mutated[index] = getRandom(index)
            params.append(mutated)
    return params

#Function which implements the creation of population.
def initilaize():
    params = []
    for _ in range (10):
        params.append([getRandom(0), getRandom(1),getRandom(2), getRandom(3), getRandom(4)])
    return params

#Function which implements the breeding procedure. Limit determines the number of children to be produced
def createAndIncludeChildren(params, limit):
    _parms = copy.deepcopy(params)
    c_params = []
    end = len(_parms) -1
    while(limit>0):
        father = _parms[random.randint(0,end)]
        mother = _parms[random.randint(0,end)]
        if father!=mother :
            c_params.append([
            random.choice([father[0],mother[0]]),
            random.choice([father[1],mother[1]]),
            random.choice([father[2],mother[2]]),
            random.choice([father[3],mother[3]]),
            random.choice([father[4],mother[4]])
            ])
            limit-=1
    return c_params

scores = {}
params = initilaize()
for param in params:
    scores.setdefault(execute(param),[]).append(param)

x = 2
if debug:
    x = 40

##Repeating the population generation process
while(x>1):
    params = selection(scores,3)
    params = mutate(params)
    c_params = createAndIncludeChildren(params,10-len(params))
    params = params + c_params
    if(len(params)>1):
        scores = {}
    for param in params:
        scores.setdefault(execute(param),[]).append(param)
    x-=1

best_params = selection(scores,1)[0]
accuracy, model = execute(best_params,True)
print("Accuracy: ",accuracy)
print("====Params=====")
print("No of Hidden Units,  ",best_params[0])
print("No of Epochs,        ",best_params[1])
print("No of Batches used,  ",best_params[2])
print("Learning Rate,       ",best_params[3])
print("Optimizer,           ",best_params[4])

if(distinctiveness and len(best_params)==5):
    _distinctiveness = Distinctiveness(model,best_params[0])
    updated_model =  _distinctiveness.updatedModel()
    _,_,validation_features,validation_target = trainAndValidationSets()
    x_v = Variable(torch.from_numpy(np.array(validation_features))).float()
    y_v = Variable(torch.from_numpy(np.array(validation_target))).float()
    y_pred = updated_model(x_v)
    y_pred = (y_pred > 0.5).float()
    loss_fn = torch.nn.L1Loss(size_average=False)
    total_loss = (int(loss_fn(y_pred,y_v))/len(validation_target))
    loss_fraction = total_loss
    print("Accuracy after distincitveness "+str(1-loss_fraction))

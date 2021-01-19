import pandas as pd
import numpy as np
from extrafunctions import *
import math

df = pd.read_csv('dataset.csv')
x_train = df.iloc[:, :-1].to_numpy()
y_train = df.iloc[:, -1].to_numpy()

data_len = len(y_train)
input_dim = x_train.shape[-1]
nb_epochs = 20
lr = 0.01

np.random.seed(7)


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

def runClassicalNeuronReturnProbability (inputVector, weightVector):
    inputVector = normalize(inputVector)
    weightVector = normalize(weightVector)
    
    innerProduct = np.inner(inputVector, weightVector)
    prob = innerProduct**2
    return prob

def returnProbability2 (inputVector, weightVector):
    inputVector = normalize(inputVector)
    weightVector = normalize(weightVector)
    
    innerProduct = np.inner(inputVector, weightVector)
    ampl1 = innerProduct#**2
    ampl0 = math.sqrt(1-innerProduct**2)
    
    return ampl1, ampl1


def runNeuron (nb_epochs, binaryWeights=False, stochastic=True, threshold=0.09, lr=0.01):
    w = np.random.uniform(-1,1,input_dim)# #np.random.rand(input_dim) # Real weights 
    w = normalize(w)
    
    wB = makeBinarization(w, stochastic) # Binarization of Real weights
    wB = normalize(wB)   
    maxHit = 0
    for epoch in range(nb_epochs):
        y_pred = np.zeros(data_len)
        for i, x in enumerate(x_train):
            x = normalize(x)
            if binaryWeights:
                out = np.sum(np.multiply(x, wB))
            else:
                out = np.sum(np.multiply(x, w))

            if abs(out)**2 > threshold:
                y_pred[i] = 1
                
            if y_pred[i]!= y_train[i]:
                delta = y_train[i] - y_pred[i]

                for j in range(input_dim):
                    w[j] = w[j] + (lr * delta * x_train[i][j])

                wB = makeBinarization(w, stochastic)
                wB = normalize(wB)

        hits = (y_train == y_pred).sum()
        maxHit = max(maxHit, (hits / data_len) * 100)
        print('Epoch {:d} accuracy: {:.2f} max acc {:.2f}'.format(epoch + 1, (hits / data_len) * 100, maxHit))

# print("100 epocas, pesos reais")
# runNeuron (nb_epochs=100, binaryWeights=False)
"""
for threshold in [0.07, 0.08, 0.09, 0.01, 0.001, 0.01, 0.02]:
    for lr in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]:
        print("thr", threshold)
        print("lr", lr)
        
        print("100 epocas, pesos binarizados deterministicamente")
        runNeuron (nb_epochs=1000, binaryWeights=True, stochastic=False, threshold=threshold, lr=lr)
"""
# print("100 epocas, pesos binarizados estocasticamente")
# runNeuron (nb_epochs=100, binaryWeights=True, stochastic=True)



def deltaRule (inputVector, weightVector, threshold=0.09, lr=0.01, y_train=0, out=0):
    y_pred = 0
    if abs(out) > threshold:
        y_pred = 1
    #print("atualizando pesos")
    #delta = y_train - y_pred
    delta = y_train-out
    input_dim = len(weightVector)
    for j in range(input_dim):
        weightVector[j] =  weightVector[j] - (lr * delta * inputVector[j])     
    """
    old
    """
    """
    y_pred = 0
    if abs(out) > threshold:
        y_pred = 1
    if y_pred != y_train:
        #print("atualizando pesos")
        #delta = y_train - y_pred
        delta = y_train-out
        input_dim = len(weightVector)

        for j in range(input_dim):
            weightVector[j] =  weightVector[j] + (lr * delta * inputVector[j])
    """
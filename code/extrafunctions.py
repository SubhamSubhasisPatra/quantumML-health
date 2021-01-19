import pandas as pd
import numpy as np
import math

def deterministicBinarization (w):
    # This function makes a Deterministic Binarization of a given weights list
    w2 = [0] * len(w)

    for i in range(len(w)):
        if w[i] >= 0:
            w2[i] = 1
        else:
            w2[i] = -1
    
    return w2

def hardSigmoid(x):
    result = max(0,min(1,(x+1)/2))
    return result 

def stochasticBinarization(w):
    # This function makes a Stochastic Binarization of a given weights list based on the hard sigmoid function
    w2 = [0] * len(w)

    for i in range(len (w)):
        theWeight = int(hardSigmoid(w[i]))
        if theWeight == 0:
            w2[i] = 1
        else:
            w2[i] = -1

    return w2

def makeBinarization(w, stochastic=True):
    if stochastic:
        return stochasticBinarization(w)
    else:
        return deterministicBinarization(w)
    
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

def generateRandomState(n):
    """
    gera um estado quantico aleatório com valores -1 e 1 de amplitude de tamanho n
    """
    state = np.random.randint(2, size=n)
    stateREW = []
    for amplitude in state:
        if (amplitude == 0):
            stateREW.append(-1)
        else:
            stateREW.append(1)
    return stateREW

def generateWeightWithSomeDifference(inputVector, difference):
    """
    retorna o estado inputvector com #'difference' posicoes diferentes
    """
    weightVector = inputVector[:]
    posicoes = []
    size = len(inputVector)
    finalizou = False
    while len(posicoes) != difference:
        posicao = np.random.randint(size)
        if not posicao in posicoes:
            weightVector[posicao] *= -1
            posicoes.append(posicao)
    return weightVector

def costFuncPSO(w):
    new_w = []
    for i in range(0, len(w)):
        new_w += (w[i]/2)

    return new_w
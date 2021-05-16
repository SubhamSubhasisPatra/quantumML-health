from code.neuron import *
from code.encodingsource import *
from code.hsgs import *
from code.classical_pso import *
from code.sf import *
simulator = Aer.get_backend('qasm_simulator')
import pandas as pd
import numpy as np


'''
AUXILIARY FUNCTIONS
'''

def inverterSinalPeso(w):
    for i in range(len(w)):
        w[i] *= -1
        
def deltaRule(inputVector, weightVector, threshold=0.09, lr=0.01, y_train=0, out=0):
    y_pred = 0
    if abs(out) > threshold:
        y_pred = 1
    #print("atualizando pesos")
    delta = y_train - y_pred
    #delta = y_train-out
    input_dim = len(weightVector)
    for j in range(input_dim):
        weightVector[j] =  weightVector[j] - (lr * delta * inputVector[j])     
        
   

def quantumNeuronFIT(Xs_train, ys_train, w0, lrParameter=0.09, threshold=0.5, n_epochs=400, seed=1, trainingBias=True, trainingApproaches={}, epoch_results = False, phaseEstrategyOperator = 'phase-encoding-phase'):
    
    print('lrParameter: ', lrParameter)
    print('threshold: ', threshold)
    print('trainingBias: ', trainingBias)
    print('phaseEstrategyOperator: ', phaseEstrategyOperator)

    input_dim = len(Xs_train[0])

    if (trainingBias):
        #for i in range(len(list(set(ys_train)))):
        #vRandom = deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim))
        weightVectorHSGS = w0 + [1]*len(w0)
        weightVectorPhaseEncoding = w0 + [1]*len(w0)
    else:
        #for i in range(len(list(set(ys_train)))):
        #vRandom = deterministicBinarization(np.random.uniform(-1, 1, input_dim))
        weightVectorHSGS = w0
        weightVectorPhaseEncoding = w0

    
    bestWeightHSGS = []
    bestWeightPhaseEncoding = []

    bestErrorHSGS = 999999
    bestErrorPhaseEncoding = 999999
    best_epoch_errosHSGS=[]
    best_epoch_errosPhaseEncoding = []

    for iteration in range(n_epochs):

        epoch_erroHSGS = 0
        epoch_erroPhaseEncoding = 0

        errosHSGS=[]
        errosPhaseEncoding = []

        for posicaoTreinamento in range(len(Xs_train)):
            
            inputVector = Xs_train[posicaoTreinamento]
            y_train = ys_train[posicaoTreinamento]
            
            if (trainingBias):
                inputVector = inputVector + len(inputVector)*[1]


            """
            executando o HSGS
            """
            if ("hsgs" in trainingApproaches):
                operator = "hsgs"
                
                weightVectorHSGS_bin = deterministicBinarization(weightVectorHSGS) # Binarization of Real weights
                neuronHSGS = createNeuron(inputVector, weightVectorHSGS_bin, operator)
                resultadoHSGS = executeNeuron(neuronHSGS, simulator, threshold=None)
                weightVectorHSGS = deltaRule(inputVector, weightVectorHSGS, lr=lrParameter, threshold=threshold, y_train=y_train, out=resultadoHSGS)

            
            """
            executando o phase encoding 
            """
            if ("phase-encoding" in trainingApproaches):
                operator = phaseEstrategyOperator

                neuronPhase = createNeuron(inputVector, weightVectorPhaseEncoding, operator)
                resultadoPhaseEncoding = executeNeuron(neuronPhase, simulator, threshold=None)
                print(weightVectorPhaseEncoding)
                weightVectorPhaseEncoding = deltaRule(inputVector, weightVectorPhaseEncoding, lr=lrParameter, threshold=threshold, y_train=y_train, out=resultadoPhaseEncoding)

            """
            computar erros
            """
            
            if (resultadoHSGS > threshold):
                epoch_erroHSGS += abs(y_train - 1)
                errosHSGS.append((1, abs(y_train)))
            else:
                epoch_erroHSGS += abs(y_train - 0)
                errosHSGS.append((0, y_train))
            
            
            if (resultadoPhaseEncoding > threshold):
                epoch_erroPhaseEncoding += abs(y_train - 1)
                errosPhaseEncoding.append((1, abs(y_train)))
            else:
                epoch_erroPhaseEncoding += abs(y_train - 0)
                errosPhaseEncoding.append((0, y_train))
            
            
        if (epoch_erroHSGS < bestErrorHSGS):
            bestWeightHSGS = weightVectorHSGS
            bestErrorHSGS = epoch_erroHSGS
            best_epoch_errosHSGS = errosHSGS
        
        if (epoch_erroPhaseEncoding < bestErrorPhaseEncoding):
            bestWeightPhaseEncoding = weightVectorPhaseEncoding
            bestErrorPhaseEncoding = epoch_erroPhaseEncoding
            best_epoch_errosPhaseEncoding = errosPhaseEncoding

        if epoch_results == True:    
            print("\nerro HSGS", epoch_erroHSGS)
            print("erro phase encoding", epoch_erroPhaseEncoding)
            
        if epoch_erroPhaseEncoding == 0:
            return [bestWeightPhaseEncoding, bestWeightHSGS, best_epoch_errosHSGS, best_epoch_errosPhaseEncoding]
    
    print("best error phase-encoding training: ", bestErrorPhaseEncoding)
    print("best error HSGS training: ", bestErrorHSGS)

    return [bestWeightPhaseEncoding, bestWeightHSGS, best_epoch_errosHSGS, best_epoch_errosPhaseEncoding]


def quantumNeuronPREDICT(Xs_test, ys_test, weightVectorsPhaseEncoding, weightVectorsHSGS,  threshold=0.5, repeat=30, bias=True, testingApproaches={}, phaseEstrategyOperator = 'phase-encoding-phase'):
    
    
    errosHSGS = []
    errosEncodingWeight = []
    errosEncodingInput = []
    errosPhaseEncoding = []
    errosClassico = []
    errosClassicoBin = []

    outputsHSGS = []
    outputsEncodingWeight = []
    outputsEncodingInput = []
    outputsPhaseEncoding = []



    for i in range(repeat):
        erroHSGS = 0
        erroEncodingWeight = 0
        erroEncodingInput = 0
        erroPhaseEncoding = 0
        erroClassico =0
        erroClassicoBin=0

        for pos in range(len(Xs_test)):
            inputVector = Xs_test[pos] # inputVectors[pos]

            if bias == True:
                inputVector = inputVector + len(inputVector)*[1]

            target = ys_test[pos]

            valorMaiorHSGS=0
            neuronMaiorHSGS=0

            valorMaiorEncodingWeight=0
            neuronMaiorEncodingWeight=0

            valorMaiorEncodingInput=0
            neuronMaiorEncodingInput=0

            valorMaiorPhaseEncoding=0
            neuronMaiorPhaseEncoding=0


            #for neuronClass in range(len(list(set(ys_test)))):
            neuronClass=0
            if neuronClass == 0:
                if ("hsgs" in testingApproaches):
                    operator = "hsgs"
                    wBinaryBinary = deterministicBinarization(weightVectorsHSGS[neuronClass]) # Binarization of Real weights
                    neuron = createNeuron(inputVector, wBinaryBinary, operator)
                    resultadoHSGS1 = executeNeuron(neuron, simulator, threshold=threshold)
                    #if(resultadoHSGS1 > valorMaiorHSGS):
                    #    neuronMaiorHSGS = neuronClass
                    #    valorMaiorHSGS = resultadoHSGS1

                if ("phase-encoding" in testingApproaches):
                    operator = phaseEstrategyOperator
                    neuron = createNeuron(inputVector, weightVectorsPhaseEncoding[neuronClass], operator)
                    resultadoPhaseEncoding1 = executeNeuron(neuron, simulator, threshold=threshold)
                    #if(resultadoPhaseEncoding1 > valorMaiorPhaseEncoding):
                    #    neuronMaiorPhaseEncoding = neuronClass
                    #    valorMaiorPhaseEncoding = resultadoPhaseEncoding1
                        
                if ("encoding-weight" in testingApproaches):
                    operator = "encoding-weight"
                    inputBinary = thresholdBinarization(inputVector) # Binarization of Real Inputs
                    neuron = createNeuron(inputBinary,  weightVectorsEncodingWeight[neuronClass], operator)
                    resultadoEncodingWeight1 = executeNeuron(neuron, simulator, threshold=None)
                    if(resultadoEncodingWeight1 > valorMaiorEncodingWeight):
                        neuronMaiorEncodingWeight = neuronClass
                        valorMaiorEncodingWeight = resultadoEncodingWeight1
                        
                if ("encoding-input" in testingApproaches):
                    operator = "encoding-input"
                    neuron = createNeuron(inputVector, weightVectorsEncodingInput[neuronClass], operator)
                    resultadoEncodingInput1 = executeNeuron(neuron, simulator, threshold=None)
                    if(resultadoEncodingInput1 > valorMaiorEncodingInput):
                        neuronMaiorEncodingInput = neuronClass
                        valorMaiorEncodingInput = resultadoEncodingInput1

                # get predicted probability results to class 1
                #if neuronClass == 1:
                #    outputsHSGS.append(resultadoHSGS1)
                #    outputsEncodingWeight.append(resultadoEncodingWeight1)
                #    outputsEncodingInput.append(resultadoEncodingInput1)
                #    outputsPhaseEncoding.append(resultadoPhaseEncoding1)

            ##################################################
            """
            erros
            """

            # get predicted probability results to class 1
            outputsHSGS.append(resultadoHSGS1)
            outputsPhaseEncoding.append(resultadoPhaseEncoding1)
            
            #outputsHSGS.append(neuronMaiorHSGS)
            #outputsPhaseEncoding.append(neuronMaiorPhaseEncoding)
            #erroHSGS_bin = 0
            #if (neuronMaiorHSGS != target):   
            #    erroHSGS_bin = 1
            #erroPhaseEncoding_bin = 0
            #if (neuronMaiorPhaseEncoding != target):   
            #    erroPhaseEncoding_bin = 1
        
            erroHSGS_bin = 0
            if (resultadoHSGS1 != target):   
                erroHSGS_bin = 1

            erroPhaseEncoding_bin = 0
            if (resultadoPhaseEncoding1 != target):   
                erroPhaseEncoding_bin = 1

            erroHSGS += erroHSGS_bin####abs(resultadoHSGS_bin-y_train)
            erroPhaseEncoding += erroPhaseEncoding_bin####abs(resultadoEncoding_bin-y_train)
 

        #print("erro HSGS", erroHSGS/len(Xs_test))
        #print("erro phase encoding", erroPhaseEncoding/len(Xs_test))


        errosHSGS.append(round(erroHSGS/len(Xs_test), 4))
        #errosEncodingWeight.append(round(erroEncodingWeight/len(Xs_test), 4))
        #errosEncodingInput.append(round(erroEncodingInput/len(Xs_test), 4))
        errosPhaseEncoding.append(round(erroPhaseEncoding/len(Xs_test), 4))

    print("ERROS HSGS            ",  np.average(errosHSGS))
    #print("ERROS ENCODING WEIGHT ",  np.average(errosEncodingWeight))
    #print("ERROS ENCODING INPUT  ",  np.average(errosEncodingInput))
    print("ERROS PHASE ENCODING  ",  np.average(errosPhaseEncoding))

    """
    #results and metrics
    
    results = { 'error_HSGS': errosHSGS,
                'error_phase_encoding':errosPhaseEncoding,
               
                'output_HSGS': outputsHSGS,
                'output_phase_encoding':outputsPhaseEncoding,

                'weights_learned_HSGS':weightVectorsHSGS,
                'weights_learned_phase_encoding':weightVectorsPhaseEncoding
        }
    
    """
    return  np.average(errosPhaseEncoding), outputsPhaseEncoding, weightVectorsPhaseEncoding, np.average(errosHSGS), outputsHSGS,  weightVectorsHSGS
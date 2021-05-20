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
        
        
def inverterSinalPeso(w):
    for i in range(len(w)):
        w[i] *= -1
        
            
'''
EXPERIMENT FUNCTIONS
'''

def quantumNeuronFIT(Xs_train, ys_train, init_weight, lrParameter=0.09, threshold=0.5, n_epochs=400, trainingBias=True, trainingApproaches={}, epoch_results = False, phaseEstrategyOperator = 'phase-encoding-phase'):
    
    print('lrParameter: ', lrParameter)
    print('threshold: ', threshold)
    print('trainingBias: ', trainingBias)
    print('phaseEstrategyOperator: ', phaseEstrategyOperator)

    input_dim = len(Xs_train[0])

    if (trainingBias):
        weightVectorHSGS = init_weight.copy() + [1]*len(init_weight)
        weightVectorPhaseEncoding = init_weight.copy() + [1]*len(init_weight)
    else:
        weightVectorHSGS = init_weight.copy()
        weightVectorPhaseEncoding = init_weight.copy()

    #bestWeightHSGS = []
    #bestWeightPhaseEncoding = []

    bestErrorHSGS = 999999
    bestErrorPhaseEncoding = 999999
    best_epoch_errosHSGS=[]
    best_epoch_errosPhaseEncoding = []
    
    epoch_evolutionHSGS = []
    epoch_evolutionPhaseEncoding = []

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
                deltaRule(inputVector, weightVectorHSGS, lr=lrParameter, threshold=threshold, y_train=y_train, out=resultadoHSGS)

            
            """
            executando o phase encoding 
            """
            if ("phase-encoding" in trainingApproaches):
                operator = "phase-encoding"
                

                if phaseEstrategyOperator == 'angle':
                    inputVector = [math.atan(inputVector[i]/inputVector[i+1]) for i in range(0, len(inputVector), 2)] + [0]*int(len(inputVector)/2)
                elif phaseEstrategyOperator == 'radius':
                    inputVector = [math.sqrt(inputVector[i]**2 + inputVector[i+1]**2) for i in range(0, len(inputVector), 2)] + [0]*int(len(inputVector)/2)
                elif phaseEstrategyOperator == 'angleradius':
                    inputVector = [math.sqrt(inputVector[i]**2 + inputVector[i+1]**2) for i in range(0, len(inputVector), 2)] + [math.atan(inputVector[i]/inputVector[i+1]) for i in range(0, len(inputVector), 2)] 
    
                neuronPhase = createNeuron(inputVector, weightVectorPhaseEncoding, operator)
                resultadoPhaseEncoding = executeNeuron(neuronPhase, simulator, threshold=None)
                deltaRule(inputVector, weightVectorPhaseEncoding, lr=lrParameter, threshold=threshold, y_train=y_train, out=resultadoPhaseEncoding)

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
            
        epoch_evolutionHSGS.append(epoch_erroHSGS)
        epoch_evolutionPhaseEncoding.append(epoch_erroPhaseEncoding)
        
        if (epoch_erroHSGS < bestErrorHSGS):
            bestErrorHSGS = epoch_erroHSGS
            best_epoch_errosHSGS = errosHSGS
        
        if (epoch_erroPhaseEncoding < bestErrorPhaseEncoding):
            bestErrorPhaseEncoding = epoch_erroPhaseEncoding
            best_epoch_errosPhaseEncoding = errosPhaseEncoding

        if epoch_results == True:    
            print("\nerro HSGS", epoch_erroHSGS)
            print("weights HSGS", weightVectorHSGS)
            print("erro phase encoding", epoch_erroPhaseEncoding)
            print("weights phase encoding", weightVectorPhaseEncoding)
            
            
        if epoch_erroPhaseEncoding == 0 and epoch_erroHSGS == 0:
            print("\nbest error phase-encoding training: ", bestErrorPhaseEncoding)
            print("best error HSGS training: ", bestErrorHSGS)
            return [weightVectorPhaseEncoding, weightVectorHSGS, best_epoch_errosHSGS, best_epoch_errosPhaseEncoding, epoch_evolutionHSGS, epoch_evolutionPhaseEncoding]
    
    print("\nbest error phase-encoding training: ", bestErrorPhaseEncoding)
    print("best error HSGS training: ", bestErrorHSGS)
    return [weightVectorPhaseEncoding, weightVectorHSGS, best_epoch_errosHSGS, best_epoch_errosPhaseEncoding, epoch_evolutionHSGS, epoch_evolutionPhaseEncoding]


def quantumNeuronPREDICT(Xs_test, ys_test, weightVectorsPhaseEncoding, weightVectorsHSGS,  threshold=0.5, repeat=30, bias=True, testingApproaches={}, phaseEstrategyOperator = 'phase-encoding-phase'):
    
    
    errosHSGS = []
    errosPhaseEncoding = []

    outputsHSGS = []
    outputsPhaseEncoding = []
    
    y_targets =[]
    y_examples =[]


    for i in range(repeat):
        erroHSGS = 0
        erroPhaseEncoding = 0

        for pos in range(len(Xs_test)):
            inputVector = Xs_test[pos] 

            if bias == True:
                inputVector = inputVector + len(inputVector)*[1]

            target = ys_test[pos]

            valorMaiorHSGS=0
            neuronMaiorHSGS=0

            valorMaiorPhaseEncoding=0
            neuronMaiorPhaseEncoding=0


            #for neuronClass in range(len(list(set(ys_test)))):
            if ("hsgs" in testingApproaches):
                operator = "hsgs"
                wBinaryBinary = deterministicBinarization(weightVectorsHSGS) # Binarization of Real weights
                neuron = createNeuron(inputVector, wBinaryBinary, operator)
                resultadoHSGS1 = executeNeuron(neuron, simulator, threshold=threshold)


            if ("phase-encoding" in testingApproaches):
                operator = 'phase-encoding'
                
                if phaseEstrategyOperator == 'angle':
                    inputVector = [math.atan(inputVector[i]/inputVector[i+1]) for i in range(0, len(inputVector), 2)] + [0]*int(len(inputVector)/2)
                elif phaseEstrategyOperator == 'radius':
                    inputVector = [math.sqrt(inputVector[i]**2 + inputVector[i+1]**2) for i in range(0, len(inputVector), 2)] + [0]*int(len(inputVector)/2)
                elif phaseEstrategyOperator == 'angleradius':
                    inputVector = [math.sqrt(inputVector[i]**2 + inputVector[i+1]**2) for i in range(0, len(inputVector), 2)] + [math.atan(inputVector[i]/inputVector[i+1]) for i in range(0, len(inputVector), 2)]                 

                neuron = createNeuron(inputVector, weightVectorsPhaseEncoding, operator)
                resultadoPhaseEncoding1 = executeNeuron(neuron, simulator, threshold=threshold)



            """
            erros
            """

            # get predicted probability results to class 1
            outputsHSGS.append(resultadoHSGS1)
            outputsPhaseEncoding.append(resultadoPhaseEncoding1)
            y_targets.append(target)
            y_examples.append(Xs_test[pos])

            
            erroHSGS_bin = 0
            if (resultadoHSGS1 != target):   
                erroHSGS_bin = 1

            erroPhaseEncoding_bin = 0
            if (resultadoPhaseEncoding1 != target):   
                erroPhaseEncoding_bin = 1

            erroHSGS += erroHSGS_bin####abs(resultadoHSGS_bin-y_train)
            erroPhaseEncoding += erroPhaseEncoding_bin####abs(resultadoEncoding_bin-y_train)
            
            
        #acerto_HSGS_ = [1 if outputsHSGS[i] == y_targets[i] else 0 for i in range(len(y_targets))]
        #acerto_phase_ = [1 if outputsPhaseEncoding[i] == y_targets[i] else 0 for i in range(len(y_targets))]
        
        #print("erro HSGS", erroHSGS/len(Xs_test))
        #print("erro phase encoding", erroPhaseEncoding/len(Xs_test))


        errosHSGS.append(round(erroHSGS/len(Xs_test), 4))
        errosPhaseEncoding.append(round(erroPhaseEncoding/len(Xs_test), 4))

    print("AVG TEST ERROR HSGS   ",  np.average(errosHSGS))
    print("AVG TEST ERROR PHASE  ",  np.average(errosPhaseEncoding))

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
    return  np.average(errosPhaseEncoding), outputsPhaseEncoding, weightVectorsPhaseEncoding, np.average(errosHSGS), outputsHSGS,  weightVectorsHSGS, y_targets, y_examples
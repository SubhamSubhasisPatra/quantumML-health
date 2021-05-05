from code.neuron import *
from code.encodingsource import *
from code.hsgs import *
from code.classical_neuron import *
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
        
def treinamentoNeuronio(operator, inputVector, weightVector, y_train, lrParameter = 0.05):


    n = int(math.log(len(inputVector), 2))

    if (operator == "hsgs"):
        wBinaryBinary = deterministicBinarization(weightVector) # Binarization of Real weights
        neuron = createNeuron(inputVector, wBinaryBinary, operator)
        resultado = executeNeuron(neuron, simulator, threshold=None)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado
    elif (operator == "encoding-weight"):        
        inputBinary = thresholdBinarization(inputVector) # Binarization of Real inputs
        neuron = createNeuron(inputBinary, weightVector, operator)
        resultado = executeNeuron(neuron, simulator, threshold=None)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado
    elif (operator == "encoding-input"):        
        neuron = createNeuron(inputVector, weightVector, operator)
        resultado = executeNeuron(neuron, simulator, threshold=None)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado
    elif (operator == "phase-encoding"): 
        neuron = createNeuron(inputVector, weightVector, operator, ancilla=True)
        resultado = executeNeuron(neuron, simulator, threshold=None)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado
    elif (operator == "neuronio-classico"):
        resultado = runClassicalNeuronReturnProbability(inputVector, weightVector)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado
    elif (operator == "neuronio-classico-bin"):
        wBinaryBinary = deterministicBinarization(weightVector)
        resultado = runClassicalNeuronReturnProbability(inputVector, wBinaryBinary)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado

'''
EXPERIMENT FUNCTIONS
'''



def experiment_TRAIN(Xs_train, ys_train, lrParameter=0.09, thresholdTrain=None, n_epochs=400, seed=1, trainingBias=True, trainingApproaches={}, error_by_epoch =False, epoch_results = True):

    np.random.seed(seed)
    weightVectorsHSGS = []
    weightVectorsEncodingWeight = []
    weightVectorsEncodingInput = []
    weightVectorsPhaseEncoding = []

    input_dim = len(Xs_train[0])

    if (trainingBias):
        for i in range(len(list(set(ys_train)))):
            vRandom = deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim))
            weightVectorsHSGS.append(vRandom.copy())
            weightVectorsEncodingWeight.append(vRandom.copy()) #.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
            weightVectorsEncodingInput.append(vRandom.copy()) #.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
            weightVectorsPhaseEncoding.append(vRandom.copy()) #.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
    else:
        for i in range(len(list(set(ys_train)))):
            vRandom = deterministicBinarization(np.random.uniform(-1, 1, input_dim))
            weightVectorsHSGS.append(vRandom.copy())
            weightVectorsEncodingWeight.append(vRandom.copy()) #= weightVectorsHSGS.copy() #.append(deterministicBinarization(np.random.uniform(-1, 1, input_dim)))
            weightVectorsEncodingInput.append(vRandom.copy()) #= weightVectorsHSGS.copy() #.append(deterministicBinarization(np.random.uniform(-1, 1, input_dim)))
            weightVectorsPhaseEncoding.append(vRandom.copy()) #= weightVectorsHSGS.copy() #.append(deterministicBinarization(np.random.uniform(-1, 1, input_dim)))

   

    bestWeightsHSGS = []
    bestWeightsEncodingWeight = []
    bestWeightsEncodingInput = []
    bestWeightsPhaseEncoding = []

    bestErrorHSGS=999999
    bestErrorEncodingWeight = 999999
    bestErrorEncodingInput = 999999
    bestErrorPhaseEncoding = 999999

    #bestWeightsHSGSInTime = []
    #bestWeightsEncodingWeightInTime = []
    #bestWeightsEncodingInputInTime = []
    #bestWeightsPhaseEncodingInTime = []

    limiarErroToleravel = 0.03
    tamTreinamento = len(Xs_train)

    resultadoHSGS=0
    resultadoEncodingWeight=0
    resultadoEncodingInput=0
    resultadoPhaseEncoding=0
    resultadoClassico=0
    resultadoClassicoBin=0
    
    #epoch_errosHSGS=[]
    #epoch_errosEncodingWeight = []
    #epoch_errosEncodingInput = []
    #epoch_errosPhaseEncoding = []
  

    for iteration in range(n_epochs):

        erroHSGS = 0
        erroEncodingWeight = 0
        erroEncodingInput = 0
        erroPhaseEncoding = 0

        errosHSGS=[]
        errosEncodingWeight = []
        errosEncodingInput = []
        errosPhaseEncoding = []

        for posicaoTreinamento in range(tamTreinamento):
            
            inputVector = Xs_train[posicaoTreinamento] # inputVectors[posicaoTreinamento]
            y_train = ys_train[posicaoTreinamento]
            
            if (trainingBias):
                inputVector = inputVector + len(inputVector)*[1]


            """
            executando o HSGS
            """
            if ("hsgs" in trainingApproaches):
                operator = "hsgs"
                resultadoHSGS = treinamentoNeuronio(operator = operator, 
                                                    inputVector= inputVector, 
                                                    weightVector = weightVectorsHSGS[y_train], 
                                                    y_train=1, 
                                                    lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsHSGS[y_train])
                for i in range(len(weightVectorsHSGS[y_train])):
                    weightVectorsHSGS[y_train][i] = round(weightVectorsHSGS[y_train][i]/norm,12)

            
            """
            executando o phase encoding 
            """
            if ("phase-encoding" in trainingApproaches):
                operator = "phase-encoding"
                resultadoPhaseEncoding = treinamentoNeuronio(operator = operator, 
                                                    inputVector= inputVector, 
                                                    weightVector = weightVectorsPhaseEncoding[y_train], 
                                                    y_train=1, 
                                                    lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsPhaseEncoding[y_train])
                for i in range(len(weightVectorsPhaseEncoding[y_train])):
                    weightVectorsPhaseEncoding[y_train][i] = round(weightVectorsPhaseEncoding[y_train][i]/norm,12)

            
            """
            executando o encoding weight
            """
            if ("encoding-weight" in trainingApproaches):
                operator = "encoding-weight"
                resultadoEncodingWeight = treinamentoNeuronio(operator = operator, 
                                                        inputVector= inputVector, 
                                                        weightVector = weightVectorsEncodingWeight[y_train], 
                                                        y_train=1, 
                                                        lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsEncodingWeight[y_train])
                for i in range(len(weightVectorsEncodingWeight[y_train])):
                    weightVectorsEncodingWeight[y_train][i] = round(weightVectorsEncodingWeight[y_train][i]/norm,12)
                
            """
            executando o encoding input
            """
            if ("encoding-input" in trainingApproaches):
                operator = "encoding-input"
                resultadoEncodingInput = treinamentoNeuronio(operator = operator, 
                                                        inputVector= inputVector, 
                                                        weightVector = weightVectorsEncodingInput[y_train], 
                                                        y_train=1, 
                                                        lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsEncodingInput[y_train])
                for i in range(len(weightVectorsEncodingInput[y_train])):
                    weightVectorsEncodingInput[y_train][i] = round(weightVectorsEncodingInput[y_train][i]/norm,12)
                
            """
            erros
            """

            errosHSGS.append(1-resultadoHSGS)
            errosEncodingWeight.append(1-resultadoEncodingWeight)
            errosEncodingInput.append(1-resultadoEncodingInput)
            errosPhaseEncoding.append(1-resultadoPhaseEncoding)

            
        if thresholdTrain == None:
            if (resultadoHSGS < bestErrorHSGS):
                bestWeightsHSGS = weightVectorsHSGS[:]
                bestErrorHSGS = resultadoHSGS
            if (resultadoEncodingWeight < bestErrorEncodingWeight):
                bestWeightsEncodingWeight = weightVectorsEncodingWeight[:]
                bestErrorEncodingWeight = resultadoEncodingWeight
            if (resultadoEncodingInput < bestErrorEncodingInput):
                bestWeightsEncodingInput = weightVectorsEncodingInput[:]
                bestErrorEncodingInput = resultadoEncodingInput
            if (resultadoPhaseEncoding < bestErrorPhaseEncoding):
                bestWeightsPhaseEncoding = weightVectorsPhaseEncoding[:]
                bestErrorPhaseEncoding = resultadoPhaseEncoding
        else:
            erroHSGS += 1-resultadoHSGS
            erroEncodingWeight += 1-resultadoEncodingWeight
            erroEncodingInput += 1-resultadoEncodingInput
            erroPhaseEncoding += 1-resultadoPhaseEncoding


            if (erroHSGS < bestErrorHSGS):
                bestWeightsHSGS = weightVectorsHSGS[:]
                bestErrorHSGS = erroHSGS
            if (erroEncodingWeight < bestErrorEncodingWeight):
                bestWeightsEncodingWeight = weightVectorsEncodingWeight[:]
                bestErrorEncodingWeight = erroEncodingWeight
            if (erroEncodingInput < bestErrorEncodingInput):
                bestWeightsEncodingInput = weightVectorsEncodingInput[:]
                bestErrorEncodingInput = erroEncodingInput
            if (erroPhaseEncoding < bestErrorPhaseEncoding):
                bestWeightsPhaseEncoding = weightVectorsPhaseEncoding[:]
                bestErrorPhaseEncoding = erroPhaseEncoding           
                    

        if epoch_results == True:    
            print("\nerro HSGS", bestErrorHSGS)
            print("erro encoding weight", bestErrorEncodingWeight)
            print("erro encoding input", bestErrorEncodingInput)
            print("erro phase encoding", bestErrorPhaseEncoding)


    if error_by_epoch == True:
        return errosHSGS, errosEncodingWeight, errosEncodingInput, errosPhaseEncoding
    
    print("\nerro HSGS", bestErrorHSGS)
    print("erro encoding weight", bestErrorEncodingWeight)
    print("erro encoding input", bestErrorEncodingInput)
    print("erro phase encoding", bestErrorPhaseEncoding)

    return bestWeightsEncodingWeight, bestWeightsEncodingInput, bestWeightsPhaseEncoding, bestWeightsHSGS


def experiment_TEST(Xs_test, ys_test, weightVectorsEncodingWeight, weightVectorsEncodingInput, weightVectorsPhaseEncoding, weightVectorsHSGS,  thresholdParameter=0.5, lrParameter=0.1, repeat=30, bias=True, testingApproaches={}):
    
    
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


            for neuronClass in range(len(list(set(ys_test)))):

                if ("hsgs" in testingApproaches):
                    operator = "hsgs"
                    wBinaryBinary = deterministicBinarization(weightVectorsHSGS[neuronClass]) # Binarization of Real weights
                    neuron = createNeuron(inputVector, wBinaryBinary, operator)
                    resultadoHSGS1 = executeNeuron(neuron, simulator, threshold=None)
                    if(resultadoHSGS1 > valorMaiorHSGS):
                        neuronMaiorHSGS = neuronClass
                        valorMaiorHSGS = resultadoHSGS1

                if ("phase-encoding" in testingApproaches):
                    operator = "phase-encoding"
                    neuron = createNeuron(inputVector, weightVectorsPhaseEncoding[neuronClass], operator, ancilla=True)
                    resultadoPhaseEncoding1 = executeNeuron(neuron, simulator, threshold=None)
                    if(resultadoPhaseEncoding1 > valorMaiorPhaseEncoding):
                        neuronMaiorPhaseEncoding = neuronClass
                        valorMaiorPhaseEncoding = resultadoPhaseEncoding1
                        
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

                # get predicted probability results
                if neuronClass == 1:
                    outputsHSGS.append(resultadoHSGS1)
                    outputsEncodingWeight.append(resultadoEncodingWeight1)
                    outputsEncodingInput.append(resultadoEncodingInput1)
                    outputsPhaseEncoding.append(resultadoPhaseEncoding1)

            ##################################################
            """
            erros
            """


            erroHSGS_bin = 0
            if (neuronMaiorHSGS != target):   
                erroHSGS_bin = 1

            erroEncodingWeight_bin = 0
            if (neuronMaiorEncodingWeight != target):   
                erroEncodingWeight_bin = 1

            erroEncodingInput_bin = 0
            if (neuronMaiorEncodingInput != target):   
                erroEncodingInput_bin = 1

            erroPhaseEncoding_bin = 0
            if (neuronMaiorPhaseEncoding != target):   
                erroPhaseEncoding_bin = 1


            erroHSGS += erroHSGS_bin####abs(resultadoHSGS_bin-y_train)
            erroEncodingWeight += erroEncodingWeight_bin####abs(resultadoEncoding_bin-y_train)
            erroEncodingInput += erroEncodingInput_bin####abs(resultadoEncoding_bin-y_train)
            erroPhaseEncoding += erroPhaseEncoding_bin####abs(resultadoEncoding_bin-y_train)
 

        print("erro HSGS", erroHSGS/len(Xs_test))
        print("erro encoding weight", erroEncodingWeight/len(Xs_test))
        print("erro encoding input", erroEncodingInput/len(Xs_test))
        print("erro phase encoding", erroPhaseEncoding/len(Xs_test))


        errosHSGS.append(round(erroHSGS/len(Xs_test), 4))
        errosEncodingWeight.append(round(erroEncodingWeight/len(Xs_test), 4))
        errosEncodingInput.append(round(erroEncodingInput/len(Xs_test), 4))
        errosPhaseEncoding.append(round(erroPhaseEncoding/len(Xs_test), 4))

    print("ERROS HSGS            ",  np.average(errosHSGS))
    print("ERROS ENCODING WEIGHT ",  np.average(errosEncodingWeight))
    print("ERROS ENCODING INPUT  ",  np.average(errosEncodingInput))
    print("ERROS PHASE ENCODING  ",  np.average(errosPhaseEncoding))

    """
    results and metrics
    """
    results = { 'error_HSGS': errosHSGS,
                'error_encoding_weight':errosEncodingWeight,
                'error_encoding_input':errosEncodingInput,
                'error_phase_encoding':errosPhaseEncoding,
               
                'output_HSGS': outputsHSGS,
                'output_encoding_weight':outputsEncodingWeight,
                'output_encoding_input':outputsEncodingInput,
                'output_phase_encoding':outputsPhaseEncoding,

                'weights_learned_HSGS':weightVectorsHSGS,
                'weights_learned_encoding_weight':weightVectorsEncodingWeight,
                'weights_learned_encoding_input':weightVectorsEncodingInput,
                'weights_learned_phase_encoding':weightVectorsPhaseEncoding
        }
    return results
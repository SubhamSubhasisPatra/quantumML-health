from neuron import *
from encodingsource import *
from hsgs import *
from classical_neuron import *
from classical_pso import *
from sf import *
simulator = Aer.get_backend('qasm_simulator')
import pandas as pd
import numpy as np


'''
AUXILIARY FUNCTIONS
'''

def inverterSinalPeso(w):
    for i in range(len(w)):
        w[i] *= -1
        
def treinamentoNeuronio(operator, inputVector, weightVector, y_train, lrParameter = 0.09):


    n = int(math.log(len(inputVector), 2))

    if (operator == "hsgs"):
        wBinaryBinary = deterministicBinarization(weightVector) # Binarization of Real weights
        neuron = createNeuron(inputVector, wBinaryBinary, operator)
        #print(operator); print(neuron)
        resultado = executeNeuron(neuron, simulator, threshold=None)
        deltaRule(inputVector, weightVector, lr=lrParameter, y_train=y_train, out=resultado)
        return resultado
    elif (operator == "encoding-weight"):        
        neuron = createNeuron(inputVector, weightVector, operator)
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



def experiment_TRAIN(Xs_train, ys_train, lrParameter=0.09, n_epochs=400, seed=1, trainingBias=True, trainingApproaches={}, error_by_epoch =False):

    np.random.seed(seed)
    weightVectorsHSGS = []
    weightVectorsEncoding = []
    weightVectorsClassico = []
    weightVectorsClassicoBin = []
    input_dim = len(Xs_train[0])

    if (trainingBias):
        for i in range(len(list(set(ys_train)))):
            vRandom = deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim))
            weightVectorsHSGS.append(vRandom.copy())
            weightVectorsEncoding.append(vRandom.copy()) #.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
            weightVectorsClassico.append(vRandom.copy())# = weightVectorsHSGS.copy()#.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
            weightVectorsClassicoBin.append(vRandom.copy()) #= weightVectorsHSGS.copy()#.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
    else:
        for i in range(len(list(set(ys_train)))):
            vRandom = deterministicBinarization(np.random.uniform(-1, 1, input_dim))
            weightVectorsHSGS.append(vRandom.copy())
            weightVectorsEncoding.append(vRandom.copy()) #= weightVectorsHSGS.copy() #.append(deterministicBinarization(np.random.uniform(-1, 1, input_dim)))
            weightVectorsClassico.append(vRandom.copy()) #= weightVectorsHSGS.copy()#.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
            weightVectorsClassicoBin.append(vRandom.copy()) #= weightVectorsHSGS.copy()#.append(deterministicBinarization(np.random.uniform(-1, 1, 2*input_dim)))
            
   


    bestWeightsHSGS = []
    bestWeightsEncoding = []
    bestWeightsClassico = []
    bestErrorHSGS=999999
    bestErrorEncoding = 999999

    bestWeightsHSGSInTime = []
    bestWeightsEncodingInTime = []

    limiarErroToleravel = 0.03
    tamTreinamento = len(Xs_train)

    resultadoHSGS=0
    resultadoEncoding=0
    resultadoClassico=0
    resultadoClassicoBin=0
    
    epoch_errosHSGS=[]
    epoch_errosEncoding = []
  

    for iteration in range(n_epochs):
        erroHSGS = 0
        erroEncoding = 0
        erroClassico=0
        erroClassicoBin=0

        errosHSGS=[]
        errosEncoding = []
        errosClassico=[]
        errosClassicoBin=[]

        for posicaoTreinamento in range(tamTreinamento):
            
            inputVector = Xs_train[posicaoTreinamento] # inputVectors[posicaoTreinamento]
            y_train = ys_train[posicaoTreinamento]
            
            if (trainingBias):
                inputVector = inputVector + len(inputVector)*[1]

            #print(inputVector, y_train)
            
            """
            executando classico
            """
            if ("neuronio-classico" in trainingApproaches):
                operator = "neuronio-classico"
                resultadoClassico = treinamentoNeuronio(operator = operator, 
                                                        inputVector= inputVector, 
                                                        weightVector = weightVectorsClassico[y_train], 
                                                        y_train=1, 
                                                        lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsClassico[y_train])
                for i in range(len(weightVectorsClassico[y_train])):
                    weightVectorsClassico[y_train][i] = round(weightVectorsClassico[y_train][i]/norm,12)
            """
            executando classico binarizado
            """
            if ( "neuronio-classico-bin" in trainingApproaches):
                operator = "neuronio-classico-bin"
                resultadoClassicoBin = treinamentoNeuronio(operator = operator, 
                                                            inputVector= inputVector, 
                                                            weightVector = weightVectorsClassicoBin[y_train], 
                                                            y_train=1, 
                                                            lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsClassicoBin[y_train])
                for i in range(len(weightVectorsClassicoBin[y_train])):
                    weightVectorsClassicoBin[y_train][i] = round(weightVectorsClassicoBin[y_train][i]/norm,12)

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
            executando o encoding
            """
            if ("encoding-weight" in trainingApproaches):
                operator = "encoding-weight"
                resultadoEncoding = treinamentoNeuronio(operator = operator, 
                                                        inputVector= inputVector, 
                                                        weightVector = weightVectorsEncoding[y_train], 
                                                        y_train=1, 
                                                        lrParameter=lrParameter)

                norm = np.linalg.norm(weightVectorsEncoding[y_train])
                for i in range(len(weightVectorsEncoding[y_train])):
                    weightVectorsEncoding[y_train][i] = round(weightVectorsEncoding[y_train][i]/norm,12)
                
            """
            erros
            """

            errosHSGS.append(1-resultadoHSGS)
            errosEncoding.append(1-resultadoEncoding)
            errosClassico.append(1-resultadoClassico)
            errosClassicoBin.append(1-resultadoClassicoBin)

            erroHSGS += abs(1-resultadoHSGS)####abs(resultadoHSGS_bin-y_train)
            erroEncoding += abs(1-resultadoEncoding)####abs(resultadoEncoding_bin-y_train)
            
            erroClassico += abs(1-resultadoClassico)
            erroClassicoBin += abs(1-resultadoClassicoBin)
            
        if (erroHSGS < bestErrorHSGS):
            bestWeightsHSGS = weightVectorsHSGS[:]
            bestErrorHSGS = erroHSGS
        
        if (erroEncoding < bestErrorEncoding):
            bestWeightsEncoding = weightVectorsEncoding[:]
            bestErrorEncoding = erroEncoding
        
        if (iteration % 90 == 0):
            bestWeightsHSGSInTime.append(bestWeightsHSGS)
            bestWeightsEncodingInTime.append(bestWeightsEncoding)
    
            
        print("\nerro HSGS", erroHSGS)
        print("erro encoding", erroEncoding)
        print("melhores erros HSGS / Encoding", bestErrorHSGS, bestErrorEncoding)
        print("erro Classico", erroClassico)
        print("erro classico Bin", erroClassicoBin)
        
        epoch_errosHSGS.append(erroHSGS)
        epoch_errosEncoding.append(erroEncoding)
    


        #if erroEncoding < limiarErroToleravel and erroHSGS < limiarErroToleravel:
        #        break
    print("erros", errosHSGS, errosEncoding)
    print("erros", errosClassico, errosClassicoBin)

    if error_by_epoch == True:
        return epoch_errosHSGS, epoch_errosEncoding

    return bestWeightsEncoding, bestWeightsHSGS, weightVectorsClassico, weightVectorsClassicoBin

def experiment_TEST(Xs_test, ys_test, weightVectorsEncoding, weightVectorsHSGS, weightVectorsClassico, weightVectorsClassicoBin,  thresholdParameter=0.5, lrParameter=0.1, repeat=30, bias=True, testingApproaches={}):
    
    
    errosHSGS = []
    errosEncoding = []
    errosClassico = []
    errosClassicoBin = []

    for i in range(repeat):
        erroHSGS = 0
        erroEncoding = 0
        erroClassico =0
        erroClassicoBin=0

        for posicaoTreinamento in range(len(Xs_test)):
            inputVector = Xs_test[posicaoTreinamento] # inputVectors[posicaoTreinamento]

            if bias == True:
                inputVector = inputVector + len(inputVector)*[1]

            y_train = ys_test[posicaoTreinamento]

            valorMaiorHSGS=0
            neuronMaiorHSGS=0

            valorMaiorEncoding=0
            neuronMaiorEncoding=0

            valorMaiorClassico =0
            neuronMaiorClassico=0

            valorMaiorClassicoBin =0
            neuronMaiorClassicoBin=0

            for neuronClass in range(len(list(set(ys_test)))):

                if ("neuronio-classico" in testingApproaches):        
                    operator="neuronio-classico"    
                    resultadoClassico = runClassicalNeuronReturnProbability(inputVector, weightVectorsClassico[neuronClass])
                    if(resultadoClassico>valorMaiorClassico):
                        neuronMaiorClassico = neuronClass
                        valorMaiorClassico = resultadoClassico
                if ("neuronio-classico-bin" in testingApproaches):
                    operator="neuronio-classico-bin"    
                    wBinaryBinary = deterministicBinarization(weightVectorsClassicoBin[neuronClass]) 
                    resultadoClassicoBin = runClassicalNeuronReturnProbability(inputVector, wBinaryBinary)
                    if(resultadoClassicoBin>valorMaiorClassicoBin):
                        neuronMaiorClassicoBin = neuronClass
                        valorMaiorClassicoBin = resultadoClassicoBin

                if ("hsgs" in testingApproaches):
                    operator = "hsgs"
                    wBinaryBinary = deterministicBinarization(weightVectorsHSGS[neuronClass]) # Binarization of Real weights
                    neuron = createNeuron(inputVector, wBinaryBinary, operator)
                    resultadoHSGS1 = executeNeuron(neuron, simulator, threshold=None)

                    if(resultadoHSGS1>valorMaiorHSGS):
                        neuronMaiorHSGS = neuronClass
                        valorMaiorHSGS = resultadoHSGS1
                if ("encoding-weight" in testingApproaches):
                    operator = "encoding-weight"
                    neuron = createNeuron(inputVector, weightVectorsEncoding[neuronClass], operator)
                    resultadoEncoding1 = executeNeuron(neuron, simulator, threshold=None)

                    if(resultadoEncoding1 > valorMaiorEncoding):
                        neuronMaiorEncoding = neuronClass
                        valorMaiorEncoding = resultadoEncoding1


            """
            erros
            """
            erroClassico_bin = 0
            if (neuronMaiorClassico != y_train):   
                erroClassico_bin = 1

            erroClassicoBin_bin = 0
            if (neuronMaiorClassicoBin != y_train):   
                erroClassicoBin_bin = 1

            erroHSGS_bin = 0
            if (neuronMaiorHSGS != y_train):   
                erroHSGS_bin = 1

            erroEncoding_bin = 0
            if (neuronMaiorEncoding != y_train):   
                erroEncoding_bin = 1

            #print("classe", y_train, "HSGS", neuronMaiorHSGS ,"ENCODING", neuronMaiorEncoding)
            #print("classe", y_train, "Classico", neuronMaiorClassico, "Classico Bin", neuronMaiorClassicoBin)

            erroHSGS += erroHSGS_bin####abs(resultadoHSGS_bin-y_train)
            erroEncoding += erroEncoding_bin####abs(resultadoEncoding_bin-y_train)
            erroClassico += erroClassico_bin####abs(resultadoEncoding_bin-y_train)
            erroClassicoBin += erroClassicoBin_bin####abs(resultadoEncoding_bin-y_train)

        print("erro HSGS", erroHSGS/len(Xs_test))
        print("erro encoding", erroEncoding/len(Xs_test))
        print("erro classico", erroClassico/len(Xs_test))
        print("erro classico bin", erroClassicoBin/len(Xs_test))


        errosHSGS.append(round(erroHSGS/len(Xs_test), 4))
        errosEncoding.append(round(erroEncoding/len(Xs_test), 4))
        errosClassico.append(round(erroClassico/len(Xs_test),4))
        errosClassicoBin.append(round(erroClassicoBin/len(Xs_test),4))

    print("ERROS HSGS        ", errosHSGS, np.average(errosHSGS))
    print("ERROS ENCODING    ", errosEncoding, np.average(errosEncoding))
    print("ERROS Classico    ", errosClassico, np.average(errosClassico))
    print("ERROS Classico Bin", errosClassicoBin, np.average(errosClassicoBin))

    """
    results and metrics
    """
    results = { 'error_HSGS': errosHSGS,
                'error_encoding':errosEncoding,
                'error_classic':errosClassico,
                'error_classic_bin':errosClassicoBin,

                'weights_learned_HSGS':weightVectorsHSGS,
                'weights_learned_encoding':weightVectorsEncoding,
                'weights_learned_classic':weightVectorsClassico,
                'weights_learned_classic_bin':weightVectorsClassicoBin    
        }
    return results
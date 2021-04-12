
import json 
import os
import pandas as pd


#=============================
# results threshold
#=============================

def getData(selecta, selectb, experiment_path, experiment, target_test):

    with open(experiment_path+experiment+'.json') as f:
        data = json.load(f)

    # get outputs for 1st test 
    outputs = ['output_encoding_weight', 'output_encoding_input', 'output_phase_encoding']
    outdata = {}
    for i in outputs:
        outdata[i] = data[i][selecta:selectb]
    outdata = pd.DataFrame(outdata)

    # get true values
    with open(experiment_path + target_test +'.json') as f:
        outdata['flag'] = json.load(f)[1]
    
    return outdata

def searchThreshold(models, output_data, search_space=None):
    # THRESHOLD SEARCH
    model_best = {'model':[], 'threshold':[], 'accuracy':[]}

    for model in models:
        model_search = {}
        if search_space == None:
            search = [0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            search = [search_space[models.index(model)]]
        for threshold in search:
        # get predictions based on threshold
            preds = [1 if x >= threshold else 0 for x in output_data[model]]
            if search_space != None:
                right = list(map(lambda x, y:  1 if x == y else 0, output_data['flag'], preds))
                model_search[threshold] = sum(right)/len(right)
            else:
                if sum(preds) >= int(len(preds)/20): # se pelo menos 5% dos casos foram previstos como 1, continue
                    # get accuracy
                    right = list(map(lambda x, y:  1 if x == y else 0, output_data['flag'], preds))
                    model_search[threshold] = sum(right)/len(right)
        # get threshold with the max value
        model_best['model'].append(model)
        if search_space == None:
            model_best['threshold'].append(max(model_search, key=model_search.get))
            model_best['accuracy'].append(max(model_search.values()))
        else:
            print(model_search)
            model_best['threshold'].append(list(model_search.keys())[0])
            model_best['accuracy'].append(list(model_search.values())[0] )         

    return pd.DataFrame(model_best)



def runSearch(range_value, experiment_path, experiment, target_test):

    outputs = ['output_encoding_weight', 'output_encoding_input', 'output_phase_encoding']

    outdata = getData(0, range_value, experiment_path, experiment, target_test)
    s1 = searchThreshold(outputs, outdata)

    geta = range_value
    getb = range_value*2
    for i in range(9):
        outdata2 = getData(0, range_value, experiment_path, experiment, target_test)
        s2 = searchThreshold(outputs, outdata2, list(s1.threshold))
        s1 = pd.concat([s1, s2])
        geta += range_value
        getb += range_value

    s1.to_csv(experiment_path+experiment+'_search.csv', index=False)

    return s1


experiment_path = 'results/version6/'
target_test = 'test_xor'

runSearch(66, experiment_path, 'experiments_unbiased', target_test)
runSearch(66, experiment_path, 'experiments_biased', target_test)



#=============================
# error by epoch
#=============================

#===================================================================

def readAndResults(i):
    with open('testesout/outputs/'+i) as f:
        data = json.load(f)

    erros = ['error_HSGS', 'error_encoding', 'error_classic', 'error_classic_bin']
    new_data = {'error':[], 'model':[]}

    for erro in erros:
        new_data['error'] = new_data['error'] + data[erro] 
        new_data['model'] = new_data['model'] +  ( (erro+' ') * len(data[erro])).split(' ')[:-1]  

    new_data = pd.DataFrame(new_data)

    new_data.to_csv('testesout/outputs/datasets_accuracy/'+i[:-5]+'.csv', index=False)



#===== dataset 2

readAndResults('dataset2_experiments_original_1noise.json')
readAndResults('dataset2_experiments_bias_1noise.json')

readAndResults('dataset2_experiments_original_2noises.json')
readAndResults('dataset2_experiments_bias_2noises.json')

readAndResults('dataset2_experiments_original_3noises.json')
readAndResults('dataset2_experiments_bias_3noises.json')

#======== dataset 3

readAndResults('dataset3_experiments_original_1noise.json')
readAndResults('dataset3_experiments_bias_1noise.json')

readAndResults('dataset3_experiments_original_2noises.json')
readAndResults('dataset3_experiments_bias_2noises.json')

readAndResults('dataset3_experiments_original_3noises.json')
readAndResults('dataset3_experiments_bias_3noises.json')

#======== dataset 4

readAndResults('dataset4_experiments_original_1noise.json')
readAndResults('dataset4_experiments_bias_1noise.json')

readAndResults('dataset4_experiments_original_2noises.json')
readAndResults('dataset4_experiments_bias_2noises.json')

readAndResults('dataset4_experiments_original_3noises.json')
readAndResults('dataset4_experiments_bias_3noises.json')

#===== dataset 5

readAndResults('dataset5_experiments_original_1noise.json')
readAndResults('dataset5_experiments_bias_1noise.json')

readAndResults('dataset5_experiments_original_2noises.json')
readAndResults('dataset5_experiments_bias_2noises.json')

readAndResults('dataset5_experiments_original_3noises.json')
readAndResults('dataset5_experiments_bias_3noises.json')


#=========================================
from neuron import *
from encodingsource import *
from hsgs import *
from classical_neuron import *
from classical_pso import *
from sf import *
simulator = Aer.get_backend('qasm_simulator')
import pandas as pd
import numpy as np



inputVector = [1,1,1,1,
               1,-1,1,1,
               1,-1,-1,1,
               -1,1,1,-1]


wBinaryBinary = [-1,-1,-1,-1,
                -1,1,-1,-1,
                -1,1,1,-1,
                1,-1,-1,1]


neuron = createNeuron(inputVector, wBinaryBinary, "hsgs")
print(neuron)

#==================================================
# real weights quantum neuron test with 2x2 image
#==================================================

input_= [1,-1, 1, 1]
weights = [0.117,-0.77, -0.177, 0.5]

neuron = createNeuron(input_, weights, "hsgs")
print(neuron)

neuron = createNeuron(input_, weights, "encoding-weight")
print(neuron)


#==================================================
# real weights quantum neuron test with 4x4 image
#==================================================


operator = "encoding-weight"
weights = [-0.064,  0.064 , 0.064, -0.064,
             0.064, 0.487, 0.487,  0.064,
             0.064,  0.487, 0.487,  0.064,
             -0.064, 0.064, 0.064, -0.06]

neuron = createNeuron(inputVector, weights, 'hsgs')
print(neuron)
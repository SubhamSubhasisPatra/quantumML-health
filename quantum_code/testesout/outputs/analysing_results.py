
import json 
import os
import pandas as pd


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
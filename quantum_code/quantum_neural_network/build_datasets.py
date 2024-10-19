'''
Quantum Artificial Neural Network
@claudioalvesmonteiro, 2020
'''

import copy
import json
import pandas as pd

#------------------------------
# build datasets with lines
#------------------------------

def build_lines_data(square_col):
    ''' build list of pixels with horizontal and vertical lines 
        marked with -1, based on squared image of dimension 
        square_col X square_col
    '''
    list_data=[]
    # generate list clean
    example_base = [1]*(square_col*square_col)
    # loop
    cont=0
    while cont < len(example_base):
        example = copy.deepcopy(example_base)
        # track rows
        example[cont:cont+square_col] = [-1]*square_col
        # append to data
        list_data.append(example)
        cont=cont+square_col
    cont=0
    while cont < square_col:
        example = copy.deepcopy(example_base)
        # track columns 
        for i in range(square_col):
            example[cont+(i*square_col)] = -1
        # append to data
        list_data.append(example)
        cont=cont+1    
    return list_data


# gerar dados e salvar
complete_data = {}
for i in range(11):
    complete_data[i] =  build_lines_data(i)

# save all
with open('quantum_neural_network/data.txt', 'w') as outfile:
    json.dump(complete_data, outfile)

# save 4x4
by4 = pd.DataFrame(complete_data[4])
by4['target'] = 1
by4.to_csv('quantum_neural_network/dataset3.csv', index=False)

#------------------------------
# build random datasets images 
#------------------------------

def build_lines_data(square_col):
    ''' build list of pixels with horizontal and vertical lines 
        marked with -1, based on squared image of dimension 
        square_col X square_col
    '''
    list_data=[]
    # generate list clean
    example_base = [1]*(square_col*square_col)
    # loop
    cont=0
    while cont < len(example_base):
        example = copy.deepcopy(example_base)
        # track rows
        example[cont:cont+square_col] = [-1]*square_col
        # append to data
        list_data.append(example)
        cont=cont+square_col
    cont=0
    while cont < square_col:
        example = copy.deepcopy(example_base)
        # track columns 
        for i in range(square_col):
            example[cont+(i*square_col)] = -1
        # append to data
        list_data.append(example)
        cont=cont+1    
    return list_data


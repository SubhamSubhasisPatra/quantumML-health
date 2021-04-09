from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.compiler import transpile, assemble
from sympy.combinatorics.graycode import GrayCode

#from qiskit.circuit.library.standard_gates.multi_control_rotation_gates import _apply_mcu3_graycode, mcrx, mcrz

import numpy as np
import random
import math


def makePhaseEncoding1(angle, n, circuit, ctrls, rotation_ctrl, q_aux, q_target): 
    # Funcao que aplica uma rotacao multi-controlada nos qubits de controle

    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    
    circuit.mcrz(angle, rotation_ctrl, q_target)
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])

    return circuit

def makePhaseEncoding2(angle, n, circuit, ctrls, q_aux, q_target): 
    # Funcao que aplica uma rotacao multi-controlada nos qubits de controle
    circuit.mcrz(angle, q_aux[n-2], q_target)    
    return circuit


def makePhaseEncoding3(angle, n, circuit, ctrls, q_aux, q_target): 
    # Funcao que aplica uma rotacao multi-controlada nos qubits de controle
    m=0
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    
    print(q_aux[len(ctrls)-2])
    circuit.mcrz(angle, q_aux[len(ctrls)-2], q_target)
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])


def normalizePi(input_vector):
    ''' vector normalization for values between 0 and 1
    '''
    
    normalized_vector=[]

    for value in input_vector:
        new_value = ((value-0)/(1-0))*np.pi
        normalized_vector.append(new_value)
    
    return normalized_vector


            
def phaseEncodingGenerator(inputVector, circuit, q_input, nSize, q_aux=None):
    """
    PhaseEncoding Sign-Flip Block Algorithm
    
    inputVector is a Python list 
    eg. inputVector=[1, -1, 1, 1]
    nSize is the input size

    this functions returns the quantum circuit that generates the quantum state 
    whose amplitudes values are the values of inputVector using the SFGenerator approach.
    """ 

    # definindo as posições do vetor onde a amplitude é -1 
    # e tranformando os valores dessas posições em strings binárias
    # conseguindo os estados da base que precisarão ser modificados 
    #positions = findDec(inputVector, nSize)
    #pos_binary = findBin(positions, nSize)
    
    # compute angles
    betas = normalizePi(inputVector)

    # laço para percorrer cada estado base em pos_binay
    for q_basis_state in range(len(betas)):

        # pegando cada posição da string do estado onde o bit é 0
        # aplicando uma porta Pauli-X para invertê-lo
        for indice_position in range(nSize):
            circuit.x(q_input[indice_position])
        
        # identificando controles
        q_bits_controllers = [q_control for q_control in q_input[:nSize-1]]
        q_target = q_input[[nSize-1]]
        
        # aplica phase encoding
        makePhaseEncoding3(betas[q_basis_state], nSize, circuit, q_bits_controllers, q_aux, q_target[0])

        # desfazendo a aplicação da porta Pauli-X nos mesmos qubits
        for indice_position in range(nSize):
                circuit.x(q_input[indice_position])
    return circuit
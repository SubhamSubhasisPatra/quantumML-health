from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.compiler import transpile, assemble
from sympy.combinatorics.graycode import GrayCode

#from qiskit.circuit.library.standard_gates.multi_control_rotation_gates import _apply_mcu3_graycode, mcrx, mcrz

import numpy as np
import random
import math



def decToBin(num, n):# Função para transformar um numero decimal numa string binária de tamanho n
    num_bin = bin(num)[2:].zfill(n)
    return num_bin

def findDec(input_vector, n): # Fução que pega as posições dos fatores -1 do vetor de entrada
    num_dec = []
    for i in range(0, len(input_vector)):
        if input_vector[i] == -1:
            num_dec.append(i)
    return num_dec

def findBin(num_dec, n): # Função que tranforma os numeros das posições em strings binarias
    num_bin = []
    for i in range(0, len(num_dec)):
        num_bin.append(decToBin(num_dec[i], n))
    return num_bin


def makePhaseEncodingV1(pi_angle, n, circuit, ctrls, q_aux, q_target, q_bits_controllers): 

    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
        
    circuit.mcrz(pi_angle, q_bits_controllers, q_target[0])
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])

    return circuit


def makePhaseEncodingV2(pi_angle, n, circuit, ctrls, q_aux, q_target): 
    
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
        
    circuit.mcrz(pi_angle, [q_aux[n-2]], q_target[0])
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])

    return circuit


def makePhaseEncodingV3(pi_angle, circuit, q_target, q_bits_controllers): 
     
    circuit.mcrz(pi_angle, q_bits_controllers, q_target[0])
    
    return circuit
    

def makePhaseEncodingVBin(pi_angle, n, circuit, ctrls, q_aux, q_target): 
         
    circuit.cx(ctrls[0], q_aux[0])
    #for m in range(2, len(ctrls)):
    #    circuit.cx(ctrls[m], q_aux[m-1])
        
    circuit.rz(pi_angle, q_target[0])

    #for m in range(len(ctrls)-1, 1, -1):
    #    circuit.cx(ctrls[m], q_aux[m-1])
    circuit.cx(ctrls[0], q_aux[0])

    return circuit
    

    
def normalizePi(input_vector):
    ''' Pi vector normalization for values between 0 and 1
    '''
    
    normalized_vector=[]

    for value in input_vector:
        new_value = ((value-0)/(1-0))*np.pi
        normalized_vector.append(new_value)
    
    return normalized_vector


            
def phaseEncodingGenerator(inputVector, circuit, q_input, nSize, q_aux=None, phase=3):
    """
    PhaseEncoding Sign-Flip Block Algorithm
    
    inputVector is a Python list 
    nSize is the input size

    this functions returns the quantum circuit that generates the quantum state 
    whose amplitudes values are the values of inputVector using the SFGenerator approach.
    """
    
    # normalizacao Pi para o input_vector
    inputVector = normalizePi(inputVector)
            
    # seleciona as possicioes do vetor 
    # e tranforma os valores dessas posicoes em strings binarias
    # conseguindo os estados da base que precisarao ser modificados 
    
    positions = list(range(len(inputVector)))
    pos_binary = findBin(positions, nSize)

    # laco para percorrer cada estado base em pos_binay
    pi_angle_pos=0
    for q_basis_state in pos_binary:
        # pegando cada posicao da string do estado onde o bit e 0
        # aplicando uma porta Pauli-X para invertê-lo
        for indice_position in range(nSize):
            if q_basis_state[indice_position] == '0':
                circuit.x(q_input[indice_position])
        
        # aplicando porta multi-controlada entres os qubits em q_input
        q_bits_controllers = [q_control for q_control in q_input[:nSize-1]]
        q_target = q_input[[nSize-1]]
        
        # make phase encoding
        #makePhaseEncodingV1(inputVector[pi_angle_pos], nSize, circuit, q_input, q_aux, q_target, q_bits_controllers)
        #makePhaseEncodingV2(inputVector[pi_angle_pos], nSize, circuit, q_input, q_aux, q_target)
        #makePhaseEncodingV3(inputVector[pi_angle_pos], circuit, q_target, q_bits_controllers)
        makePhaseEncodingVBin(inputVector[pi_angle_pos], nSize, circuit, q_input, q_aux, q_target)
        pi_angle_pos+=1
        
        # desfazendo a aplicação da porta Pauli-X nos mesmos qubits
        for indice_position in range(nSize):
            if q_basis_state[indice_position] == '0':
                circuit.x(q_input[indice_position])
    return circuit
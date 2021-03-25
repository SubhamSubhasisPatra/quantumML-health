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

def makePhaseEncoding1(angle, n, circuit, ctrls, rotation_ctrl, q_aux, q_target): 
    # Função que aplica uma porta multi-controlada nos qubits de controle

    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])
    for m in range(2, len(ctrls)):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    
    #########
    circuit.mcrz(angle, rotation_ctrl, q_target)
    
    for m in range(len(ctrls)-1, 1, -1):
        circuit.ccx(ctrls[m], q_aux[m-2], q_aux[m-1])
    circuit.ccx(ctrls[0], ctrls[1], q_aux[0])

    return circuit

def makePhaseEncoding2(angle, n, circuit, ctrls, q_aux, q_target): 
    # Função que aplica uma porta Pauli-Z multi-controlada nos qubits de controle
    circuit.mcrz(angle, q_aux[n-2], q_target[0])    
    return circuit

def recursive_compute_beta(input_vector, betas):
    if len(input_vector) > 1:
        new_x = []
        beta = []
        for k in range(0, len(input_vector), 2):
            norm = np.sqrt(input_vector[k] ** 2 + input_vector[k + 1] ** 2)
            new_x.append(norm)
            if norm == 0:
                beta.append(0)
            else:
                if input_vector[k] < 0:
                    beta.append(2 * np.pi - 2 * np.arcsin(input_vector[k + 1] / norm)) ## testing
                else:
                    beta.append(2 * np.arcsin(input_vector[k + 1] / norm))
        recursive_compute_beta(new_x, betas)
        betas.append(beta)
        output = []
    return betas
            
def phaseEncodingGenerator(inputVector, circuit, q_input, nSize, q_aux=None, phase1=True, ancila = True):
    """
    PhaseEncoding Sign-Flip Block Algorithm
    
    inputVector is a Python list 
    eg. inputVector=[1, -1, 1, 1]
    nSize is the input size

    this functions returns the quantum circuit that generates the quantum state 
    whose amplitudes values are the values of inputVector using the SFGenerator approach.
    """ 
    
    """
    if ancila == True:
        q_aux = QuantumRegister(nSize-1, 'q_aux')
        circuit.add_register(q_aux)
    
    positions = []
    """
    # definindo as posições do vetor onde a amplitude é -1 
    # e tranformando os valores dessas posições em strings binárias
    # conseguindo os estados da base que precisarão ser modificados 
    #positions = findDec(inputVector, nSize)
    #pos_binary = findBin(positions, nSize)
    
    # compute angles
    betas = []
    betas = recursive_compute_beta(inputVector, betas)

    # laço para percorrer cada estado base em pos_binay
    for q_basis_state in range(len(betas)):
        # pegando cada posição da string do estado onde o bit é 0
        # aplicando uma porta Pauli-X para invertê-lo
        for indice_position in range(nSize):
            circuit.x(q_input[indice_position])
        
        # aplicando porta Pauli-Z multi-controlada entres os qubits em q_input
        q_bits_controllers = [q_control for q_control in q_input[:nSize-1]]
        q_target = q_input[[nSize-1]]
        if phase1 == True:
            if (betas[q_basis_state]) == 1:
                makePhaseEncoding1(betas[q_basis_state], nSize, circuit, q_input, q_bits_controllers, q_aux, q_target[0])
            else:
                 for k, angle in enumerate(reversed(betas[q_basis_state])):
                        makePhaseEncoding1(angle, nSize, circuit, q_input, q_bits_controllers, q_aux, q_target[0])
        else:
            if (betas[q_basis_state]) == 1:
                circuit.mcrz(betas[q_basis_state][0], q_bits_controllers, q_target[0])
            else:
                 for k, angle in enumerate(reversed(betas[q_basis_state])):
                    circuit.mcrz(angle, q_bits_controllers, q_target[0])
        
        # desfazendo a aplicação da porta Pauli-X nos mesmos qubits
        for indice_position in range(nSize):
                circuit.x(q_input[indice_position])
    return circuit
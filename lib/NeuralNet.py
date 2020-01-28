'''
The neural network folder will compose of modified seq2seq models
'''

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as tfk
import math
import tensorflow.keras.activations as act
import numpy as np

# Molecular weight of amino acid (3 dec)
mole_weight = {"A": 89.0935, "R": 174.2017, "N": 132.1184,
               "D": 133.1032, "C": 121.1590, "E": 147.1299,
               "Q": 146.1451, "G": 75.0669, "H": 155.1552,
               "I": 131.1736, "L": 131.1736, "K": 146.1882,
               "M": 149.2124, "F": 165.1900, "P": 115.1310,
               "S": 105.0930, "T": 119.1197, "W": 204.2262,
               "Y": 181.1894, "V": 117.1469}

'''
Sequence to Sequence model
'''


class Layers(layers):
    # This class is responsible of generating
    # layers and output size
    def __init__(self, num_Layers=128):
        self.num_Layers = num_Layers


class EncoderLayer(layers):
    def __init__(self, inputs=None):
        self.encode_NN = layers(num_Layers=256)

    def embed_layer(self):
        return self

    def recurrent_layer(self):
        return self  # output vector


class DecoderLayer(layers):
    def __init__(self, vectors=None):
        self.decode_NN = layers(num_Layers=256)

    def embed_layer(self):
        return self

    def recurrent_layer(self):
        return self  # output vector

    def output_layer(self):
        output = self.recurrent_layer()
        # Modify the output using a cost function
        return output


class Seq2Seq(object):  # FIXME: Change the name of the class into resNet
    # This is where the true sequence layer will be formed
    def __init__(self, inputs=None):
        self.EncoderLayer = EncoderLayer(inputs=None)
        self.DecoderLayer = DecoderLayer(vectors=None)
        self.inputs = inputs

    def model_dist(self):
        # The output should form a triangular shape
        # which it is symmetrical
        return self  # returns the distances

    def model_phi(self):
        return self # returns phi angles

    def model_psi(self):
        return self # returns psi angles

class NeuralNetworkTypes(object):
    def __init__(self, sequence):
        self.sequence = sequence
        self.Seq2Seq = Seq2Seq(inputs=None)

    def easy_input(self, angled=True):
        # Organise inputs in specific ways
        if angled == True:
            # Make sure they have an array structure like this (assuming 4 amino acid residues):
            # [0,1,0,0] - One residue to predict on
            return []
        else:
            # Make sure they have an array structure like this (assuming 4 amino acid residues):
            # [0,1,0,1] - Two residues to 'compare'
            return []

    def residualNeuralNetworkAngle(self):
        seq_arr = self.easy_input(angled=True)
        seq_protein = layers.Input(shape=(len(seq_arr),), name='protein_sequence')
        num_unit = 360 * 360
        phi_angle = self.Seq2Seq.model_phi()
        psi_angle = self.Seq2Seq.model_psi()

        mod_phi = tfk.Model(inputs=seq_protein, outputs=phi_angle, name='NanoFold_Model phi_angle')
        mod_psi = tfk.Model(inputs=seq_protein, outputs=psi_angle, name='NanoFold_Model psi_angle')

        return mod_phi, mod_psi

    def residualNeuralNetworkDistance(self):
        #seq_arr = [mole_weight[char] for char in self.sequence]
        seq_arr = self.easy_input(angled=True)

        seq_protein = layers.Input(shape=(len(seq_arr),), name='protein_sequence')
        num_unit = len(seq_arr) * len(seq_arr)  # FIXME: change this into a fixed number
        dist = self.Seq2Seq.model_dist()

        return tfk.Model(inputs=seq_protein, outputs=dist, name='NanoFold_Model dist')

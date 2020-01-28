'''
Class of ResNets
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

# FIXME: Create the pointer network here
'''
Sequence to Sequence model
'''


class Layers(layers):
    # This class is responsible of generating
    # layers and output size
    def __init__(self, num_Layers=128):
        self.num_Layers = num_Layers


class EncoderLayer(layers):
    def __init__(self, inputs):
        self.encode_NN = layers(num_Layers=256)

    def embed_layer(self):
        return self

    def recurrent_layer(self):
        return self  # output vector


class DecoderLayer(layers):
    def __init__(self, vectors):
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
        self.inputs = inputs

    def model(self):
        return self  # returns the output


class NeuralNetworkTypes(object):

    def residualNeuralNetworkAngle(self, sequence):
        seq_arr = [mole_weight[char] for char in sequence]
        seq_protein = layers.Input(shape=(len(seq_arr),), name='protein_sequence')
        num_unit = 360 * 360
        angle = self.residualNeuralNetwork(seq_protein)
        angle = layers.Flatten()(angle)
        angle = layers.Dense(units=num_unit)(angle)  # FIXME: may need this to change into a covolutional

        return tfk.Model(inputs=seq_protein, outputs=angle, name='NanoFold_Model angle')

    def residualNeuralNetworkDistance(self, sequence):
        seq_arr = [mole_weight[char] for char in sequence]
        seq_protein = layers.Input(shape=(len(seq_arr),), name='protein_sequence')
        num_unit = len(seq_arr) * len(seq_arr)  # FIXME: change this into a fixed number
        dist = self.residualNeuralNetwork(seq_protein)
        dist = layers.Flatten()(dist)
        dist = layers.Dense(units=num_unit)(dist)

        return tfk.Model(inputs=seq_protein, outputs=dist, name='NanoFold_Model dist')

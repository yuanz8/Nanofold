'''
The neural network folder will compose of modified seq2seq models
'''

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as tfk
import math
import tensorflow.keras.activations as act
import numpy as np
'''
Sequence to Sequence model:
This model could be improved by using functions specifically tied to 
tensorflow but for learning how the functions of the model work.
'''
'''
class Highway(layers.Layer): # FIXME: decide to delete or keep

  def __init__(self, units=32):
    super(Highway, self).__init__()
    self.units = units

  # Builds required weights nad biases
  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)

    self.w_k = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)

    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

    self.b_k = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)
  # Where the main calculation of the highway blocks
  def call(self, inputs):
    h_function = tf.add(tf.matmul(inputs, self.w_k), self.b_k) #(32, 100)
    t_function = tfk.activations.sigmoid(tf.add(tf.matmul(inputs, self.w), self.b)) #(32,100)
    c_function = tf.constant(1.) - t_function # (32,100)
    # FIXME: inputs * c_function are the wrong shape
    return c_function#tf.multiply(h_function, t_function) + tf.matmul(tf.transpose(inputs), c_function)
'''
class Blocks(): # Contains esidual blocks
    # This class is responsible of generating
    # layers and output size
    def __init__(self, layer_type="Unknown_block"):
        print("Block {} Created.....".format(layer_type))

    # Residual block for encoder
    def encoder_residualBlock(self, last_layer=None, num_nodes_LSTM=64, layer_name="UNKNOWN"):
        resid_layer = layers.BatchNormalization()(last_layer)
        resid_layer = layers.Activation("relu")(resid_layer)
        resid_layer, state_H, state_C = layers.LSTM(num_nodes_LSTM, return_state=True, name=layer_name)(resid_layer)
        resid_layer = layers.Dense(last_layer.shape[len(last_layer.shape)-1], input_shape=(num_nodes_LSTM,), name=layer_name + "-Dense")(resid_layer)
        states = [state_H, state_C]

        return layers.Add()([resid_layer, last_layer]), states

    # Residual block for decoder
    def decoder_residualBlock(self, last_layer=None, encoder_states=None, num_nodes_LSTM=64, layer_name="UNKNOWN"):
        resid_layer = layers.BatchNormalization()(last_layer)
        resid_layer = layers.Activation("relu")(resid_layer)
        resid_layer = layers.LSTM(num_nodes_LSTM, name=layer_name)(resid_layer)
        resid_layer = layers.Dense(last_layer.shape[len(last_layer.shape)-1],
                                   input_shape=(num_nodes_LSTM,), name=layer_name + "-Dense")(resid_layer)
        #resid_layer = layers.BatchNormalization()(resid_layer)
        return layers.Add()([resid_layer, last_layer])

class EncoderLayer(layers.Layer):
    def __init__(self, seq_length=37, type="UNKNOWN"):
        self.blocks = Blocks(layer_type="Encoder_layer_"+type)
        self.seq_length = seq_length

    def embed_layer(self, units=64, e_inputs=None):
        # Embeds the inputs and converts them into vectors
        # for the hidden layers to interpret
        return layers.Embedding(input_dim=self.seq_length, output_dim=units)(e_inputs)

    # The hidden layers in the encoder sequence
    def recurrent_layer(self, prev_vector=None):
        # TODO: Automate the process of creation
        encoderVec, state_0 = self.blocks.encoder_residualBlock(prev_vector, num_nodes_LSTM=32, layer_name="E-RRN-lay0")
        encoderVec, state_1 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=32, layer_name="E-RRN-lay1")
        encoderVec, state_2 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay2")
        encoderVec, state_3 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay3")
        encoderVec, state_4 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay4")
        encoderVec, state_5 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay5")
        encoderVec, state_6 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=256, layer_name="E-RRN-lay6")
        encoderVec, state_7 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=256, layer_name="E-RRN-lay7")
        encoderVec, state_8 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay8")
        encoderVec, state_9 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay9")
        encoderVec, state_10 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay10")
        encoderVec, state_11 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay11")
        encoderVec, state_12 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=32, layer_name="E-RRN-lay12")
        encoderVec, state_13 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=32, layer_name="E-RNN-lay13")
        encoderVec, state_14 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=10, layer_name="E-RRN-lay14")
        encoderVec, state_15 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=10, layer_name="E-RRN-lay15")
        states = [state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8,
                  state_9, state_10, state_11, state_12, state_13, state_14, state_15]
        return states

class RNN_Layer(layers.Layer):
    def __init__(self, seq_length=37, type="UNKNOWN"):
        self.blocks = Blocks(layer_type="RNN_layer_"+type)
        self.seq_length = seq_length

    def embed_layer(self, units=64, e_inputs=None):
        # Embeds the inputs and converts them into vectors
        # for the hidden layers to interpret
        return layers.Embedding(input_dim=self.seq_length, output_dim=units)(e_inputs)

    # The hidden layers in the encoder sequence
    def recurrent_layer(self, prev_vector=None):
        # TODO: Automate the process of creation
        encoderVec, state_0 = self.blocks.encoder_residualBlock(prev_vector, num_nodes_LSTM=32, layer_name="E-RRN-lay0")
        encoderVec, state_1 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=32, layer_name="E-RRN-lay1")
        encoderVec, state_2 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay2")
        encoderVec, state_3 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay3")
        encoderVec, state_4 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay4")
        encoderVec, state_5 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay5")
        encoderVec, state_6 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=256, layer_name="E-RRN-lay6")
        encoderVec, state_7 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=256, layer_name="E-RRN-lay7")
        encoderVec, state_8 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay8")
        encoderVec, state_9 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=128, layer_name="E-RRN-lay9")
        encoderVec, state_10 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay10")
        encoderVec, state_11 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=64, layer_name="E-RRN-lay11")
        encoderVec, state_12 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=32, layer_name="E-RRN-lay12")
        encoderVec, state_13 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=32, layer_name="E-RNN-lay13")
        encoderVec, state_14 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=10, layer_name="E-RRN-lay14")
        encoderVec, state_15 = self.blocks.encoder_residualBlock(encoderVec, num_nodes_LSTM=10, layer_name="E-RRN-lay15")
        states = [state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8,
                  state_9, state_10, state_11, state_12, state_13, state_14, state_15]
        return states

class DecoderLayer(layers.Layer):
    def __init__(self, seq_length=37, states=None, type="UNKNOWN"):
        self.decode_NN = Blocks(layer_type="Decoder_Layer_"+type)
        self.seq_length = seq_length
        self.states = states

    def embed_layer(self, units=64, d_vectors=None):
        # Embeds the inputs and converts them into vectors
        # for the hidden layers to interpret
        return layers.Embedding(input_dim=self.seq_length, output_dim=units)(d_vectors)

    # The hidden layers in the decoding sequence
    def recurrent_layer(self, states=None, prev_vector=None):
        # TODO: Automate the process of creation
        decoderVec = self.decode_NN.decoder_residualBlock(prev_vector, encoder_states=states[0]
                                                          , num_nodes_LSTM=32, layer_name="D-RRN-lay0")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[1]
                                                          , num_nodes_LSTM=32, layer_name="D-RRN-lay1")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[2]
                                                          , num_nodes_LSTM=64, layer_name="D-RRN-lay2")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[3]
                                                          , num_nodes_LSTM=64, layer_name="D-RRN-lay3")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[4]
                                                          , num_nodes_LSTM=128, layer_name="D-RRN-lay4")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[5]
                                                          , num_nodes_LSTM=128, layer_name="D-RRN-lay5")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[6]
                                                          , num_nodes_LSTM=256, layer_name="D-RRN-lay6")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[7]
                                                          , num_nodes_LSTM=256, layer_name="D-RRN-lay7")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[8]
                                                          , num_nodes_LSTM=128, layer_name="D-RRN-lay8")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[9]
                                                          , num_nodes_LSTM=128, layer_name="D-RRN-lay9")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[10]
                                                          , num_nodes_LSTM=64, layer_name="D-RRN-lay10")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[11]
                                                          , num_nodes_LSTM=64, layer_name="D-RRN-lay11")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[12]
                                                          , num_nodes_LSTM=32, layer_name="D-RRN-lay12")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[13]
                                                          , num_nodes_LSTM=32, layer_name="D-RRN-lay13")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[14]
                                                          , num_nodes_LSTM=10, layer_name="D-RRN-lay14")
        decoderVec = self.decode_NN.decoder_residualBlock(decoderVec, encoder_states=states[15]
                                                          , num_nodes_LSTM=10, layer_name="D-RRN-lay15")
        output = layers.Dense(1, activation="softmax")(decoderVec)
        return output

class Seq2Seq(object):  # FIXME: Change the name of the class into resNet
    # This is where the true sequence layer will be formed
    # Model for distance
    def model_dist(self, encoder_inputs=None, decoder_inputs=None):
        encoderDist = EncoderLayer(type="Distance")
        embed_encoder = encoderDist.embed_layer(units=32, e_inputs=encoder_inputs) # TODO: change unit amount
        encoder_states = encoderDist.recurrent_layer(embed_encoder)
        # The resulting states are passed onto the decoding sequence.
        decoderDist = DecoderLayer(type="Distance")
        embed_decoder = decoderDist.embed_layer(units=32, d_vectors=decoder_inputs) # TODO: change unit amount
        decoder_output = decoderDist.recurrent_layer(states=encoder_states)

        return decoder_output # returns the distances

    # Model for phi angle
    def model_phi(self, encoder_inputs=None, decoder_inputs=None):
        phi = RNN_Layer(type="Phi")
        embed_encoder = phi.embed_layer(units=32, e_inputs=encoder_inputs)
        output = phi.recurrent_layer(prev_vector=embed_encoder)
        '''
        encoderPhi = EncoderLayer(type="Phi_Angle")
        embed_encoder = encoderPhi.embed_layer(units=32, e_inputs=encoder_inputs)  # TODO: change unit amount
        encoder_states = encoderPhi.recurrent_layer(prev_vector=embed_encoder)

        decoderPhi = DecoderLayer(type="Phi_Angle")
        embed_decoder = decoderPhi.embed_layer(units=32, d_vectors=decoder_inputs)  # TODO: change unit amount
        decoder_output = decoderPhi.recurrent_layer(states=encoder_states, prev_vector=embed_decoder)
        '''
        return output # returns phi angles

    # Model for psi angle
    def model_psi(self, encoder_inputs=None, decoder_inputs=None):
        phi = RNN_Layer(type="Phi")
        embed_encoder = phi.embed_layer(units=32, e_inputs=encoder_inputs)
        output = phi.recurrent_layer(prev_vector=embed_encoder)
        '''
        encoderPsi = EncoderLayer(type="Psi_Angle")
        embed_encoder = encoderPsi.embed_layer(units=32, e_inputs=encoder_inputs)  # TODO: change unit amount
        encoder_states = encoderPsi.recurrent_layer(prev_vector=embed_encoder)

        decoderPsi = DecoderLayer(type="Psi_Angle")
        embed_decoder = decoderPsi.embed_layer(units=32, d_vectors=decoder_inputs)  # TODO: change unit amount
        decoder_output = decoderPsi.recurrent_layer(states=encoder_states, prev_vector=embed_decoder)
        '''

        return output # returns psi angles

class NeuralNetworkTypes(object):
    def __init__(self, sequence):
        self.sequence = sequence # Feed in as string
        self.NN = Seq2Seq()

    def easy_input(self, angled=True):
        seq_length = len(self.sequence)
        if angled == True:
            # Make sure they have an array structure like this (assuming 4 amino acid residues):
            # [0,1,0,0] - One residue to predict on
            return []
        else:
            # Make sure they have an array structure like this (assuming 4 amino acid residues):
            # [0,1,0,1] - Two residues to 'compare'
            return []

    def residualNeuralNetworkAngle(self):
        #seq_arr = self.easy_input(angled=True) # FIXME: May need to get rid of this later
        seq_protein = layers.Input(shape=(None,), name='protein_sequence')
        num_unit = 360 * 360
        phi_angle = self.NN.model_phi(encoder_inputs=seq_protein, decoder_inputs=seq_protein)
        psi_angle = self.NN.model_psi(encoder_inputs=seq_protein, decoder_inputs=seq_protein)

        mod_phi = tfk.Model(inputs=seq_protein, outputs=phi_angle, name='NanoFold_Model phi_angle')
        mod_psi = tfk.Model(inputs=seq_protein, outputs=psi_angle, name='NanoFold_Model psi_angle')

        return mod_phi, mod_psi

    def residualNeuralNetworkDistance(self):
        #seq_arr = [mole_weight[char] for char in self.sequence]
        seq_arr = self.easy_input(angled=True)  # FIXME: May need to get rid of this later
        seq_protein = layers.Input(shape=(None,), name='protein_sequence')
        dist = self.NN.model_dist()

        return tfk.Model(inputs=seq_protein, outputs=dist, name='NanoFold_Model distance')

NN = NeuralNetworkTypes("QWERER")
phi, psi = NN.residualNeuralNetworkAngle()
phi.summary()
'''
# Testing highway networks
x = tf.ones((32,12))

highway_layer = Highway(100)  # At instantiation, we don't know on what inputs this is going to get called
y = highway_layer(x)  # The layer's weights are created dynamically the first time the layer is called
print(y.shape)
print(highway_layer.weights)
print(len(highway_layer.trainable_weights))
'''
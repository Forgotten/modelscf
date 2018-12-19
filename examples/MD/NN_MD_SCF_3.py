"""
script that builds an Mnn-H2 network, loads the pre-trained network data
and it encapsulates it in order to be called by julia

"""
# ------------------ keras ----------------
from keras.models import Sequential, Model
# layers
from keras.layers import Input, Activation, Flatten
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D
from keras.layers import BatchNormalization, Add, multiply, dot, Reshape, SeparableConv1D, add
from keras.layers import Lambda


from keras import backend as K
from keras import regularizers, optimizers
from keras.engine.topology import Layer
from keras.constraints import non_neg
from keras.utils import np_utils
#from keras.utils import plot_model

K.set_floatx('float32')

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import timeit
import argparse
import h5py
import numpy as np
import random
#np.random.seed(123)  # for reproducibility
#random.seed(123)
import math

# important to be able to find the symbols
import sys
sys.path.insert(0, '../')

# periodic padding layer, for a more general case
class periodic_padding_tensor(Layer):
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        assert(pad_size%2 == 0)
        super(periodic_padding_tensor, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(periodic_padding_tensor, self).build(input_shapes)

    def call(self, inputs):
        x = inputs
        size_x = x.shape[1]
        t1_pad = x[:,size_x-self.pad_size//2:size_x]
        t2_pad = x[:,0:self.pad_size//2,:]
        y = K.concatenate([t1_pad, x, t2_pad], axis=1)
        return y

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0],input_shapes[1]+self.pad_size,input_shapes[2])


class NN_MD():

  def __init__(self):
    script__name = 'Hmatrix_1D_py'

    # epoch  =
    alpha  = 12
    L      = 5
    N_cnn  = 6
    N_train  = 29500
    N_test   = 500
    Nx = 64
    input_prefix = 'KS_MD_scf_3'
    # setup: parameters

    lr       = 0.001

    data_folder = '../data/'
    log_folder = '../logs/'
    models_folder = '' # folder to save the models (weights)

    n_input = Nx
    nn_output = Nx

    # parameters
    w_size = Nx // (2**(L - 1))
    #w_size = 2 * ((w_size+1)//2) - 1 # if w_size is even, set it as w_size-1
    mean_rho = 48.

    # size of the window for the non-linear CNN in the processing layers
    w_cnn = 7

    # Defining specific layers
    def ResetAverage(x):
        #return x + (8 - K.sum(x,axis=1, keepdims=True)) * 0.05
        x2 = K.clip(x, 0, 1000)
        axis_sum = tuple(range(np.size(x2.shape))[1:])
        factor = mean_rho / K.sum(x2, axis=axis_sum, keepdims=True)
        return x2 * factor

    w_cnn = 7

    w = int(Nx//(2**(L)))
    w_adj = w
    print(w)

    # TODO: clean the code
    Ipt = Input(shape=(n_input, 1))

    Layers = []
    models = []
    for k in range(0, L-2):
        w = w_size * 2**(L-k-3)
        #restriction: we can add more layers
        Av_l = Conv1D(alpha, w, strides=w, activation='linear')(Ipt)

        ########### deep but thin network  ############
        if(k==0):
            for i in range(0,N_cnn):
                w_cnn_coarse = w_cnn - 2
                Av_l = periodic_padding_tensor(pad_size=w_cnn_coarse-1)(Av_l)
                Av_l = Conv1D(alpha, w_cnn_coarse, activation='relu')(Av_l)
        else:
            for i in range(0,N_cnn):
                Av_l = periodic_padding_tensor(pad_size=w_cnn-1)(Av_l)
                Av_l = Conv1D(alpha, w_cnn, activation='relu')(Av_l)

        Av_l = Conv1D(w, 1, activation='linear')(Av_l)
        Av_l = Flatten()(Av_l)
        Layers.append(Av_l)


    Av_adj = Reshape((Nx//w_size, w_size))(Ipt)

    for i in range(0, N_cnn-1):
        Av_adj = periodic_padding_tensor(pad_size=2)(Av_adj)
        Av_adj = Conv1D(w_size, 3, activation='relu')(Av_adj)

    Av_adj = periodic_padding_tensor(pad_size=2)(Av_adj)
    Av_adj = Conv1D(w_size, 3, activation='linear')(Av_adj)
    Av_adj = Flatten()(Av_adj)

    Layers.append(Av_adj)

    Opt = Add()(Layers)
    Opt2 = Lambda(ResetAverage)(Opt)

    str_best_model =  models_folder  +   \
                  'weights_'  + script__name +   \
                  'Input_'    + input_prefix + \
                  '_alpha_'   + str(alpha) + \
                  '_L_'       + str(L) + \
                  '_N_cnn_'   + str(N_cnn) + \
                  '_n_train_' + str(N_train) + \
                  '_n_test_'  + str(N_test) +'_best.h5'
    # building the model and getting to work with multiple GPU's

    # loading the weights of the best model found in the training step
    model = Model(inputs=Ipt, outputs=Opt2)

    model.load_weights(str_best_model, by_name=False)

    model.compile(loss='mean_squared_error', optimizer='Nadam')#,  metrics=['accuracy'])
    model.summary()

    self.model = model

  def eval(self,x):
    q = self.model.predict(np.reshape(x, (1,x.size,1)))
    return np.reshape(q, (q.size,))

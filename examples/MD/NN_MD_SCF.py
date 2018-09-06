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
import sys
sys.path.insert(0, '../')

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

def recursiveCNN(N_cnn,alpha,w_cnn,w, Layer, level,Nx):
      # we use a Conv1D layer to subsample the tensor from Nx, alpha to
      # Nx/w, alpha
      Layer = Conv1D(alpha, w, strides=w, activation='linear')(Layer)
  
      if level > 0:
          assert Nx%2 == 0
  
          # recursive call to a coarser level
          shortcut = recursiveCNN(N_cnn,alpha,w_cnn, w, Layer, level-1, int(Nx//2))
  
      if level == 0:
        #coarser level
          w_cnn = 3
  
      for kk in range(0, N_cnn):
          Layer = periodic_padding_tensor(pad_size=w_cnn-1)(Layer)
          Layer = Conv1D(alpha, w_cnn, strides=1, activation='relu')(Layer)
  
      if level > 0:
          assert Nx%2 == 0
          Layer = add([Layer,shortcut])
  
      Layer = Conv1D(w*alpha, 1, strides=1, activation='linear')(Layer)
      Layer = Reshape((Nx, alpha))(Layer)
  
      return Layer

class NN_MD():

  def __init__(self):
    script__name = 'mnnh2_data_aug'
    
    # epoch  =
    alpha  = 10
    L      = 5
    N_cnn  = 5
    N_train  = 19500
    N_test   = 500
    Nx = 64
    input_prefix = 'KS_MD_scf_2_sigma_1.0'
    # setup: parameters
    
    lr       = 0.001
    Augfactor= 10
    
    data_folder = '../data/'
    log_folder = '../logs/'
    models_folder = '' # folder to save the models (weights)
  
    N_input = Nx
    N_output = Nx
    
    # parameters
    w_size = Nx // (2**(L - 1))
    #w_size = 2 * ((w_size+1)//2) - 1 # if w_size is even, set it as w_size-1
    mean_rho = 32.
    
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
    Ipt = Input(shape=(Nx, 1))
    
    # local basis
    Layer = Conv1D(alpha, w, strides=w, activation='linear')(Ipt)
    
    shortcut = recursiveCNN(N_cnn,alpha,w_cnn,2, Layer,L-3, Nx//w)
    
    # print(w_cnn)
    for i in range(0, N_cnn):
        Layer = periodic_padding_tensor(pad_size=w_cnn-1)(Layer)
        Layer = Conv1D(alpha, w_cnn, activation='relu')(Layer)
    
    Layer = add([Layer,shortcut])
    
    Layer = Conv1D(w, 1, strides=1, activation='linear')(Layer)
    LayerH = Flatten()(Layer)
    
    #adjacent part
    Layer = Reshape((Nx//w_adj, w_adj))(Ipt);
    for i in range(0, N_cnn-1):
        Layer = periodic_padding_tensor(pad_size=2)(Layer)
        Layer = Conv1D(w_adj, 3, activation='relu')(Layer)
    
    Layer = periodic_padding_tensor(pad_size=2)(Layer)
    Layer = Conv1D(w_adj, 3, activation='linear')(Layer)
    Layer = Flatten()(Layer)
    
    Opt = Add()([Layer, LayerH])
    Opt2 = Lambda(ResetAverage)(Opt)
    
    str_best_model =  models_folder  +   \
                  'weights_'  + script__name +   \
                  'Input_'    + input_prefix + \
                  '_alpha_'   + str(alpha) + \
                  '_L_'       + str(L) + \
                  '_lr_'      + str(lr) + \
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
import tensorflow as tf
import numpy as np
from dnn_bl01_para  import *

class dnn_bl01_conf(object):

    def __init__(self, input_layer_val, input_layer_val_dim, mode):

        self.dnn_para = dnn_bl01_para()
        
        self.input_layer_val     = input_layer_val 
        self.input_layer_val_dim = input_layer_val_dim 

        self.mode = mode

        with tf.device('/cpu:0'), tf.variable_scope("fully_connected_layer01") as scope:
            self.output_layer01 = self.fully_layer(self.input_layer_val,
                                                   self.input_layer_val_dim,

                                                   self.dnn_para.l01_fc, 

                                                   self.dnn_para.l01_is_act,
                                                   self.dnn_para.l01_act_func, 

                                                   self.dnn_para.l01_is_drop,
                                                   self.dnn_para.l01_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )
        ### ======== Layer 02: full connection
        with tf.device('/cpu:0'), tf.variable_scope("fully_connected_layer02") as scope:
            self.output_layer02 = self.fully_layer(self.output_layer01,
                                                   self.dnn_para.l01_fc,

                                                   self.dnn_para.l02_fc, 

                                                   self.dnn_para.l02_is_act,
                                                   self.dnn_para.l02_act_func, 

                                                   self.dnn_para.l02_is_drop,
                                                   self.dnn_para.l02_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )
        ### ======== Layer 03: full connection
        with tf.device('/cpu:0'), tf.variable_scope("fully_connected_layer03") as scope:
            self.output_layer03 = self.fully_layer(self.output_layer02,
                                                   self.dnn_para.l02_fc,

                                                   self.dnn_para.l03_fc, 

                                                   self.dnn_para.l03_is_act,
                                                   self.dnn_para.l03_act_func, 

                                                   self.dnn_para.l03_is_drop,
                                                   self.dnn_para.l03_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )
        ### ======== Layer 04: full connection
        with tf.device('/cpu:0'), tf.variable_scope("fully_connected_layer04") as scope:
            self.output_layer04 = self.fully_layer(self.output_layer03,
                                                   self.dnn_para.l03_fc,

                                                   self.dnn_para.l04_fc, 

                                                   self.dnn_para.l04_is_act,
                                                   self.dnn_para.l04_act_func, 

                                                   self.dnn_para.l04_is_drop,
                                                   self.dnn_para.l04_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )


            self.final_output = self.output_layer04
 
    ### 02/ FULL CONNECTTION  LAYER
    def fully_layer(self, 
                    input_val, 
                    input_size, 
                    output_size, 
                    is_act,
                    act_func,
                    is_drop,
                    drop_prob, 
                    mode,
                    scope=None
                   ):

        with tf.variable_scope(scope or 'fully-connected_layer') as scope:

            #initial parameter
            W    = 0.1*tf.random_normal([input_size, output_size], stddev=1., dtype=tf.float32)
            bias = 0.1*tf.random_normal([output_size], stddev=1., dtype=tf.float32) 
            W    = tf.Variable(W)
            bias = tf.Variable(bias)

            #Dense 
            dense_output = tf.add(tf.matmul(input_val, W), bias)  

            #Active function
            if(is_act == True):
                if (act_func == 'RELU'):    
                    act_func_output = tf.nn.relu(dense_output)   
                elif (act_func == 'SOFTMAX'):
                    act_func_output  = tf.nn.softmax(dense_output)             
                elif (act_func == 'TANH'):
                    act_func_output  = tf.nn.tanh(dense_output)                 
                elif (act_func == 'GELU'):
                    act_func_output = self.gelu(dense_output)
                elif (act_func == 'SELU'):
                    act_func_output = self.selu(dense_output)
                elif (act_func == 'ISRLU'):
                    act_func_output = self.isrlu(dense_output)
                else:
                    act_func_output = dense_output
            else:
                act_func_output = dense_output

            #Drop out
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                act_func_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:
                drop_output = act_func_output

            #Return 
            return drop_output

#===================================================================================
#===========================new activation function ================================
    def gelu(self,x):
        """
        GELU activation function
        """
        with tf.variable_scope("GELU"):
            cdf = 0.5*(1.0+tf.tanh(np.sqrt(2 / np.pi)*(x+(0.044715+tf.pow(x,3)))))
            return x*cdf

    def selu(self, x):
        """
        SELU activation function
        """
        with tf.variable_scope("SELU"):
            scale_constant = 1.0507
            alpha = 1.6732
            threshold = 0.
            condition = tf.greater(x,threshold)
            output = tf.where(condition,scale_constant*x,scale_constant*alpha*(tf.exp(x)-1))
            return output

    def isrlu(self, x):
        """
        ISRLU activation function
        """
        with tf.variable_scope("ISRLU"):
            alpha = 2. # value of alpha is between 1. to 3.
            condition = tf.greater_equal(x,0.)
            output = tf.where(condition, x, x*(1./tf.sqrt(1 + alpha*tf.pow(x,2))))
            return output

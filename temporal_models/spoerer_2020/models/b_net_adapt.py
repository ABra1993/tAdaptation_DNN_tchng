'''
Keras implementation of B networks
'''

import tensorflow as tf
import numpy as np
import sys

class BConvLayer(object):
    '''BL recurrent convolutional layer
    Note that this is NOT A KERAS LAYER but is an object containing Keras layers
    Args:
        filters: Int, number of output filters in convolutions
        kernel_size: Int or tuple/list of 2 integers, specifying the height and
            width of the 2D convolution window. Can be a single integer to
            specify the same value for all spatial dimensions.
        layer_name: String, prefix for layers in the RCL
        '''

    def __init__(self, filters, kernel_size, layer_name, alpha, beta):

        # adaptation values
        self.alpha = tf.constant(alpha)
        self.beta = tf.constant(beta)
        self.layer_name = layer_name

        # initialise convolutional layers
        self.b_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            name='{}_BConv'.format(self.layer_name))

        # holds the most recent bottom-up conv
        # useful when the bottom-up input does not change, e.g. input image
        self.previous_b_conv = None

    def __call__(self, t, n, b_input=None, l_input=None, s_input=None):

        if not b_input is None: # run bottom-up conv and save result
            b_input_current = self.b_conv(b_input)
            self.previous_b_conv = b_input_current
        elif not self.previous_b_conv is None: # use the most recent bottom-up conv
            b_input_current = self.previous_b_conv
        else:
            raise ValueError('b_input must be given on first pass')

        # comput current suppression state
        if not s_input is None:

            # layer for summing convolutions
            sum_convs = tf.keras.layers.Lambda(
                tf.math.add_n, name='{}_S_Time_{}'.format(n, t))

            # compute activation with intrinsic suppression
            s_previous = tf.math.multiply(self.alpha, s_input)
            r_previous = tf.math.multiply(tf.math.subtract(1, self.alpha), l_input)

            s_current = sum_convs([s_previous, r_previous])
            # s_current = tf.math.add(s_previous, r_previous, name='{}_S_Time_{}'.format(n, t))
            r_current = tf.math.subtract(b_input_current, tf.math.multiply(self.beta, s_current))

            # return element-wise sum of convolutions
            return r_current, s_current

        else:
            return b_input_current

def b_net_adapt(input_tensor, classes, model_arch, alpha=0.96, beta=0.7, n_timesteps=8, cumulative_readout=False):
        """ Returns a feedforward B-model with intrinsic adaptation
        """

        data_format = tf.keras.backend.image_data_format()
        norm_axis = -1 if data_format == 'channels_last' else -3

        # initialise trainable layers (ACLs and linear readout)
        if model_arch == 'b':
            layers = [
                BConvLayer(96, 7, 'ACL_0', alpha, beta),
                BConvLayer(128, 5, 'ACL_1', alpha, beta),
                BConvLayer(192, 3, 'ACL_2', alpha, beta),
                BConvLayer(256, 3, 'ACL_3', alpha, beta),
                BConvLayer(512, 3, 'ACL_4', alpha, beta),
                BConvLayer(1024, 3, 'ACL_5', alpha, beta),
                BConvLayer(2048, 1, 'ACL_6', alpha, beta),
            ]
        elif model_arch == 'b_k':
            layers = [
                BConvLayer(96, 11, 'ACL_0', alpha, beta),
                BConvLayer(128, 7, 'ACL_1', alpha, beta),
                BConvLayer(192, 5, 'ACL_2', alpha, beta),
                BConvLayer(256, 5, 'ACL_3', alpha, beta),
                BConvLayer(512, 5, 'ACL_4', alpha, beta),
                BConvLayer(1024, 5, 'ACL_5', alpha, beta),
                BConvLayer(2048, 3, 'ACL_6', alpha, beta),
            ]

        elif model_arch == 'b_f':
            layers = [
                BConvLayer(192, 7, 'ACL_0', alpha, beta),
                BConvLayer(256, 5, 'ACL_1', alpha, beta),
                BConvLayer(384, 3, 'ACL_2', alpha, beta),
                BConvLayer(512, 3, 'ACL_3', alpha, beta),
                BConvLayer(1024, 3, 'ACL_4', alpha, beta),
                BConvLayer(2048, 3, 'ACL_5', alpha, beta),
                BConvLayer(4096, 1, 'ACL_6', alpha, beta),
            ]
        elif model_arch == 'b_d':

            layers = [
                BConvLayer(96, 7, 'ACL_0', alpha, beta),
                BConvLayer(96, 7, 'ACL_1', alpha, beta),
                BConvLayer(128, 5, 'ACL_2', alpha, beta),
                BConvLayer(128, 5, 'ACL_3', alpha, beta),
                BConvLayer(192, 3, 'ACL_4', alpha, beta),
                BConvLayer(192, 3, 'ACL_5', alpha, beta),
                BConvLayer(256, 3, 'ACL_6', alpha, beta),
                BConvLayer(256, 3, 'ACL_7', alpha, beta),
                BConvLayer(512, 3, 'ACL_8', alpha, beta),
                BConvLayer(512, 3, 'ACL_9', alpha, beta),
                BConvLayer(1024, 3, 'ACL_10', alpha, beta),
                BConvLayer(1024, 3, 'ACL_11', alpha, beta),
                BConvLayer(2048, 1, 'ACL_12', alpha, beta),
                BConvLayer(2048, 1, 'ACL_13', alpha, beta),
            ]

        else:
            print('Model architecture does not exist')
            sys.exit()

        # output layer
        readout_dense = tf.keras.layers.Dense(
                classes, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                name='ReadoutDense')

        # initialise list for activations and outputs
        n_layers = len(layers)
        activations = [[None for _ in range(n_layers)]
                       for _ in range(n_timesteps)]

        s = [[None for _ in range(n_layers)]
                for _ in range(n_timesteps)]

        presoftmax = [None for _ in range(n_timesteps)]
        outputs = [None for _ in range(n_timesteps)]

        # build the model
        for t in range(n_timesteps):
            for n, layer in enumerate(layers):

                # get the bottom-up input
                if n == 0:

                    # B conv on the image does not need to be recomputed
                    if n == 0:
                        b_input = input_tensor[:, t, :, :, :]
                        print(input_tensor.shape)
                    else:
                        b_input =  None

                else:

                    if model_arch != 'b_d':

                        # pool b_input for all layers apart from input
                        b_input = tf.keras.layers.MaxPool2D(
                            pool_size=(2, 2),
                            name='MaxPool_Layer_{}_Time_{}'.format(n, t)
                            )(activations[t][n-1])

                    else:

                        if n % 2 == 0:

                            # pool b_input for all layers apart from input
                            b_input = tf.keras.layers.MaxPool2D(
                                pool_size=(2, 2),
                                name='MaxPool_Layer_{}_Time_{}'.format(n, t)
                                )(activations[t][n-1])

                # get the lateral input and suppression state
                if t == 0:
                    l_input = None
                    s_input = None
                    x_tn = layer(t, n, b_input, l_input, s_input)
                else:
                    l_input = activations[t-1][n]
                    s_input = s[t-1][n]
                    x_tn, s_current = layer(t, n, b_input, l_input, s_input)

                # batch normalization
                x_tn = tf.keras.layers.BatchNormalization(
                    norm_axis,
                    name='BatchNorm_Layer_{}_Time_{}'.format(n, t))(x_tn)

                # ReLU
                activations[t][n] = tf.keras.layers.Activation(
                    'relu', name='ReLU_Layer_{}_Time_{}'.format(n, t))(x_tn)

                if t == 0:
                    s[t][n] = tf.keras.layers.Activation(
                        'relu', name=None)(x_tn*0)
                else:
                    s[t][n] = s_current

            # add the readout layers
            x = tf.keras.layers.GlobalAvgPool2D(
                name='GlobalAvgPool_Time_{}'.format(t)
                )(activations[t][n-1])
            presoftmax[t] = readout_dense(x)

            # select cumulative or instant readout
            if cumulative_readout and t > 0:
                x = tf.keras.layers.Add(
                    name='CumulativeReadout_Time_{}'.format(t)
                    )(presoftmax[:t+1])
            else:
                x = presoftmax[t]
            outputs[t] = tf.keras.layers.Activation(
                'softmax', name='Sotfmax_Time_{}'.format(t))(x)

        # create Keras model and return
        model = tf.keras.Model(
            inputs=input_tensor,
            outputs=outputs,
            name='b_net_adapt')

        return model

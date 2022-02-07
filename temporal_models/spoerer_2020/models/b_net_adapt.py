'''
Keras implementation of B networks
'''

import tensorflow as tf
import numpy as np
import sys


class BConvLayer(object):

    def __init__(self, filters, kernel_size, layer_name, alpha, beta):
        '''Base layer for B models with intrinsic adaptation

        Note that this is NOT A KERAS LAYER but is an object containing Keras layers

        Args:
            filters: Int, number of output filters in convolutions
            kernel_size: Int or tuple/list of 2 integers, specifying the height and
                width of the 2D convolution window. Can be a single integer to
                specify the same value for all spatial dimensions.
            layer_name: String, prefix for layers in the RCL
            '''

        # adaptation values
        self.alpha = alpha
        self.beta = beta

        # initialise convolutional layer
        self.b_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False,               # padding with zeros evenly to the left/right or up/down of the input.
            kernel_initializer='glorot_uniform',                                # the Glorot methods allows for scaling the weight distribution on a layer-by-layer basis.
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            name='{}_BConv'.format(layer_name))

        self.comp_s = tf.keras.layers.Lambda(
            tf.add_n, name='{}_CompS'.format(layer_name))

        self.sum_convs = tf.keras.layers.Lambda(
            tf.add_n, name='{}_ConvSum'.format(layer_name))

    def __call__(self, b_input=None, l_input=None, s_input=None):

        if (l_input == None) & (s_input == None):

            b_current = self.b_conv(b_input)
            return b_current, None

        else:

            b_current = self.b_conv(b_input)
            s_current = self.comp_s([self.alpha * s_input, (1 - self.alpha) * l_input])
            r_current = self.sum_convs([b_current, -self.beta * s_current])

            return r_current, s_current


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

            print('Model architecture still needs to be implemented...')
            sys.exit()
        #     layers = [
        #         BConvLayer(96, 7, 'ACL_0'),
        #         BConvLayer(96, 7, 'ACL_1'),
        #         BConvLayer(128, 5, 'ACL_2'),
        #         BConvLayer(128, 5, 'ACL_3'),
        #         BConvLayer(192, 3, 'ACL_4'),
        #         BConvLayer(192, 3, 'ACL_5'),
        #         BConvLayer(256, 3, 'ACL_6'),
        #         BConvLayer(256, 3, 'ACL_7'),
        #         BConvLayer(512, 3, 'ACL_8'),
        #         BConvLayer(512, 3, 'ACL_9'),
        #         BConvLayer(1024, 3, 'ACL_10'),
        #         BConvLayer(1024, 3, 'ACL_11'),
        #         BConvLayer(2048, 1, 'ACL_12'),
        #         BConvLayer(2048, 1, 'ACL_13'),
        #     ]
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

                if (n == 0) | (n % 2 == 0): # (layer.layer_name[-1] == '1') |
                    b_input = input_tensor
                else:
                    # print('MaxPool_Layer_{}_Time_{}'.format(n, t))
                    b_input = tf.keras.layers.MaxPool2D(
                        pool_size=(2, 2),
                        name='MaxPool_Layer_{}_Time_{}'.format(n, t)
                        )(activations[t][n-1])

                # get the lateral input and suppression state
                if t == 0:
                    l_input = None
                    s_input = None
                else:
                    l_input = activations[t-1][n]
                    s_input = s[t-1][n]

                # convolutions
                x_tn, s_current = layer(b_input, l_input, s_input)

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

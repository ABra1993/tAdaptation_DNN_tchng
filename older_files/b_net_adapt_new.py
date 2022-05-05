'''
Keras implementation of B networks
'''

import tensorflow as tf
import numpy as np
import sys

n_timesteps = 8
activations = [[None for _ in range(7)] for _ in range(n_timesteps)]
suppressions = [[None for _ in range(7)] for _ in range(n_timesteps)]
b_inputs = [None for _ in range(7*n_timesteps)]
presoftmax = [None for _ in range(n_timesteps)]
outputs = [None for _ in range(n_timesteps)]
trackb = None

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

    def __init__(self, timestep, filters, kernel_size, layer_name, alpha, beta):

        global b_inputs

        # adaptation values
        self.alpha = tf.constant(alpha)
        self.beta = tf.constant(beta)

        # initialise convolutional layers
        self.b_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding='same', use_bias=False,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            name='{}_BConv_{}'.format(layer_name, timestep))

        # holds the most recent bottom-up conv
        # useful when the bottom-up input does not change, e.g. input image
        if timestep == 0:
            self.previous_b_conv = None
        else: self.previous_b_conv = b_inputs[(timestep-1)*7]

    def __call__(self, timestep, n, b_input, l_input=None, s_input=None):

        global b_inputs
        global trackb

        # Get current index of b_input list (current timestep*amount of layers + current layer)
        current_idx = (timestep*7)+n

        # Return same b_input at last timestep and last layer to prevent index error
        if current_idx-1 == n_timesteps*7:
            return b_inputs[current_idx]

        if not b_inputs[current_idx] is None: # run bottom-up conv and save result
            b_inputs[current_idx+1]= self.b_conv(b_inputs[current_idx])
            self.previous_b_conv = b_inputs[current_idx+1]

        elif not self.previous_b_conv is None: # use the most recent bottom-up conv
            b_inputs[current_idx+1] = trackb

        else:
            raise ValueError('b_input must be given on first pass')

        if n == timestep == 0:
            trackb = self.previous_b_conv

        # compute current suppression state
        if not s_input is None:

            # compute activation with intrinsic suppression
            s_current = tf.math.add(tf.math.multiply(self.alpha, s_input, name='multiply1_{}'.format(timestep)), tf.math.multiply(tf.math.subtract(1, self.alpha, name='subtract1_{}'.format(timestep)), l_input, name='multiply2_{}'.format(timestep)), name='add_{}'.format(timestep))
            r_current = tf.math.subtract(b_inputs[current_idx+1], tf.math.multiply(self.beta, s_current, name='multiply3_{}'.format(timestep)), name='subtract2_{}'.format(timestep))

            # return element-wise sum of convolutions
            return r_current, s_current

        else: return b_input[current_idx+1]



def b_net_adapt(input_tensor, classes, model_arch, timestep, alpha=0.96, beta=0.7, cumulative_readout=False):
        """ Returns a feedforward B-model with intrinsic adaptation

        """
        global activations
        global suppressions
        global b_inputs
        global outputs

        data_format = tf.keras.backend.image_data_format()
        norm_axis = -1 if data_format == 'channels_last' else -3

        # initialise trainable layers (ACLs and linear readout)
        if model_arch == 'b':
            layers = [
                BConvLayer(timestep, 96, 7, 'ACL_0'+str(timestep), alpha, beta),
                BConvLayer(timestep, 128, 5, 'ACL_1'+str(timestep), alpha, beta),
                BConvLayer(timestep, 192, 3, 'ACL_2'+str(timestep), alpha, beta),
                BConvLayer(timestep, 256, 3, 'ACL_3'+str(timestep), alpha, beta),
                BConvLayer(timestep, 512, 3, 'ACL_4'+str(timestep), alpha, beta),
                BConvLayer(timestep, 1024, 3, 'ACL_5'+str(timestep), alpha, beta),
                BConvLayer(timestep, 2048, 1, 'ACL_6'+str(timestep), alpha, beta),
            ]
        elif model_arch == 'b_k':
            layers = [
                BConvLayer(timestep, 96, 11, 'ACL_0', alpha, beta),
                BConvLayer(timestep, 128, 7, 'ACL_1', alpha, beta),
                BConvLayer(timestep, 192, 5, 'ACL_2', alpha, beta),
                BConvLayer(timestep, 256, 5, 'ACL_3', alpha, beta),
                BConvLayer(timestep, 512, 5, 'ACL_4', alpha, beta),
                BConvLayer(timestep, 1024, 5, 'ACL_5', alpha, beta),
                BConvLayer(timestep, 2048, 3, 'ACL_6', alpha, beta),
            ]

        elif model_arch == 'b_f':
            layers = [
                BConvLayer(timestep, 192, 7, 'ACL_0', alpha, beta),
                BConvLayer(timestep, 256, 5, 'ACL_1', alpha, beta),
                BConvLayer(timestep, 384, 3, 'ACL_2', alpha, beta),
                BConvLayer(timestep, 512, 3, 'ACL_3', alpha, beta),
                BConvLayer(timestep, 1024, 3, 'ACL_4', alpha, beta),
                BConvLayer(timestep, 2048, 3, 'ACL_5', alpha, beta),
                BConvLayer(timestep, 4096, 1, 'ACL_6', alpha, beta),
            ]

        elif model_arch == 'b_d':
            layers = [
                BConvLayer(timestep, 96, 7, 'ACL_0', alpha, beta),
                BConvLayer(timestep, 96, 7, 'ACL_1', alpha, beta),
                BConvLayer(timestep, 128, 5, 'ACL_2', alpha, beta),
                BConvLayer(timestep, 128, 5, 'ACL_3', alpha, beta),
                BConvLayer(timestep, 192, 3, 'ACL_4', alpha, beta),
                BConvLayer(timestep, 192, 3, 'ACL_5', alpha, beta),
                BConvLayer(timestep, 256, 3, 'ACL_6', alpha, beta),
                BConvLayer(timestep, 256, 3, 'ACL_7', alpha, beta),
                BConvLayer(timestep, 512, 3, 'ACL_8', alpha, beta),
                BConvLayer(timestep, 512, 3, 'ACL_9', alpha, beta),
                BConvLayer(timestep, 1024, 3, 'ACL_10', alpha, beta),
                BConvLayer(timestep, 1024, 3, 'ACL_11', alpha, beta),
                BConvLayer(timestep, 2048, 1, 'ACL_12', alpha, beta),
                BConvLayer(timestep, 2048, 1, 'ACL_13', alpha, beta),
            ]

        else:
            print('Model architecture does not exist')
            sys.exit()

        # output layer
        readout_dense = tf.keras.layers.Dense(
                classes, kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                name='ReadoutDense_Time{}'.format(timestep))

        # initialise list for outputs
        n_layers = len(layers)

        # build the model
        for n, layer in enumerate(layers):

            # Determine current b_input index
            current_idx = (timestep*7)+n

            # get the bottom-up input
            if n == 0:

                # B conv on the image does not need to be recomputed
                if timestep == 0:
                    b_inputs[current_idx] = input_tensor
                else:
                    b_inputs[current_idx] =  None

            else:

                if model_arch != 'b_d':

                    # pool b_input for all layers apart from input
                    b_inputs[current_idx] = tf.keras.layers.MaxPool2D(
                        pool_size=(2, 2),
                        name='MaxPool_Layer_{}_{}'.format(n, timestep)
                        )(activations[timestep][n-1])

                else:

                    if n % 2 == 0:

                        # pool b_input for all layers apart from input
                        b_inputs[current_idx] = tf.keras.layers.MaxPool2D(
                            pool_size=(2, 2),
                            name='MaxPool_Layer_{}_Time_{}'.format(n, timestep)
                            )(activations[timestep][n-1])


            # get the lateral input and suppression state
            if timestep == 0:
                l_input = None
                s_input = None
                x_tn = layer(timestep=timestep, n=n, b_input=b_inputs, l_input=l_input, s_input=s_input)
            else:
                l_input = activations[timestep-1][n]
                s_input = suppressions[timestep-1][n]
                x_tn, s_current = layer(timestep=timestep, n=n, b_input=b_inputs, l_input=l_input, s_input=s_input)

            # batch normalization
            x_tn = tf.keras.layers.BatchNormalization(
                norm_axis,
                name='BatchNorm_Layer_{}_Time_{}'.format(n, timestep))(x_tn)

            # ReLU
            activations[timestep][n] = tf.keras.layers.Activation(
                'relu', name='ReLU_Layer_{}_Time_{}'.format(n, timestep))(x_tn)

            if timestep == 0:
                suppressions[timestep][n] = tf.keras.layers.Activation(
                    'relu', name=None)(x_tn*0)
            else:
                suppressions[timestep][n] = s_current

        # add the readout layers
        x = tf.keras.layers.GlobalAvgPool2D(
            name='GlobalAvgPool_Time_{}'.format(timestep)
            )(activations[timestep][n])
        presoftmax = readout_dense(x)

        # select cumulative or instant readout
        if cumulative_readout and timestep > 0:
            x = tf.keras.layers.Add(
                name='CumulativeReadout_Time_{}'.format(timestep)
                )(presoftmax[:timestep+1])

        else:
            x = presoftmax
        outputs[timestep] = tf.keras.layers.Activation(
            'softmax', name='Sotfmax_Time_{}'.format(timestep))(x)


        # create Keras model and return
        if timestep == 0:
            model1 = tf.keras.Model(
                inputs=input_tensor,
                outputs=outputs[timestep],
                name='b_net_adapt_'+str(timestep))
            return model1
        else:
            model2 = tf.keras.Model(
                inputs=input_tensor,
                outputs=outputs[timestep],
                name='b_net_adapt_'+str(timestep))
            return model2

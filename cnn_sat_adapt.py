# required packages
import urllib
import os
import sys
import numpy as np
import tensorflow as tf
import time
from keras.models import Model

# required scripts
from temporal_models.spoerer_2020.models.preprocess import preprocess_image
from temporal_models.utils import *
from temporal_models.spoerer_2020.models.b_net_adapt import b_net_adapt

"""
Author: A. Brands

Description: This script simulates a feedforward CNN  implemented with intrinsic
suppression over a number of timesteps and plots the activations for a
user-defined layer of the network.

"""

def main():

    # GPU acces
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    # start time
    startTime = time.time()

    # ------------- values than can be adjusted -------------

    # define model and dataset
    model_arch = 'b'                                                          # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # determine layer from which to extract activations for visualization
    layer = '1'

    # set timeseries
    n_timesteps = 100                                                        # number of timesteps
    stim_duration = 30                                                          # stimulus duration
    start = [10, 50]                                                              # starting points of stimuli

    # adaptation parameters
    alpha = 0.96
    beta = 0.7

    # --------------------------------------------------------

    # # establishing a single random seed for reproducibility of results
    # seed = 7
    # np.random.seed(seed)

    # stimulus timecourse
    sample_rate = 512
    dt = 1/sample_rate
    t = tf.range(n_timesteps, dtype=float)*dt

    # define number of output classes and extract image categories
    if dataset == 'ecoset':
        classes = 565
        categories = np.loadtxt('temporal_models/spoerer_2020/pretrained_output_categories/' + dataset + '_categories.txt', dtype=str)
    elif dataset == 'imagenet':
        classes = 1000
        categories = np.loadtxt('temporal_models/spoerer_2020/pretrained_output_categories/' + dataset + '_categories.txt', dtype=str)
    else:
        print('Categories cannot be loaded. Dataset does not exist!')
        sys.exit()

    # input_shape (HARD-coded)
    input_shape = [n_timesteps, 128, 128, 3]
    input_layer = tf.keras.layers.Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

    # initiate model architecture
    model = b_net_adapt(input_layer, classes, model_arch, alpha=alpha, beta=beta, n_timesteps=n_timesteps, cumulative_readout=True)
    # TODO: add trained weights!
    # print(model.summary())

    # print layer names
    for clayer in model.layers:
        print(clayer.name)

    # # print model summary
    # print(model.summary())

    # load input (over time)
    img1_idx = 1
    img2_idx = 1
    input_img, raw_img = load_input_images(input_layer, input_shape, n_timesteps, [img1_idx, img2_idx])

    # load input over time
    input_tensor = load_input_timepts(input_img, input_shape, n_timesteps, stim_duration, start)
    # input_tensor.expand_dims(input_tensor, 0)
    # input_tensor = tf.convert_to_tensor(input_tensor)
    input_tensor = tf.expand_dims(input_tensor, 0)
    # print(input_tensor.shape)

    # run model and extract linear readout
    readout = np.zeros((n_timesteps, classes))
    readout_max = np.zeros((n_timesteps))
    activations = np.zeros(n_timesteps)
    suppressions = np.zeros(n_timesteps)
    s = np.zeros(n_timesteps)

    # get network info
    max_categories = []
    for t in range(n_timesteps):

        # extract readout (i.e. softmax)
        get_layer_activation_readout = tf.keras.backend.function(
            [model.input],
            [model.get_layer('Sotfmax_Time_{}'.format(t)).output])
        readout[t, :] = get_layer_activation_readout(input_tensor)[0][0]
        readout_max[t] = max(readout[t, :])

        # extract index with highest value
        cat_idx_max = np.argmax(readout[t, :])
        cat_max = categories[cat_idx_max]
        max_categories.append(cat_max)

        # print classification
        print('Timestep: ', t)
        print('Model prediction: ', cat_max, '(softmax output: ', readout_max[t], ')')
        print('\n')

        # extract activation for specific layer
        get_layer_activation = tf.keras.backend.function(
            [model.input],
            [model.get_layer('ReLU_Layer_{}_Time_{}'.format(layer, t)).output])
        activations_temp = get_layer_activation(input_tensor)
        activations[t] = np.nanmean(activations_temp)

        # extract suppression
        if t > 0:
            get_layer_suppression = tf.keras.backend.function(
                [model.input],
                [model.get_layer('{}_S_Time_{}'.format(layer, t)).output])
            suppressions_temp = get_layer_suppression(input_tensor)
            # print(np.mean(suppressions_temp))
            suppressions[t] = np.nanmean(suppressions_temp)

            s[t] = alpha * s[t-1] + (1 - alpha) * activations[t-1]

    # determine time it took to run script (check GPU-access)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    # plot activations of conv5 through time
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # plot activations
    ax.axvspan(start[0], start[0]+stim_duration, color='grey', alpha=0.2, label='stimulus')
    ax.axvspan(start[1], start[1]+stim_duration, color='grey', alpha=0.2)
    ax.plot(suppressions/np.amax(suppressions), 'grey', label='suppression_layer')
    ax.plot(s/np.amax(s), 'grey', linestyle='dashed', label='suppression')
    ax.plot(activations/np.amax(activations), 'k', label='activation')
    # ax.plot(suppressions, 'grey', label='suppression')
    # ax.plot(activations, 'k', label='activation')
    ax.set_title('Feedforward network with adaptation (layer: ' + layer + ', ' + model_arch + ')')
    ax.set_xlabel('Timesteps')
    # ax.set_ylabel('Normalized activations (a.u)')
    ax.set_ylabel('Activations (a.u)')

    # show plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/amber/ownCloud/Documents/code/DNN_adaptation_git/visualizations/repetition_adapt_' + model_arch + '_' + dataset)
    plt.show()

if __name__ == '__main__':
    main()

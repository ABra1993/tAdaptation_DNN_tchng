# required packages
import urllib
import os
import sys
import numpy as np
import tensorflow as tf
import time

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

    # start time
    startTime = time.time()

    # ------------- values than can be adjusted -------------

    # define model and dataset
    model_arch = 'b_f'                                                          # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # determine layer from which to extract activations for visualization
    layer = '6'

    # set timeseries
    n_timesteps = 8                                                             # number of timesteps
    stim_duration = 2                                                           # stimulus duration
    start = [1, 4]                                                              # starting points of stimuli

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
    input_shape = [128, 128, 3]
    input_layer = tf.keras.layers.Input((input_shape[0], input_shape[1], input_shape[2]))

    # initiate model architecture
    model = b_net_adapt(input_layer, classes, model_arch, alpha=alpha, beta=beta, n_timesteps=n_timesteps, cumulative_readout=True)
    model.load_weights('bl_ecoset.h5', by_name = True, skip_mismatch = True)

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
    input_tensor1 = load_input_timepts(input_img, input_shape, n_timesteps, stim_duration, start)

    # retrieve activations
    activations = np.zeros(n_timesteps)
    for i in range(n_timesteps):

        get_layer_activation = tf.keras.backend.function(
            [model.input],
            [model.get_layer('ReLU_Layer_{}_Time_{}'.format(layer, i)).output])
        temp = get_layer_activation(input_tensor1[i, :, :, :])
        activations[i] = np.nanmean(temp)

    # compute suppression
    s = np.zeros(n_timesteps)
    for i in range(1, n_timesteps):
        s[i] = alpha * s[i-1] + (1 - alpha) * activations[i-1]

    # determine time it took to run script (check GPU-access)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    # plot activations of conv5 through time
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # plot activations
    ax.axvspan(t[start[0]], t[start[0]]+stim_duration*dt, color='grey', alpha=0.2, label='stimulus')
    ax.axvspan(t[start[1]], t[start[1]]+stim_duration*dt, color='grey', alpha=0.2)
    ax.plot(t, s/np.amax(s), 'grey', label='suppression')
    ax.plot(t, activations/np.amax(activations), 'k', label='activation')
    ax.set_title('Feedforward network with adaptation (layer: ' + layer + ', ' + model_arch + ')')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized activations (a.u)')

    # show plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/repetition_adapt_' + model_arch + '_' + dataset)
    plt.show()

if __name__ == '__main__':
    main()

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
    model_arch = 'b_f'                                                            # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # determine layer from which to extract activations for visualization
    layer = '2'

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
    sample_rate = 1000
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
    # TODO: add trained weights!

    # print layer names
    for clayer in model.layers:
        print(clayer.name)

    ## print model summary
    #print(model.summary())

    # load input (over time)
    img1_idx = 1
    img2_idx = 'testturtle'
    input_img, raw_img = load_input_images(input_layer, input_shape, n_timesteps, [img2_idx, img2_idx])

    # load input over time
    input_tensor1 = load_input_timepts(input_img, input_shape, n_timesteps, stim_duration, start)

    # create activation and supression dict
    act_dict = {}
    sup_dict = {}

    # fill dicts with zeros
    for n in range(0, int(layer)+1):
        act_dict[n] = np.zeros(n_timesteps)
        sup_dict[n] = np.zeros(n_timesteps)

        for i in range(n_timesteps):

            # retrieve activations
            get_layer_activation = tf.keras.backend.function(
                [model.input],
                [model.get_layer('ReLU_Layer_{}_Time_{}'.format(str(n), i)).output])
            temp = get_layer_activation(input_tensor1[i, :, :, :])
            act_dict[n][i] = np.nanmean(temp)

            # compute suppression
            sup_dict[n][i] = alpha * sup_dict[n][i-1] + (1 - alpha) * act_dict[n][i-1]

    print (act_dict)
    print (sup_dict)

    # determine time it took to run script (check GPU-access)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    # list of colours for activations per layer
    clls = ["r", "b", "m", "y", "g", "c", "k", "w"]

    # make string of all layers for title
    titlestr = ''
    for key in act_dict:
        titlestr = titlestr + str(key) + ', '
    titlestr = titlestr[:-2]


    # plot activations of conv5 through time
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # plot activations
    ax.axvspan(t[start[0]], t[start[0]]+stim_duration*dt, color='grey', alpha=0.2, label='stimulus')
    ax.axvspan(t[start[1]], t[start[1]]+stim_duration*dt, color='grey', alpha=0.2)
#    ax.plot(t, s/np.amax(s), 'grey', label='suppression')
    for lay in act_dict:
        ax.plot(t, act_dict[lay]/np.amax(act_dict[lay]), 'k', label='act layer '+str(lay), color=clls[lay])
        ax.plot(t, sup_dict[lay]/np.amax(sup_dict[lay]), label='sup layer '+str(lay), color=clls[lay], linestyle = '--', alpha=0.5)
    ax.set_title('Feedforward network with adaptation (layer: ' + titlestr + '. Model ' + model_arch + ')')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized activations (a.u)')

    # show plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/repetition_adapt_' + model_arch + '_' + dataset)
    plt.show()

if __name__ == '__main__':
    main()

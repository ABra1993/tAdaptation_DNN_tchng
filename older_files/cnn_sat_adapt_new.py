# required packages
import urllib
import os
import sys
import numpy as np
import tensorflow as tf
import time
import h5py
from difflib import SequenceMatcher


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

    # Compute range of layers or only one layer
    range_or_one = 'range'                                                      # options: 'range' or 'one'

    # define model and dataset
    model_arch = 'b'                                                            # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # determine layer from which to extract activations for visualization (range(0, int(layer)+1))
    layer = '6'

    # Train or test set
    train_or_test = 'test'                                                      # options: 'train' or 'test'

    # list of classes of images used (options are all subfiles in the test or train directories)
    imgclass = ['0132_tiger', '0001_man', '0005_house', '0065_tree', '0145_egg']

    # set timeseries
    n_timesteps = 8                                                             # number of timesteps
    stim_duration = 2                                                           # stimulus duration
    start = [0, 4]                                                              # starting points of stimuli

    # adaptation parameters
    alpha = 0.96
    beta = 0.7

    # Boolean indicating whether to show the category prediction of the model
    predict = False

    # --------------------------------------------------------

    # # establishing a single random seed for reproducibility of results
    # seed = 7
    # np.random.seed(seed)

    # determine amount of layers
    if range_or_one == 'range':
        layernum = int(layer)
    else: layernum = 1

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
    model = b_net_adapt(input_tensor =input_layer, classes = classes, model_arch = model_arch, alpha=alpha, beta=beta, n_timesteps=n_timesteps, cumulative_readout=True)

    weight_dict = read_hdf5("weights/b_ecoset.h5")

    manually_load_weights(weight_dict, model)

    imgls = []
    for cat in imgclass:
        if train_or_test == 'train':
            classls = os.listdir('visualizations/stimuli/ecoset_subset_train_25/' + cat)
            classls = ['ecoset_subset_train_25/' + cat + '/' + img for img in classls][0:7]
        else:
            classls = os.listdir('visualizations/stimuli/ecoset_subset_test_25/' + cat)
            classls = ['ecoset_subset_test_25/' + cat + '/' + img for img in classls][0:7]
        for file in classls:
            imgls.append(file)

    act_array, sup_array = compute_act_sup(model, imgls, layer, range_or_one, n_timesteps, input_layer, input_shape, stim_duration, start, alpha, beta, predict, classes, categories)

    # Make array with average activations and supressions
    avg_act_array = np.mean(act_array, axis=2)
    avg_sup_array = np.mean(sup_array, axis=2)
    print('act array average is ', avg_act_array)

    # Create and print array of RS (activation at beginning of stim1 - act at beginning of stim2)
    RS = np.zeros(layernum)
    for lay in range(layernum):
        RS[lay] = avg_act_array[lay][start[0]] - avg_act_array[lay][start[1]]

    print('Repetition Suppression per layer: \n',RS)

    # determine time it took to run script (check GPU-access)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    # list of colours for activations per layer
    clls = ["r", "b", "m", "y", "g", "c", "k", "w"]

    # make string of all layers for title
    titlestr = ''
    if range_or_one == 'range':
        for lay in range(1, int(layer)+1):
            titlestr = titlestr + str(lay) + ', '
        titlestr = titlestr[:-2]
    else: titlestr = layer

    # plot activations of conv5 through time
    fig = plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # plot activations and suppression
    ax.axvspan(t[start[0]], t[start[0]]+stim_duration*dt, color='grey', alpha=0.2, label='stimulus')
    ax.axvspan(t[start[1]], t[start[1]]+stim_duration*dt, color='grey', alpha=0.2)

    if layernum != 1:
        for lay in range(layernum):
            ax.plot(t, avg_act_array[lay], label='act layer '+str(lay+1), color=clls[lay])
            ax.plot(t, avg_sup_array[lay], label='sup layer '+str(lay+1), color=clls[lay], linestyle = '--', alpha=0.5)

    else:
        ax.plot(t, avg_act_array[0], label='act layer '+str(layer), color=clls[0])
        ax.plot(t, avg_sup_array[0], label='sup layer '+str(layer), color=clls[0], linestyle = '--', alpha=0.5)


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

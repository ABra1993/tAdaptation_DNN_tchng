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
from temporal_models.spoerer_2020.models.b_net_adapt_new import b_net_adapt, activations, suppressions, presoftmax, b_inputs, trackb

# To compare new b_net_adapt functions to original ones
from temporal_models.spoerer_2020.models.b_net_adapt import b_net_adapt as bold

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
    range_or_one = 'one'                                                      # options: 'range' or 'one'

    # define model and dataset
    model_arch = 'b'                                                            # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # determine layer from which to extract activations for visualization (range(0, int(layer)+1))
    layer = '5'

    # Train or test set
    train_or_test = 'test'                                                      # options: 'train' or 'test'

    # list of classes of images used (options are all subfiles in the test or train directories)
    imgclass = ['0132_tiger']

    # set timeseries
    n_timesteps = 8                                                             # number of timesteps
    stim_duration = 2                                                           # stimulus duration
    start = [1, 4]                                                              # starting points of stimuli

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
    input_layer2 = tf.keras.layers.Input((input_shape[0], input_shape[1], input_shape[2]))

    img1_idx = 'testturtle.png'
    img2_idx = 'testturtle.png'
    input_img, raw_img = load_input_images(input_layer, input_shape, n_timesteps, [img1_idx, img2_idx])

    #load input over time
    #input_tensor = load_input_timepts(input_img, input_shape, n_timesteps, stim_duration, start)

    # Create model for first timestep
    model = b_net_adapt(input_tensor =input_layer, classes = classes, model_arch = model_arch, alpha=alpha, beta=beta, timestep=0, cumulative_readout=False)
    print(model.summary())

    for layer in model.layers:
        print(layer.name)
    del model

    tf.keras.backend.clear_session()

    # Create model for second timestep
    model2 = b_net_adapt(input_tensor =input_layer, classes = classes, model_arch = model_arch, alpha=alpha, beta=beta, timestep=1, cumulative_readout=False)
    print(model2.summary())

if __name__ == '__main__':
    main()

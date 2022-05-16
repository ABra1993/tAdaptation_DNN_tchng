# required packages
import urllib
import os
import sys
import numpy as np
import tensorflow as tf
import time
import json
import math
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

    # determine layer from which to extract activations for visualization
    layer = '7'
    layernum = 1

    # Compute range of layers or only one layer
    range_or_one = 'one'                                                        # options: 'range' or 'one'

    # Train or test set
    train_or_test = 'test'                                                      # options: 'train' or 'test'

    # list of classes of images used (choose from subfiles in test or train dir)
    imgclass = [f for f in os.listdir('visualizations/stimuli/ecoset_subset_test_25')]

    # Number of images to take from each class
    imgnum = 25

    # define model and dataset
    model_arch = 'b'                                                            # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # set timeseries
    n_timesteps = 8                                                            # number of timesteps
    stim_duration = 2                                                           # stimulus duration
    start = [1, 4]                                                              # starting points of stimuli

    # adaptation parameters
    alpha = 0.96
    beta = 0.7

    # --------------------------------------------------------

    # # establishing a single random seed for reproducibility of results
    # seed = 7
    # np.random.seed(seed)

    # define number of output classes and extract image categories
    classes = 565
    categories = np.loadtxt('temporal_models/spoerer_2020/pretrained_output_categories/' + dataset + '_categories.txt', dtype=str)

    # create list of path to images
    imgls = []
    for cat in imgclass:
        if train_or_test == 'train':
            classls = os.listdir('visualizations/stimuli/ecoset_subset_train_25/' + cat)
            classls = ['ecoset_subset_train_25/' + cat + '/' + img for img in classls][0:imgnum]
        else:
            classls = os.listdir('visualizations/stimuli/ecoset_subset_test_25/' + cat)
            classls = ['ecoset_subset_test_25/' + cat + '/' + img for img in classls][0:imgnum]
        for file in classls:
            imgls.append(file)

    # input_shape (HARD-coded)
    input_shape = [n_timesteps, 128, 128, 3]
    input_layer = tf.keras.layers.Input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))

    # initiate model architecture
    model = b_net_adapt(input_layer, classes, model_arch, alpha=alpha, beta=beta, n_timesteps=n_timesteps, cumulative_readout=True)

    # add trained weights
    weight_dict = read_hdf5("weights/b_ecoset.h5")
    manually_load_weights(weight_dict, model)

    # definitions of arrays to store activation, suppression and classification
    readout = np.zeros((len(imgls), layernum, n_timesteps, classes))
    readout_max = np.zeros((len(imgls), layernum, n_timesteps))
    activations_perunit = np.zeros((len(imgclass), imgnum, layernum, n_timesteps, 2048*4))
    suppressions_perunit = np.zeros((len(imgclass), imgnum, layernum, n_timesteps, 2048*4))

    # retrieve activations
    for idx, img in enumerate(imgls):

        # load input
        input_img, raw_img = load_input_images(input_layer, input_shape, n_timesteps, [img, img])

        # load input over time
        input_tensor = load_input_timepts(input_img, input_shape, n_timesteps, stim_duration, start)
        input_tensor = tf.expand_dims(input_tensor, 0)

        # get network info
        max_categories = []

        # run model and get linear readout
        for t in range(n_timesteps):
            for l in range(layernum):
                if range_or_one == 'range':
                    num = l
                else: num = int(layer)-1

                # extract average activation and activation per unit
                get_layer_activation = tf.keras.backend.function(
                    [model.input],
                    [model.get_layer('ReLU_Layer_{}_Time_{}'.format(num, t)).output])
                activations_temp = get_layer_activation(input_tensor)

                for idx1, a in enumerate(activations_temp[0][0]):
                    for idx2, b in enumerate(a):
                        for idx3, c in enumerate(b):
                            activations_perunit[math.floor(idx/imgnum), idx%imgnum, l, t, idx1*2*2048 + idx2*2048 + idx3] = c

                # extract average suppression and suppression per unit
                if t > 0:
                    get_layer_suppression = tf.keras.backend.function(
                        [model.input],
                        [model.get_layer('{}_S_Time_{}'.format(num, t)).output])
                    suppressions_temp = get_layer_suppression(input_tensor)

                    for idx1, a in enumerate(suppressions_temp[0][0]):
                        for idx2, b in enumerate(a):
                            for idx3, c in enumerate(b):
                                suppressions_perunit[math.floor(idx/imgnum), idx%imgnum, l, t, idx1*2*2048 + idx2*2048 + idx3] = c

    # save activations and suppressions per unit
    with open("activationsl7.json", "w") as f:
        json.dump(activations_perunit.tolist(), f)
    with open("suppressionsl7.json", "w") as f:
        json.dump(suppressions_perunit.tolist(), f)

if __name__ == '__main__':
    main()

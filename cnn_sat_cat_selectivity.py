# required packages
import urllib
import os
import sys
import numpy as np
import tensorflow as tf
import time

# required scripts
from temporal_models.spoerer_2020.models.preprocess import preprocess_image
from temporal_models.spoerer_2020.models.b_net import b_net, b_k_net, b_k_net, b_d_net
from temporal_models.spoerer_2020.models.bl_net import bl_net
from temporal_models.utils import *

"""
Author: A. Brands


Description: This script simulates a feedforward CNN over a number of timesteps
and plots the activations for a user-defined layer of the network.

"""

def main():

    # start time
    startTime = time.time()

    # ------------- values than can be adjusted -------------

    # define model and dataset
    model_arch = 'b'                                                          # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on
    imgclass = [f for f in os.listdir('visualizations/stimuli/ecoset_subset_test_25')]

    # determine layer from which to extract activations for visualization
    layer = '5'

    # set timeseries
    n_timesteps = 8                                                             # number of timesteps
    stim_duration = 2                                                           # stimulus duration
    start = [1, 4]                                                              # starting points of stimuli

    # --------------------------------------------------------

    # # establishing a single random seed for reproducibility of results
    # seed = 7
    # np.random.seed(seed)

    # stimulus timecourse
    sample_rate = 512
    dt = 1/sample_rate
    t = tf.range(n_timesteps, dtype=float)*dt

    # Create list with directories to images
    imgls = [[] for _ in range(len(imgclass))]
    for idx,cat in enumerate(imgclass):
        classls = os.listdir('visualizations/stimuli/ecoset_subset_test_25/' + cat)
        classls = ['ecoset_subset_test_25/' + cat + '/' + img for img in classls]
        imgls[idx] = classls[0:10]

    # define number of output classes and extract image categories
    classes = 565
    categories = np.loadtxt('temporal_models/spoerer_2020/pretrained_output_categories/' + dataset + '_categories.txt', dtype=str)

    # input_shape (HARD-coded)
    input_shape = [128, 128, 3]

    # building the model and load weights
    input_layer = tf.keras.layers.Input((input_shape[0], input_shape[1], input_shape[2]))
    model = load_pretrained_model(model_arch, dataset, input_layer, classes)

    model.load_weights('weights/' + str(model_arch) + '_ecoset.h5')

    # Create array for activations
    activations = np.zeros((len(imgls), 1, 1, 4, 4, 1024))

    # Loop through images, and create array with activations for each class.
    # The mean of these arrays per classes are added to 'activations'
    for classidx, cl in enumerate(imgls):

        classact = np.zeros((10, 1, 1, 4, 4, 1024))

        for idx, img in enumerate(cl):

            # load input (over time)
            input_img, raw_img = load_input_images(input_layer, input_shape, n_timesteps, [img, img])

            # load input over time
            input_tensor = load_input_timepts(input_img, input_shape, n_timesteps, stim_duration, start)

            # run model and extract linear readout
            get_layer_activation = tf.keras.backend.function(
                [model.input],
                [model.get_layer('ReLU_Layer_{}'.format(layer)).output])

            classact[idx] = get_layer_activation(input_tensor[2, :, :, :, :])

        activations[classidx] = np.mean(classact, axis=0)

    # Create selectivity array
    selectivity = np.zeros((1024*16, len(imgclass)))
    for idx1, cla in enumerate(activations):
        for idx2, breadth in enumerate(cla[0][0]):
            for idx3, height in enumerate(breadth):
                for idx4, maps in enumerate(height):
                    selectivity[idx2*idx3*idx4][idx1] = maps

    # Sort array
    selectivity_sort = np.sort(selectivity, axis=-1)
    for idx, unit in enumerate(selectivity_sort):
        selectivity_sort[idx] = np.flipud(unit)

    # Swap axes for plotting
    selectivity_sort_swapped = np.swapaxes(selectivity_sort, 0, 1)

    # Calculating selectivity index (inspired by Rafegas et al., 2019)
    class_freq = np.zeros((1024*16, len(imgclass)))

    for idx, unit in enumerate(selectivity_sort):
        for idx2, cat in enumerate(unit):
            class_freq[idx][idx2] = cat/sum(unit)

    select_idx = np.zeros(1024*16)

    # Determine class selectivity, setting th to 0.3
    th = 0.3

    for idx, unit in enumerate(class_freq):
        M = 0
        for idx2, cat in enumerate(unit):
            if cat > th:
                M += 1
            elif idx2 == 0: M = -1
        if M != -1:
            select_idx[idx] = (565-M)/564
        else: select_idx[idx] = 0

    print(select_idx[0:10], '\n', class_freq[0:10])

    plt.plot(selectivity_sort_swapped)
    plt.title('Ranked activations of units in layer 5 of b model')
    plt.xlabel('classes (ranked)')
    plt.ylabel('activation')
    plt.show()

if __name__ == '__main__':
    main()

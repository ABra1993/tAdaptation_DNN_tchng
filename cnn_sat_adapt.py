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

    # determine layer from which to extract activations for visualization
    layer = '7'

    # Compute range of layers or only one layer
    range_or_one = 'range'                                                      # options: 'range' or 'one'

    # Train or test set
    train_or_test = 'test'                                                      # options: 'train' or 'test'

    # list of classes of images used (choose from subfiles in test or train dir)
    imgclass = [f for f in os.listdir('visualizations/stimuli/ecoset_subset_test_25')][0:3]

    # Amount of images to take per class
    imgnum = 2

    # define model and dataset
    model_arch = 'b'                                                            # network architecture (options: b, b_k, b_f, b_d)
    dataset = 'ecoset'                                                          # dataset the network is trained on

    # set timeseries
    n_timesteps = 10                                                            # number of timesteps
    stim_duration = 2                                                           # stimulus duration
    start = [2, 6]                                                              # starting points of stimuli

    # adaptation parameters
    alpha = 0.96
    beta = 0.7

    # --------------------------------------------------------

    # # establishing a single random seed for reproducibility of results
    # seed = 7
    # np.random.seed(seed)

    # determine amount of layers
    if range_or_one == 'range':
        layernum = int(layer)
    else: layernum = 1

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

    # print model summary
    #print(model.summary())

    # definitions of arrays to store activation, suppression and classification
    readout = np.zeros((len(imgls), layernum, n_timesteps, classes))
    readout_max = np.zeros((len(imgls), layernum, n_timesteps))
    activations = np.zeros((len(imgls), layernum, n_timesteps))
    suppressions = np.zeros((len(imgls), layernum, n_timesteps))
    s = np.zeros((len(imgls), layernum, n_timesteps))

    for idx, img in enumerate(imgls):

        # load input (over time)
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

                # extract readout (i.e. softmax)
                get_layer_activation_readout = tf.keras.backend.function(
                    [model.input],
                    [model.get_layer('Sotfmax_Time_{}'.format(t)).output])
                readout[idx, l, t, :] = get_layer_activation_readout(input_tensor)[0][0]
                readout_max[idx, l, t] = max(readout[idx, l, t, :])

                # extract index with highest value
                cat_idx_max = np.argmax(readout[idx, l, t, :])
                cat_max = categories[cat_idx_max]
                max_categories.append(cat_max)

                # print classification
                print('Timestep: ', t)
                print('Model prediction: ', cat_max, '(softmax output: ', readout_max[idx, l, t], ')')
                print('\n')

                # extract activation for specific layer
                get_layer_activation = tf.keras.backend.function(
                    [model.input],
                    [model.get_layer('ReLU_Layer_{}_Time_{}'.format(num, t)).output])
                activations_temp = get_layer_activation(input_tensor)
                activations[idx, l, t] = np.nanmean(activations_temp)

                # extract suppression
                if t > 0:
                    get_layer_suppression = tf.keras.backend.function(
                        [model.input],
                        [model.get_layer('{}_S_Time_{}'.format(num, t)).output])
                    suppressions_temp = get_layer_suppression(input_tensor)
                    suppressions[idx, l, t] = np.nanmean(suppressions_temp)

                    s[idx, l, t] = alpha * s[idx, l, t-1] + (1 - alpha) * activations[idx, l, t-1]

    # Make array with average activations and supressions
    avg_activations = np.mean(activations, axis=0)
    avg_suppressions = np.mean(suppressions, axis=0)
    avg_s = np.mean(s, axis=0)

    # Create and print array of RS (activation at beginning of stim1 - act at beginning of stim2)
    RS = np.zeros(layernum)
    rel_RS = np.zeros(layernum)

    for lay in range(layernum):
        RS[lay] = avg_activations[lay][start[0]] - avg_activations[lay][start[1]]

    for lay in range(layernum):
        rel_RS[lay] = avg_activations[lay][start[0]]/np.amax(avg_activations[lay]) - avg_activations[lay][start[1]]/np.amax(avg_activations[lay])

    print('Repetition Suppression per layer: \n',RS)
    print('relative Repetition Suppression per layer: \n',rel_RS)

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

    # plot stimulus durations
    ax.axvspan(start[0], start[0]+stim_duration, color='grey', alpha=0.2, label='stimulus')
    ax.axvspan(start[1], start[1]+stim_duration, color='grey', alpha=0.2)

    tr = list(range(0, n_timesteps))
    if layernum != 1:
        for l in range(layernum):
            ax.plot(tr, avg_activations[l], label='act layer '+str(l+1), color=clls[l])
            ax.plot(avg_s[l], 'grey', linestyle='dashed')
            ax.plot(tr, avg_suppressions[l], label='sup layer '+str(l+1), color=clls[l], linestyle = '--', alpha=0.5)

    else:
        ax.plot(tr, avg_activations[0, :], label='act layer '+str(layer), color=clls[0])
        ax.plot(avg_s, 'grey', linestyle='dashed')
        ax.plot(tr, avg_suppressions[0, :], label='sup layer '+str(layer), color=clls[0], linestyle = '--', alpha=0.5)

    ax.set_title('Feedforward network with adaptation (layer: ' + titlestr + '. Model ' + model_arch + ')')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activations (a.u)')

    # show plot
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/repetition_adapt_' + model_arch + '_' + dataset)
    plt.show()

if __name__ == '__main__':
    main()

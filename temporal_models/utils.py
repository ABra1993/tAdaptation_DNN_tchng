# required pacakges
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys

# required scripts
from temporal_models.spoerer_2020.models.preprocess import preprocess_image
from temporal_models.spoerer_2020.models.b_net import *
from temporal_models.spoerer_2020.models.bl_net import bl_net

def load_input_images(input_layer, input_shape, timepts, img_idx):
    """ Loads RGB vlaues of the input image.

    params
    -----------------------
    input_layer : keras layer
        input layer to the network
    input_shape : list
        contains shape of each dimension for the input images
    timepts : int
        number of timepoints
    img_idx : int
        index of image to obtain filename (img_idx = 1, filename: '1.png')

    returns
    -----------------------
    input_img : tensor
        containing preprocessed RGB values for each input image
    raw_img : tensor
        containing processed RGB values for each input image

    """

    # retrieve data format
    data_format = tf.keras.backend.image_data_format()                                                      # channels last (i.e. 3)

    # input over time
    if data_format == 'channels_last':
        input_img = np.zeros((len(img_idx), input_shape[0], input_shape[1], input_shape[2]))
    else:
        input_img = np.zeros((len(img_idx), input_shape[1], input_shape[2], input_shape[3]))

    # import images
    raw_img = []
    for i in range(len(img_idx)):
    # for i in range(1):

        # import image
        raw_img_temp = cv2.imread('visualizations/stimuli/stimulus/' + str(img_idx[i]) + '.png')
        raw_img_temp = cv2.cvtColor(raw_img_temp, cv2.COLOR_BGR2RGB)

        # resize height and witdh
        if data_format == 'channels_last':
            img_resize = cv2.resize(raw_img_temp, (input_shape[0], input_shape[1]))
        else:
            img_resize = cv2.resize(raw_img_temp, (input_shape[1], input_shape[2]))
        raw_img.append(img_resize)

        # scale pixel values
        img_preprocessed = preprocess_image(img_resize)
        input_img[i, :, :, :] = img_preprocessed


    return input_img, raw_img


def load_input_timepts(input_img, input_shape, timepts, stim_duration, start):
    """ Creates a tensor containing the model input for each timestep.

    params
    -----------------------
    input_img : array
        contains RGB values of each input image
    input_shape : list
        contains shape of each dimension for the input images
    timepts : int
        number of timepoints
    stim_duration : int/float
        duration each stimulus is presented to the network
    start : list
        contains the first timepoint each input image is presented

    returns
    -----------------------
    input : tensor
        containing input values to the model for each timepoint

    """

    # retrieve number of images
    img_n = input_img.shape[0]

    # retrieve data format
    data_format = tf.keras.backend.image_data_format()                          # channels last (i.e. 3)

    # input over time
    if data_format == 'channels_last':
        input = np.zeros((timepts, 1, input_shape[0], input_shape[1], input_shape[2]))
    else:
        input = np.zeros((timepts, 1, input_shape[1], input_shape[2], input_shape[0]))

    for i in range(img_n):

        t_start = start[i]
        t_end = start[i] + stim_duration

        input[t_start:t_end, :, :, :, :] = input_img[i] * stim_duration

    # convert to tensor
    input_tensor = tf.convert_to_tensor(input, dtype=tf.uint8)

    return input


def load_pretrained_model(model_arch, dataset, input_layer, classes):
    """

    params
    -----------------------
    model_arch : str
        the type of model (e.g. b, b-k, b-f, b-d)
    dataset : str
        dataset on which the model was trained (e.g. ecoset, imagenet)
    input_layer : keras layer
        input layer to the network
    classes : int
        number of output classes (depends on the dataset)

    returns
    -----------------------
    model : keras model
        model containing the pretrained weights

    """

    print(30*'-')
    print('Initiate model...')
    print(30*'-')

    # initiate model architecture
    if model_arch == 'b':
        model = b_net(input_layer, classes=classes)
    elif model_arch == 'b_k':
        model = b_k_net(input_layer, classes=classes)
    elif model_arch == 'b_f':
        model = b_f_net(input_layer, classes=classes)
    elif model_arch == 'b_d':
        model = b_d_net(input_layer, classes=classes)
    else:
        print('Model does not exist!')
        sys.exit()

    # load weights into the model
    # model.load_weights('temporal_models/spoerer_2020/weights/' + model_arch + '_' + dataset + '.h5')

    return model

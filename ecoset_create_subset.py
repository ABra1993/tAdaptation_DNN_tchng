# required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os

# required scripts

"""
Author: A. Brands

Outputs: -

Description: creates a subset of the ecoset, where each class contains a USER-DEFINED

    number of images (number of images see documentation)

"""

def main():

    # number of images per class
    N = 5
    set = 'test'

    # root (location where to create subset)
    root = '/home/amber/Documents/ecoset/'

    try: # remove old folder containing the subset
        shutil.rmtree(root + 'ecoset_subset_' + set + '_' + str(N) + '/')
    except:
        print('Folder already removed...\n')

    try: # create folder containing the subset
        os.mkdir(root + 'ecoset_subset_' + set + '_' + str(N))
    except:
        print('Folder already exists...\n')

    # import labels
    ecoset_categories = np.loadtxt(root+'ecoset_categories.txt', dtype=str)
    classes_n = len(ecoset_categories)
    print(30*'-')
    print('Number of classes: ', classes_n)
    print(30*'-')

    # extract subset
    for i in range(classes_n):
    # for i in range(1):

        print(30*'-')
        print('Current class: ', ecoset_categories[i], '(', i+1, '/', classes_n, ')')
        print(30*'-')

        # create class folder
        os.mkdir(root + 'ecoset_subset_' + set + '_' + str(N) + '/' + ecoset_categories[i])

        # copy first n images
        for n in range(N):

            # retrieve filename
            file_name = os.listdir(root + 'ecoset/' + set + '/' + ecoset_categories[i] + '/')[n]

            # copy to subset folder
            source = root + 'ecoset/' + set + '/' + ecoset_categories[i] + '/' + file_name
            target = root + 'ecoset_subset_' + set + '_' + str(N) + '/' + ecoset_categories[i] + '/' + file_name
            shutil.copyfile(source, target)



if __name__ == '__main__':
    main()

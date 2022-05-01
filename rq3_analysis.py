import json
import numpy as np
import os

n_timesteps = 4
classnum = 122
start = [1, 3]

# List of classes
imgclass = [f for f in os.listdir('visualizations/stimuli/ecoset_subset_test_25')]

# Dictionary with selective units and the classes they are selective for
with open('selectivity/rafegas/selectdict.json', 'r') as fp:
    selectdict = json.load(fp)
print(selectdict)

# shape is (len(imgclass), imgnum, layernum, n_timesteps, 2048*4)
activations = np.array(json.load(open("selectivity/rq3/activationsl7.json")))
suppressions = np.array(json.load(open("selectivity/rq3/suppressionsl7.json")))

# compute average per class
activations_perclass = np.mean(activations, axis=1)
suppressions_perclass = np.mean(suppressions, axis=1)
activations_perclass.shape = (8192, classnum, n_timesteps)
suppressions_perclass.shape = (8192, classnum, n_timesteps)

# array of activations for the selective units. axis 1 indicates activations for nonprefered or prefered classes
pref_nonpref_activations = np.zeros((len(selectdict.keys()), 2, n_timesteps))

i = 0
for idx1, unit in enumerate(activations_perclass):

    if str(idx1) in selectdict.keys():

        non_pref = np.zeros((365-len(selectdict[str(idx1)]), n_timesteps))
        pref = np.zeros((len(selectdict[str(idx1)]), n_timesteps))
        a = 0
        b = 0

        # keep track of whether activations are for prefered or nonprefered classes
        for idx2, classact in enumerate(unit):
            if idx2 in selectdict[str(idx1)]:
                pref[a] = classact
                a += 1
            else:
                non_pref[b] = classact
                b += 1

        # add means of activations to main array
        pref_nonpref_activations[i, 0] = np.mean(non_pref, axis=0)
        pref_nonpref_activations[i, 1] = np.mean(pref, axis=0)

        i += 1

# array of average activations for either prefered or nonprefered classes
rep_sup = np.zeros((len(selectdict.keys()), 2))

for idx1, unit in enumerate(pref_nonpref_activations):
    rep_sup[idx1, 0] = unit[0, start[1]] - unit[0, start[0]]
    rep_sup[idx1, 1] = unit[1, start[1]] - unit[1, start[0]]

print(rep_sup, pref_nonpref_activations)

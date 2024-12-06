from scipy.io import loadmat
import os
import numpy as np


def load_mat_data(filepath):
    mat = loadmat(filepath)
    data = mat['data']  # shape: (samples, channels, trials)
    label = mat['label']  # shape: (labels,)
    label = label -1
    data = np.transpose(data, (2, 1, 0))  # (trials, channels, samples)
    return data, label
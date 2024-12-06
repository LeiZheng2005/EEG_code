
from scipy.signal import cheby2, filtfilt
import numpy as np
import config
from keras.api.layers import Dense, Activation, Permute, Dropout
def preprocess_data(X, trial, args):
    wn = [args.low_cutoff * 2 / args.sample_rate, args.high_cutoff * 2 / args.sample_rate]
    b, a = cheby2(6, 60, wn, btype='bandpass')
    trials = []
    for start_idx in trial.ravel():
        trial_start = start_idx + int(args.pre_offset * args.sample_rate)
        trial_end = start_idx + int(args.post_offset * args.sample_rate)
        if trial_end > X.shape[0]:
            continue
        segment = X[trial_start:trial_end, 0:args.eeg_channels]
        segment = filtfilt(b, a, segment, axis=0)
        trials.append(segment)
    trials = np.array(trials)  # Shape: (num_trials, time_points, num_channels)
    trials = np.transpose(trials, (0, 2, 1)) # Shape: (num_trials, num_channels, time_points, )
    return trials

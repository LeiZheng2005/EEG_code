import os
import scipy.io as sio
from scipy.signal import cheby2, filtfilt
import numpy as np
import argparse
import time
def get_args():
    parser = argparse.ArgumentParser(description="EEG Training Configuration")
    # Datasets
    parser.add_argument('--datasets_train_dir', type=str, default="/Users/leizheng/PyCharm_Study_Code/EEG_code/Datasets/Dataset-BCI-2a",choices=['Datasets/BCICIV_2a_gdf', 'Datasets/BCICIV_2b_gdf'], help="Path to the dataset directory (train)")
    # Preprocess
    parser.add_argument("--preprocess_save_dir", type=str, default='/Users/leizheng/PyCharm_Study_Code/EEG_code/Datasets/preprocess_bci2a_data', help="The dir to save preprocess data")
    parser.add_argument("--low_cutoff", type=float, default=4, help="Low cutoff frequency for band-pass filter (Hz).")
    parser.add_argument("--high_cutoff", type=float, default=40, help="High cutoff frequency for band-pass filter (Hz).")
    parser.add_argument("--sample_rate", type=float, default=250, help="Sampling rate of the EEG data (Hz).")
    parser.add_argument("--pre_offset", type=float, default=3.0, help="Start time offset relative to event marker (seconds).")
    parser.add_argument("--post_offset", type=float, default=6.0, help="End time offset relative to event marker (seconds).")
    parser.add_argument("--eeg_channels", type=int, default=22, help="Number of EEG channels.")
    return parser.parse_args()
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

def preprocess_bci2a_data(args):
    for session_type in ['T']:
        for subject_index in range(1,10):
            file_path = os.path.join(args.datasets_train_dir, f's{subject_index}/A0{subject_index}{session_type}.mat')
            data = sio.loadmat(file_path)
            all_data ,all_labels = [], []
            for ii in range(3,data['data'].size):
                struct_item = [data['data'][0, ii][0, 0]][0]
                data_item = preprocess_data(struct_item[0], struct_item[1], args)
                all_data.append(data_item)
                all_labels.append(struct_item[2])
            all_data = np.concatenate(all_data, axis=0)
            all_labels = np.concatenate(all_labels).reshape(-1, 1)
            save_path = os.path.join(args.preprocess_save_dir, f's{subject_index}/A0{subject_index}{session_type}_preprocessed.mat')
            sio.savemat(save_path, {'data': all_data, 'label': all_labels})
            print(f"Preprocessed data saved to {save_path}")

if __name__ == '__main__':
    args=get_args()
    preprocess_bci2a_data(args)
    # bci 2a
    # s4 中的A04T.mat由于在第二个struct中开始出现tials需要单独计算，其他都从第四个trial开始计数。
    # 即42行 for ii in range(3,data['data'].size): 中，3指的是第四个trial开始计数。总计288个trials


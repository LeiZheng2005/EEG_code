import os
import numpy as np
import scipy.io as sio
from config import get_args
from preprocess_functions import preprocess_data

def preprocess_bci2b_data(args):
    for session_type in ['T','E']:
        for subject_index in range(1,10):
            file_path = os.path.join(args.datasets_train_dir, f's{subject_index}/B0{subject_index}{session_type}.mat')
            data = sio.loadmat(file_path)
            all_data ,all_labels = [], []
            for ii in range(0,data['data'].size):
                if ii == 2: # 当选取T中第三个数据之后，变换时间切片，feedback切为3.5-6.5s，保持3s时长。后续处理E，采用已修改数据
                    args.pre_offset = 3.5
                    args.post_offset = 6.5
                struct_item = [data['data'][0, ii][0, 0]][0]
                data_item = preprocess_data(struct_item[0], struct_item[1], args)
                all_data.append(data_item)
                all_labels.append(struct_item[2])
            all_data = np.concatenate(all_data, axis=0)
            all_labels = np.concatenate(all_labels).reshape(-1, 1)
            save_path = os.path.join(args.preprocess_save_dir, f's{subject_index}/B0{subject_index}{session_type}_preprocessed.mat')
            sio.savemat(save_path, {'data': all_data, 'label': all_labels})
            print(f"Preprocessed data saved to {save_path}")

if __name__ == '__main__':
    args=get_args()
    preprocess_bci2b_data(args)


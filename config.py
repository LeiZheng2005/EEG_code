import argparse
import time
def get_args():
    parser = argparse.ArgumentParser(description="EEG Training Configuration")
    # wandb
    parser.add_argument('--wandb_project_key', type=str, default='f176995f11059393f499f093aa1d4e3aa9520cdd', help='The wandb key (leizheng)')
    parser.add_argument('--wandb_project', type=str, default='EEG-Training-Project', help='Name of the wandb project')
    parser.add_argument('--wandb_project_name', type=str, default='Test_'+str(time.time()), help='Name of the wandb project')
    parser.add_argument('--wandb_entity', type=str, default='eeg-project', help='Name of the wandb entity')
    # Datasets
    parser.add_argument('--load_is_train', type=bool, default=False, help="False to load_is_train")
    parser.add_argument('--load_is_test', type=bool, default=False, help="False to load_is_test")
    parser.add_argument('--datasets_train_dir', type=str, default="/Users/leizheng/PyCharm_Study_Code/EEG_code/Datasets/Dataset-BCI-2b",choices=['Datasets/BCICIV_2a_gdf', 'Datasets/BCICIV_2b_gdf'], help="Path to the dataset directory (train)")
    parser.add_argument('--datasets_test_dir', type=str, default="Datasets/Dataset-BCI-2b",choices=['Datasets/BCICIV_2a_gdf', 'Datasets/BCICIV_2b_gdf'], help="Path to the dataset directory (test)")
    # Preprocess
    parser.add_argument("--preprocess_save_dir", type=str, default='/Users/leizheng/PyCharm_Study_Code/EEG_code/Datasets/preprocess_bci2b_data', help="The dir to save preprocess data")
    parser.add_argument("--low_cutoff", type=float, default=4, help="Low cutoff frequency for band-pass filter (Hz).")
    parser.add_argument("--high_cutoff", type=float, default=40, help="High cutoff frequency for band-pass filter (Hz).")
    parser.add_argument("--sample_rate", type=float, default=250, help="Sampling rate of the EEG data (Hz).")
    parser.add_argument("--pre_offset", type=float, default=3.0, help="Start time offset relative to event marker (seconds).")
    parser.add_argument("--post_offset", type=float, default=6.0, help="End time offset relative to event marker (seconds).")
    parser.add_argument("--eeg_channels", type=int, default=3, help="Number of EEG channels.")
    parser.add_argument("--eog_channels", type=int, default=3, help="Number of EOG channels.")
    parser.add_argument("--total_channels", type=int, default=6, help="Number of total channels.")
    parser.add_argument("--trial_marker", type=int, default=768, help="Event marker indicating trial start.")
    parser.add_argument('--valid_events', type=int, nargs='+', default=[7, 8, 9, 10], help="List of valid events.")
    parser.add_argument('--crop_size', type=int, default=128, help="Crop size for training.")
    parser.add_argument('--step_size', type=int, default=64, help="Step size for cropping.")

    parser.add_argument("--subject_index", type=int, default=9, help="Subject index (1-9).")

    # Model
    parser.add_argument('--model_name', type=str, default='EEGNet', choices=['EEGNet', 'ATCNet'], help="Model name to use")
    parser.add_argument('--num_classes', type=int, default=4, help="Number of output classes")
    # Train or Test
    parser.add_argument('--is_train', type=bool, default=False, help="False to train")
    parser.add_argument('--is_test', type=bool, default=False, help="False to test")
    parser.add_argument('--seed', type=int, default=7, help="Random seed")
    parser.add_argument('--num_epochs', type=int, default=499, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split for training")
    parser.add_argument('--sub_start', type=int, default=1, help="Starting subject ID")
    parser.add_argument('--sub_end', type=int, default=9, help="Ending subject ID")

    # Results
    parser.add_argument('--results_dir', type=str, default='Results', help="Directory to save results")
    parser.add_argument('--weights_dir', type=str, default='Results/Weights', help="Directory to save model weights")
    parser.add_argument('--results_CSV_dir', type=str, default='Results/CSV', help="Directory to save training results")

    return parser.parse_args()
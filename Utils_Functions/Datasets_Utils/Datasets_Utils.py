import numpy as np


def exponential_moving_standardization(raw, factor_decay=0.999, init_samples=1000):
    data = raw.get_data()
    n_channels, n_samples = data.shape
    moving_mean = np.mean(data[:, :init_samples], axis=1, keepdims=True)
    moving_var = np.var(data[:, :init_samples], axis=1, keepdims=True)
    std_data = np.zeros_like(data)
    for t in range(n_samples):
        x_t = data[:, t:t + 1]
        std_data[:, t:t + 1] = (x_t - moving_mean) / np.sqrt(moving_var + 1e-8)
        moving_mean = factor_decay * moving_mean + (1 - factor_decay) * x_t
        moving_var = factor_decay * moving_var + (1 - factor_decay) * (x_t - moving_mean) ** 2
    raw_std = raw.copy()
    raw_std._data = std_data
    return raw_std
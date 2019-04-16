import numpy as np

import data_manager as dm
#import matplotlib as mpl
import matplotlib.pyplot as plt
import filters as filt

# Load data
my_dm = dm.DataManager(r'C:\Users\Gunnar\Google Drive\MSThesis\Data')
#my_dm.update_data_structure()
my_dm.load_emg('Subject01','20190405', reload=False)
#my_dm = dm.load_all_emg(my_dm)

# Setup parameters
low_pass = 30               # The lowpass cutoff frequency for EMG
high_pass = 200             # The highpass cutoff frequency for EMG
smoothing_window_ms = 100   # The window for moving average in milliseconds
fs = 1000                   # Sampling frequency for analog data

emg_filt_data_dict = filt.filter_emg(my_dm.emg_data_dict, low_pass, high_pass, smoothing_window_ms, fs)

for key in emg_filt_data_dict:
    t = emg_filt_data_dict[key][:, 0]
    i = 0
    plt.figure()
    for column in emg_filt_data_dict[key][:, 1:].T:
        i = i + 1
        plt.plot(t, column, label='EMG %s' % i)

    plt.legend()
    plt.savefig('./figures/' + key)

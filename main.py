import numpy as np
import matplotlib.pyplot as plt

import data_manager as dm
import filters as filt

# Load data
my_dm = dm.DataManager(r'C:\Users\Gunnar\Google Drive\MSThesis\Data')
# my_dm.update_data_structure() # Uncomment if the data_structure.json needs updating
my_dm.load_emg_and_torque('Subject01', '20190405', reload=True, load_filt=True)

# Setup parameters
emg_lowcut = 30             # The lowpass cutoff frequency for EMG
emg_highcut = 200           # The highpass cutoff frequency for EMG
torque_lowcut = 100         # The lowpass cutoff frequency for the torque
torque_filt_order = 5       # The order of the butterworth lowpass filter
smoothing_window_ms = 20    # The window for moving average in milliseconds
fs = 1000                   # Sampling frequency for analog data
fs_mean_window = 10         # This value is used to "decrease the sampling frequency" of the emg by taking the mean,
#                           # i.e. new sampling frequency = fs/fs_mean_window

filt.filter_emg(my_dm.emg_data_dict, emg_lowcut, emg_highcut, smoothing_window_ms, fs, cut_time=True,
                fs_mean_window=fs_mean_window)
filt.filter_torque(my_dm.torque_data_dict, torque_filt_order, torque_lowcut, fs, axis_of_focus=0, cut_time=True,
                   lp_filter=False)
my_dm.update_filt_data_dict(reload=True)

# TODO: Ooooog þjálfa

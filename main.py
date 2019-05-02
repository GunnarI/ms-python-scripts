import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_manager as dm
import filters as filt
import ann_methods as ann

from tensorflow import keras

from scipy.stats import ttest_ind

# Load data
my_dm = dm.DataManager(r'C:\Users\Gunnar\Google Drive\MSThesis\Data')
my_dm.update_data_structure()   # Uncomment if the data_structure.json needs updating
my_dm.load_emg_and_torque('Subject01', '20190405B', reload=False, load_filt=False)
# my_dm.load_emg_and_torque('Subject02', '20190406W', reload=False, load_filt=False)
my_dm.load_emg_and_torque('Subject02', '20190406B', reload=True, load_filt=False)
my_dm.load_emg_and_torque('Subject03', '20190413B', reload=False, load_filt=False)
my_dm.load_emg_and_torque('Subject05', '20190427', reload=False, load_filt=False)

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

# Prepare data for training
# dataset = my_dm.filt_data_dict["20190427 Subject05 concat filtered"].copy()
dframes = [my_dm.filt_data_dict["20190405B Subject01 concat filtered"].copy(),
           my_dm.filt_data_dict["20190406B Subject02 concat filtered"].copy(),
           my_dm.filt_data_dict["20190413B Subject03 concat filtered"].copy(),
           my_dm.filt_data_dict["20190427 Subject05 concat filtered"].copy()]
for frame in dframes:
    frame.pop('Exercise')
    frame.pop('Time')
    frame.pop('GlutMax')
#dataset = pd.concat()
dataset = my_dm.filt_data_dict["20190406B Subject02 concat filtered"].copy()
dataset.pop('Exercise')
dataset.pop('Time')
# dataset.pop('GlutMax')  # Looks faulty
norm_data = filt.min_max_normalize_emg(dataset)
train_dataset = norm_data.sample(frac=0.1, random_state=0)
test_dataset = norm_data.drop(train_dataset.index)
train_labels = train_dataset.pop('Torque')
test_labels = test_dataset.pop('Torque')
# TODO: Ooooog þjálfa

model = ann.build_model(train_dataset)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

EPOCHS = 1000

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split=0.2, verbose=0,
  callbacks=[early_stop, ann.PrintDot()])

ann.plot_history(history)

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)

print("Testing set knee torque Mean Abs Error: {:5.2f} Nmm".format(mae))

test_predictions = model.predict(test_dataset).flatten()

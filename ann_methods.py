import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import filters as filt


def build_model(train_dataset):
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def create_train_and_test(df, frac=0.8, randomize_by_exercise=True):
    dataset = df.copy()
    if randomize_by_exercise:
        exercise_groups = dataset.groupby('Exercise')
        group_num = np.arange(exercise_groups.ngroups)
        np.random.shuffle(group_num)

        train_dataset = dataset[
            exercise_groups.ngroup().isin(group_num[:np.floor(frac * len(group_num) - 1).astype('int')])
        ]
    else:
        train_dataset = dataset.sample(frac=frac, random_state=0)

    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset


def prepare_data(data_dict, subject_id=None):
    # TODO: unfinished, needs to be evaluated and finished
    for key in data_dict:
        if subject_id is None:
            dataset = data_dict[key].copy()
            train_dataset, test_dataset = create_train_and_test(dataset, frac=0.8)
            train_dataset.pop('Exercise')
            train_dataset.pop('Time')
            test_exercise = test_dataset.pop('Exercise')
            test_times = test_dataset.pop('Time')

            norm_train_data = filt.min_max_normalize_data(train_dataset, norm_emg=True, norm_torque=True)
            norm_test_data = filt.min_max_normalize_data(test_dataset, secondary_df=train_dataset, norm_emg=True,
                                                         norm_torque=True)
        elif subject_id in key:
            dataset = data_dict[key].copy()
            train_dataset, test_dataset = create_train_and_test(dataset, frac=0.8)
            train_dataset.pop('Exercise')
            train_dataset.pop('Time')
            test_exercise = test_dataset.pop('Exercise')
            test_times = test_dataset.pop('Time')

            norm_train_data = filt.min_max_normalize_data(train_dataset, norm_emg=True, norm_torque=True)
            norm_test_data = filt.min_max_normalize_data(test_dataset, secondary_df=train_dataset, norm_emg=True,
                                                         norm_torque=True)

            break


def split_trials_by_duration(df, time_lim=None):
    normal_walk = df.copy()
    durations = normal_walk.groupby(['Exercise'])['Time'].max().to_frame(name='Time')
    durations.reset_index(level=0, inplace=True)
    if time_lim is None:
        min_time = durations.Time.min()
        max_time = durations.Time.max()
        time_lim = (min_time + (max_time - min_time)/3, min_time + 2 * (max_time - min_time)/3)

    fast_walk = normal_walk[normal_walk.Exercise.isin(durations[durations.Time < time_lim[0]].Exercise.values)]
    slow_walk = normal_walk[normal_walk.Exercise.isin(durations[durations.Time > time_lim[1]].Exercise.values)]
    normal_walk = normal_walk.drop(fast_walk.index)
    normal_walk = normal_walk.drop(slow_walk.index)

    return slow_walk, normal_walk, fast_walk


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.show()




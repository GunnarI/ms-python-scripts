import os
import warnings

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import optimizers

import filters as filt


class ANN:
    def __init__(self, dataset=None, spec_cache_code=None, load_from_cache=None):
        if load_from_cache is None and dataset is not None:
            self.cache_path = ''
            self.create_cache_dir(spec_cache_code)
            self.dataset = dataset
            self.train_dataset = pd.DataFrame()
            self.test_dataset = pd.DataFrame()
            self.prepare_data()
            self.models = {}
            self.model_histories = {}
        else:
            if load_from_cache is None:
                cached_folders = Path('./cache/').glob('[0-9]*?*')
                newest_cache = 0
                for folder in cached_folders:
                    folder_time_stamp = int(folder.stem.split('_')[0])
                    if folder_time_stamp > newest_cache:
                        newest_cache = folder_time_stamp
                        load_from_cache = folder.stem

            self.cache_path = './cache/' + load_from_cache + '/'
            if os.path.exists(self.cache_path):
                self.load_datasets()
                self.models = {}
                self.model_histories = {}
                self.load_models()
            else:
                raise ValueError('A cache path for ' + load_from_cache + ' was not found')

    def create_train_and_test(self, frac=0.8, randomize_by_trial=True):
        """ Splits a pandas.DataFrame into train-, and test-dataset

        :param df: The DataFrame
        :param frac: The fraction of the DataFrame that will be used as training dataset, default: 0.8.
        Note if randomize_by_trial=True then this is the fraction of trials/cycles used, where number of samples within
        trial/cycle may vary
        :param randomize_by_trial: If True then the split is done trial/cycle wise such that all samples within a trial
        or cycle are put together into a dataset, default: True
        :return: the two datasets for training and testing: train_dataset, test_dataset
        """
        dataset = self.dataset.copy()
        if randomize_by_trial:
            trial_groups = dataset.groupby('Trial')
            group_num = np.arange(trial_groups.ngroups)
            np.random.shuffle(group_num)

            train_dataset = dataset[
                trial_groups.ngroup().isin(group_num[:np.floor(frac * len(group_num) - 1).astype('int')])
            ]

            trial_groups = [df for _, df in train_dataset.groupby('Trial')]
            np.random.shuffle(trial_groups)
            train_dataset = pd.concat(trial_groups)
        else:
            train_dataset = dataset.sample(frac=frac, random_state=0)

        test_dataset = dataset.drop(train_dataset.index)
        self.train_dataset = train_dataset.reset_index(drop=True)
        self.test_dataset = test_dataset.reset_index(drop=True)

    def prepare_data(self, split_frac=0.8, normalize=True):
        self.dataset.dropna(inplace=True)
        self.create_train_and_test(frac=split_frac)

        if normalize:
            self.test_dataset = filt.min_max_normalize_data(self.test_dataset, secondary_df=self.train_dataset,
                                                            norm_emg=True, norm_torque=True)
            self.train_dataset = filt.min_max_normalize_data(self.train_dataset, norm_emg=True, norm_torque=True)

        self.save_datasets()

    def update_train_and_test(self, frac=0.8, randomize_by_trial=True, normalize=True):
        self.create_train_and_test(frac=frac, randomize_by_trial=randomize_by_trial)
        if normalize:
            self.test_dataset = filt.min_max_normalize_data(self.test_dataset, secondary_df=self.train_dataset,
                                                            norm_emg=True, norm_torque=True)
            self.train_dataset = filt.min_max_normalize_data(self.train_dataset, norm_emg=True, norm_torque=True)

    def train_rnn_w_teacher_forcing(self, optimizer='rmsprop', model_name='mlp', layers_nodes=None, epochs=1000,
                                    val_split=0.2, early_stop_patience=None, dropout_rates=None):
        train_dataset = self.train_dataset.copy()
        if 'Time' in train_dataset.columns:
            train_dataset.pop('Time')
        if 'Trial' in train_dataset.columns:
            train_dataset.pop('Trial')

        train_labels = train_dataset.pop('Torque')
        train_dataset['TeachForce'] = pd.Series(np.roll(train_labels.values, 1, axis=0))

        if layers_nodes is None:
            layers_nodes = [(64, 'relu'), (64, 'relu')]

        if len(dropout_rates) != len(layers_nodes):
            warnings.warn('Number of dropout rates did not match layers and is thus excluded from the model')
            dropout_rates = None

        model = keras.Sequential()
        model.add(layers.Dense(layers_nodes[0][0], input_shape=[len(train_dataset.keys())]))
        model.add(layers.Activation(layers_nodes[0][1]))
        if dropout_rates is not None:
            model.add(layers.Dropout(dropout_rates[0]))
        for i, layer in enumerate(layers_nodes[1:]):
            model.add(layers.Dense(layer[0]))
            model.add(layers.Activation(layer[1]))
            if dropout_rates is not None:
                model.add(layers.Dropout(dropout_rates[i+1]))

        model.add(layers.Dense(1))

        if optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop()
        elif optimizer == 'adam':
            optimizer = optimizers.Adam()

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        callbacks = []
        if early_stop_patience is not None:
            if not isinstance(early_stop_patience, int):
                warnings.warn('The val_patience should be an integer representing epoch patience of early stopping')
            else:
                callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience))

        callbacks.append(keras.callbacks.CSVLogger(self.cache_path + 'history/' + model_name + '.log',
                                                   separator=',', append=False))
        model.fit(
            train_dataset, train_labels, shuffle=False,
            epochs=epochs, validation_split=val_split, verbose=1,
            callbacks=callbacks)

        self.save_model(model, model_name)

    def train_lstm(self, optimizer='rmsprop', model_name='lstm', num_nodes=32, epochs=50, val_split=0.2,
                   early_stop_patience=None, dropout_rate=0.3, look_back=1, activation_func='tanh'):
        train_dataset = self.train_dataset.copy()
        if 'Time' in train_dataset.columns:
            train_dataset.pop('Time')

        num_features = len(train_dataset.columns) - 2

        model = keras.Sequential()
        model.add(layers.LSTM(num_nodes, return_sequences=True, input_shape=(None, num_features),
                              activation=activation_func, dropout=dropout_rate))
        model.add(layers.TimeDistributed(layers.Dense(1)))

        if optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop()
        elif optimizer == 'adam':
            optimizer = optimizers.Adam()

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        callbacks = []
        if early_stop_patience is not None:
            if not isinstance(early_stop_patience, int):
                warnings.warn('The val_patience should be an integer representing epoch patience of early stopping')
            else:
                callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience))

        callbacks.append(keras.callbacks.CSVLogger(self.cache_path + 'history/' + model_name + '.log',
                                                   separator=',', append=False))
        callbacks.append(keras.callbacks.TensorBoard(log_dir=self.cache_path + 'tensorboard/' + model_name))

        trial_groups = train_dataset.groupby('Trial')
        group_num = np.arange(trial_groups.ngroups)
        np.random.shuffle(group_num)

        train_set = train_dataset[
            trial_groups.ngroup().isin(group_num[:np.floor((1-val_split) * len(group_num) - 1).astype('int')])
        ]

        train_set = [df for _, df in train_set.groupby('Trial')]
        drop_df = pd.concat(train_set)
        validation_set = train_dataset.drop(drop_df.index)
        validation_set = [df for _, df in validation_set.groupby('Trial')]
        # train_group_list = list(train_dataset.groupby('Trial').groups.keys())

        def train_generator():
            i = 0
            while True:
                dataset = train_set[i].copy()
                dataset.pop('Trial')
                train_labels = dataset.pop('Torque')
                x_train = np.array([dataset.values])
                y_train = np.zeros((1, len(train_labels), 1))
                y_train[0, :, 0] = train_labels.values

                yield x_train, y_train

        def val_generator():
            j = 0
            while True:
                dataset = validation_set[j].copy()
                dataset.pop('Trial')
                validation_labels = dataset.pop('Torque')
                x_train = np.array([dataset.values])
                y_train = np.zeros((1, len(validation_labels), 1))
                y_train[0, :, 0] = validation_labels.values

                yield x_train, y_train

        model.fit_generator(train_generator(), steps_per_epoch=len(train_set), epochs=epochs, verbose=1,
                            validation_data=val_generator(), validation_steps=len(validation_set), callbacks=callbacks)
        # model.fit(
        #     train_dataset, train_labels,
        #     batch_size=1, epochs=epochs, validation_split=val_split, verbose=1,
        #     callbacks=callbacks)

        self.save_model(model, model_name)

    def train_mlp(self, optimizer='rmsprop', model_name='mlp', layers_nodes=None, epochs=1000, val_split=0.2,
                  early_stop_patience=None, dropout_rates=None):
        train_dataset = self.train_dataset.copy()
        if 'Time' in train_dataset.columns:
            train_dataset.pop('Time')
        if 'Trial' in train_dataset.columns:
            train_dataset.pop('Trial')

        train_labels = train_dataset.pop('Torque')

        if layers_nodes is None:
            layers_nodes = [(64, 'relu'), (64, 'relu')]

        if len(dropout_rates) != len(layers_nodes):
            warnings.warn('Number of dropout rates did not match layers and is thus excluded from the model')
            dropout_rates = None

        model = keras.Sequential()
        model.add(layers.Dense(layers_nodes[0][0], input_shape=[len(train_dataset.keys())], name='MLP_Input_Layer'))
        model.add(layers.Activation(layers_nodes[0][1]))
        if dropout_rates is not None:
            model.add(layers.Dropout(dropout_rates[0]))
        for i, layer in enumerate(layers_nodes[1:]):
            model.add(layers.Dense(layer[0], name='Hidden_layer_' + str(i+1)))
            model.add(layers.Activation(layer[1]))
            if dropout_rates is not None:
                model.add(layers.Dropout(dropout_rates[i+1]))

        model.add(layers.Dense(1, name='MLP_Moment_Output'))

        if optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop()
        elif optimizer == 'adam':
            optimizer = optimizers.Adam()

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        callbacks = []
        if early_stop_patience is not None:
            if not isinstance(early_stop_patience, int):
                warnings.warn('The val_patience should be an integer representing epoch patience of early stopping')
            else:
                callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience))

        callbacks.append(keras.callbacks.CSVLogger(self.cache_path + 'history/' + model_name + '.log',
                                                   separator=',', append=False))
        callbacks.append(keras.callbacks.TensorBoard(log_dir=self.cache_path + 'tensorboard/' + model_name,
                                                     embeddings_layer_names=['MLP_Input_Layer', 'Hidden_layer_1',
                                                                             'MLP_Moment_Output'],
                                                     write_graph=True))

        model.fit(
            train_dataset, train_labels,
            epochs=epochs, validation_split=val_split, verbose=1,
            callbacks=callbacks)

        self.save_model(model, model_name)

    def evaluate_model(self, model_name, lstm=False):
        test_dataset = self.test_dataset.copy()
        if 'Time' in test_dataset.columns:
            test_dataset.pop('Time')
        if 'Trial' in test_dataset.columns:
            test_dataset.pop('Trial')

        test_labels = test_dataset.pop('Torque')

        plot_history(self.model_histories[model_name])

        if lstm:
            test_dataset = np.array([test_dataset.values])
            y = np.zeros((1, len(test_labels), 1))
            y[0, :, 0] = test_labels.values
            test_labels = y

        loss, mae, mse = self.models[model_name].evaluate(test_dataset, test_labels, verbose=0)
        print("Testing set knee torque Loss: {:5.4f}\n"
              "\t\t\t\t\t\tMean Abs Error: {:5.4f}\n"
              "\t\t\t\t\t\tMean Square Error: {:5.4f}".format(loss, mae, mse))

    def create_cache_dir(self, spec_cache_code=None):
        if spec_cache_code is None:
            self.cache_path = './cache/' + time.strftime('%Y%m%d%H%M') + '/'
        else:
            self.cache_path = './cache/' + time.strftime('%Y%m%d%H%M') + '_' + spec_cache_code + '/'
        try:
            os.mkdir(self.cache_path)
            os.mkdir(self.cache_path + 'models/')
            os.mkdir(self.cache_path + 'dataframes/')
            os.mkdir(self.cache_path + 'history/')
        except FileExistsError:
            print('Cache directory ', self.cache_path, ' already exists')

    def save_datasets(self):
        self.dataset.to_pickle(self.cache_path + 'dataframes/dataset.pkl')
        self.test_dataset.to_pickle(self.cache_path + 'dataframes/test_dataset.pkl')
        self.train_dataset.to_pickle(self.cache_path + 'dataframes/train_dataset.pkl')

    def load_datasets(self):
        self.dataset = pd.read_pickle(self.cache_path + 'dataframes/dataset.pkl')
        self.train_dataset = pd.read_pickle(self.cache_path + 'dataframes/train_dataset.pkl')
        self.test_dataset = pd.read_pickle(self.cache_path + 'dataframes/test_dataset.pkl')

    def save_model(self, model, model_name):
        model.save(self.cache_path + 'models/' + model_name + '.h5')
        self.models[model_name] = model
        self.model_histories[model_name] = pd.read_csv(self.cache_path + 'history/' + model_name + '.log',
                                                       sep=',',
                                                       engine='python')

    def load_models(self):
        models_paths = Path(self.cache_path + 'models/').glob('*.h5')
        for model_path in models_paths:
            self.models[model_path.stem] = load_model(str(model_path))

        history_paths = Path(self.cache_path + 'history/').glob('*.log')
        for history_path in history_paths:
            try:
                self.model_histories[history_path.stem] = pd.read_csv(str(history_path), sep=',', engine='python')
            except pd.errors.EmptyDataError:
                print('Logger: ', history_path.stem, ' is empty')
                continue

    def plot_test(self, model_name, cycle_to_plot='random', lstm=False, title=None, save_fig_as=None):
        if cycle_to_plot == 'random':
            cycle_to_plot = self.test_dataset.sample(n=1).Trial.to_string(index=False).strip()
        test_cycle = self.test_dataset[self.test_dataset.Trial == cycle_to_plot]
        test_time = test_cycle.pop('Time')
        test_torque = test_cycle.pop('Torque')
        test_trial = test_cycle.pop('Trial')

        if lstm:
            test_cycle = np.array([test_cycle.values])

        test_prediction = self.models[model_name].predict(test_cycle)

        if lstm:
            test_prediction = test_prediction[0, :, 0]

        if title is None:
            title = cycle_to_plot
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()
        fig.add_subplot(ax1)
        ax1.set_title(title)
        ax1.set_xlabel('gait cycle duration[s]')
        ax1.set_ylabel('normalized joint moment')
        ax1.plot(test_time, test_torque, label='Test cycle')
        ax1.plot(test_time, test_prediction, label='Prediction')
        ax1.legend()

        if save_fig_as is None:
            fig.show()
        else:
            fig.savefig('./figures/ann/' + save_fig_as + '.png', bbox_inches='tight')

    def plot_train(self, model_name, cycle_to_plot='random', title=None, save_fig_as=None):
        if cycle_to_plot == 'random':
            cycle_to_plot = self.train_dataset.sample(n=1).Trial.to_string(index=False).strip()
        train_cycle = self.train_dataset[self.train_dataset.Trial == cycle_to_plot]
        train_time = train_cycle.pop('Time')
        train_torque = train_cycle.pop('Torque')
        train_trial = train_cycle.pop('Trial')
        train_prediction = self.models[model_name].predict(train_cycle)

        if title is None:
            title = cycle_to_plot
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()
        fig.add_subplot(ax1)
        ax1.set_title(title)
        ax1.set_xlabel('gait cycle duration[s]')
        ax1.set_ylabel('normalized joint moment')
        ax1.plot(train_time, train_torque, label='Test cycle')
        ax1.plot(train_time, train_prediction, label='Prediction')
        ax1.legend()

        if save_fig_as is None:
            fig.show()
        else:
            fig.savefig('./figures/ann/' + save_fig_as + '.png', bbox_inches='tight')


def split_trials_by_duration(df, time_lim=None):
    normal_walk = df.copy()
    durations = normal_walk.groupby(['Trial'])['Time'].max().to_frame(name='Time')
    durations.reset_index(level=0, inplace=True)
    if time_lim is None:
        min_time = durations.Time.min()
        max_time = durations.Time.max()
        time_lim = (min_time + (max_time - min_time)/3, min_time + 2 * (max_time - min_time)/3)

    fast_walk = normal_walk[normal_walk.Trial.isin(durations[durations.Time < time_lim[0]].Trial.values)]
    slow_walk = normal_walk[normal_walk.Trial.isin(durations[durations.Time > time_lim[1]].Trial.values)]
    normal_walk = normal_walk.drop(fast_walk.index)
    normal_walk = normal_walk.drop(slow_walk.index)

    return slow_walk, normal_walk, fast_walk


def plot_history(hist):
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

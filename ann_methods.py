import os
import warnings

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import optimizers

import filters as filt
import stat_analysis_functions as saf


class ANN:
    def __init__(self, dataset=None, spec_cache_code=None, load_from_cache=None, load_models=None):
        self.models = {}
        self.model_histories = {}
        self.model_sessions = {}
        self.model_predictions = {}
        if load_from_cache is None and dataset is not None:
            self.cache_path = ''
            self.create_cache_dir(spec_cache_code)
            self.dataset = dataset
            self.normalized_dataset = pd.DataFrame()
            self.train_dataset = pd.DataFrame()
            self.test_dataset = pd.DataFrame()
            self.prepare_data()
        else:
            if load_from_cache is None:
                cached_folders = Path('./cache/ann/').glob('[0-9]*?*')
                newest_cache = 0
                for folder in cached_folders:
                    folder_time_stamp = int(folder.stem.split('_')[0])
                    if folder_time_stamp > newest_cache:
                        newest_cache = folder_time_stamp
                        load_from_cache = folder.stem

            self.cache_path = './cache/ann/' + load_from_cache + '/'
            if os.path.exists(self.cache_path):
                self.load_datasets()
                if load_models is None:
                    self.load_models()
                else:
                    for model_name in load_models:
                        self.load_models(model_name)
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
            fast_cycles = dataset[dataset['Trial'].str.contains('fast', case=False)]
            slow_cycles = dataset[dataset['Trial'].str.contains('slow', case=False)]
            normal_cycles = dataset.drop(fast_cycles.index)
            normal_cycles = normal_cycles.drop(slow_cycles.index)

            fast_groups = fast_cycles.groupby('Trial')
            slow_groups = slow_cycles.groupby('Trial')
            normal_groups = normal_cycles.groupby('Trial')

            fast_group_num = np.arange(fast_groups.ngroups)
            slow_group_num = np.arange(slow_groups.ngroups)
            normal_group_num = np.arange(normal_groups.ngroups)

            np.random.shuffle(fast_group_num)
            np.random.shuffle(slow_group_num)
            np.random.shuffle(normal_group_num)

            fast_dataset = fast_cycles[
                fast_groups.ngroup().isin(fast_group_num[:np.floor(frac * len(fast_group_num) - 1).astype('int')])
            ]
            slow_dataset = slow_cycles[
                slow_groups.ngroup().isin(slow_group_num[:np.floor(frac * len(slow_group_num) - 1).astype('int')])
            ]
            normal_dataset = normal_cycles[
                normal_groups.ngroup().isin(normal_group_num[:np.floor(frac * len(normal_group_num) - 1).astype('int')])
            ]

            train_dataset = pd.concat([fast_dataset, slow_dataset, normal_dataset])

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

            self.normalized_dataset = pd.concat([self.train_dataset, self.test_dataset], ignore_index=True)
        self.save_datasets()

    def update_train_and_test(self, frac=0.8, randomize_by_trial=True, normalize=True):
        self.create_train_and_test(frac=frac, randomize_by_trial=randomize_by_trial)
        if normalize:
            self.test_dataset = filt.min_max_normalize_data(self.test_dataset, secondary_df=self.train_dataset,
                                                            norm_emg=True, norm_torque=True)
            self.train_dataset = filt.min_max_normalize_data(self.train_dataset, norm_emg=True, norm_torque=True)

            self.normalized_dataset = pd.concat([self.train_dataset, self.test_dataset], ignore_index=True)

    def train_rnn_w_teacher_forcing(self, optimizer='rmsprop', model_name='mlp', layers_nodes=None, epochs=1000,
                                    val_split=0.2, early_stop_patience=None, dropout_rates=None):
        train_dataset = self.train_dataset.copy()
        train_dataset.drop(columns=['Time', 'Trial'], errors='ignore', inplace=True)

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

    def train_lstm(self, initializer='glorot_uniform', optimizer='rmsprop', learning_rate=None, model_name='lstm',
                   num_nodes=32, epochs=100, val_split=0.2, early_stop_patience=None, dropout_rate=0.3, look_back=3,
                   activation_func='relu'):
        x_train, y_train = gen_lstm_dataset(self.train_dataset.copy(), look_back)

        num_features = x_train.shape[2]

        K.clear_session()

        emg_inputs = keras.Input(shape=(look_back, num_features), name='emg')
        x = layers.LSTM(num_nodes, activation=activation_func, dropout=dropout_rate, name='LSTM_layer')(emg_inputs)
        prediction = layers.Dense(1, name='output_layer')(x)
        model = keras.Model(inputs=emg_inputs, outputs=prediction, name=model_name)

        if optimizer == 'rmsprop':
            if learning_rate is not None:
                optimizer = optimizers.RMSprop(lr=learning_rate)
            else:
                optimizer = optimizers.RMSprop()
        elif optimizer == 'adam':
            if learning_rate is not None:
                optimizer = optimizers.Adam(lr=learning_rate)
            else:
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
                                                     histogram_freq=5,
                                                     write_grads=True))
        callbacks.append(keras.callbacks.ModelCheckpoint(self.cache_path + 'models/' + model_name + '_best.h5',
                                                         save_best_only=True))

        model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_split=val_split, callbacks=callbacks)

        self.save_model(model, model_name)
        self.load_models(model_name=model_name + '_best')

    def train_lstm_old(self, optimizer='rmsprop', model_name='lstm', num_nodes=32, epochs=50, val_split=0.2,
                   early_stop_patience=None, dropout_rate=0.3, look_back=1, activation_func='tanh'):
        train_dataset = self.train_dataset.copy()
        train_dataset.drop(columns=['Time'], errors='ignore', inplace=True)

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

    def train_mlp(self, optimizer='rmsprop', model_name='mlp', layers_nodes=None, epochs=100, val_split=0.2,
                  early_stop_patience=None, dropout_rates=None):
        train_dataset = self.train_dataset.copy()
        train_dataset.drop(columns=['Time', 'Trial'], errors='ignore', inplace=True)

        train_labels = train_dataset.pop('Torque')
        num_emg = len([x for x in train_dataset.columns if x not in ['Time', 'Torque', 'Trial']])

        if layers_nodes is None:
            layers_nodes = [(64, 'relu'), (64, 'relu')]
        if dropout_rates is not None:
            if len(dropout_rates) != len(layers_nodes):
                warnings.warn('Number of dropout rates did not match layers and is thus excluded from the model')
                dropout_rates = None

        K.clear_session()

        emg_inputs = keras.Input(shape=(num_emg, ), name='emg')
        x = layers.Dense(layers_nodes[0][0], activation=layers_nodes[0][1], name='dense_1')(emg_inputs)
        if dropout_rates is not None:
            x = layers.Dropout(dropout_rates[0], name='dropout_1')(x)
        for i, layer in enumerate(layers_nodes[1:]):
            x = layers.Dense(layer[0], activation=layer[1], name='dense_' + str(i + 2))(x)
            if dropout_rates is not None:
                x = layers.Dropout(dropout_rates[i], name='dropout_' + str(i + 2))(x)

        prediction = layers.Dense(1, name='mlp_moment_output')(x)

        model = keras.Model(inputs=emg_inputs, outputs=prediction, name=model_name)

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
                                                     write_graph=True))
        callbacks.append(keras.callbacks.ModelCheckpoint(self.cache_path + 'models/' + model_name + '_best.h5',
                                                         save_best_only=True))

        model.fit(
            train_dataset, train_labels,
            epochs=epochs, validation_split=val_split, verbose=1,
            callbacks=callbacks)

        self.save_model(model, model_name)
        self.load_models(model_name=model_name + '_best')

    def evaluate_model(self, model_name, lstm=False, lstm_look_back=3):
        test_dataset = self.test_dataset.copy()
        moment_avg = test_dataset.groupby('Time')['Torque'].mean()

        if lstm:
            test_dataset, test_labels = gen_lstm_dataset(test_dataset, lstm_look_back)
        else:
            test_dataset.drop(columns=['Time', 'Trial'], errors='ignore', inplace=True)
            test_labels = test_dataset.pop('Torque')

        if model_name in self.model_histories.keys():
            plot_history(self.model_histories[model_name])

        K.set_session(self.model_sessions[model_name])
        with K.get_session().graph.as_default():
            loss, mae, mse = self.models[model_name].evaluate(x=test_dataset, y=test_labels, verbose=0)
            if model_name in self.model_predictions:
                t = np.linspace(0.0, len(moment_avg)/100, len(moment_avg))
                moment_avg_figure = saf.plot_moment_avg(self.model_predictions[model_name])
                ax1 = moment_avg_figure.get_axes()[0]
                ax1.plot(t, moment_avg)
        print("Testing set knee torque Loss: {:5.4f}\n"
              "\t\t\t\t\t\tMean Abs Error: {:5.4f}\n"
              "\t\t\t\t\t\tMean Square Error: {:5.4f}".format(loss, mae, mse))

    def create_cache_dir(self, spec_cache_code=None):
        if spec_cache_code is None:
            self.cache_path = './cache/ann/' + time.strftime('%Y%m%d%H%M') + '/'
        else:
            self.cache_path = './cache/ann/' + time.strftime('%Y%m%d%H%M') + '_' + spec_cache_code + '/'
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
        self.normalized_dataset.to_pickle(self.cache_path + 'dataframes/normalized_dataset.pkl')

    def load_datasets(self):
        self.dataset = pd.read_pickle(self.cache_path + 'dataframes/dataset.pkl')
        self.train_dataset = pd.read_pickle(self.cache_path + 'dataframes/train_dataset.pkl')
        self.test_dataset = pd.read_pickle(self.cache_path + 'dataframes/test_dataset.pkl')
        self.normalized_dataset = pd.read_pickle(self.cache_path + 'dataframes/normalized_dataset.pkl')

    def add_model_prediction(self, model_name):
        if model_name in self.model_sessions and model_name in self.models:
            test_set = self.test_dataset.copy()
            prediction_set = self.test_dataset.copy()
            cycles = list(test_set.groupby('Trial').apply(np.unique).index)

            trials = test_set.pop('Trial')
            test_set.drop(columns=['Time', 'Torque'], errors='ignore', inplace=True)

            K.set_session(self.model_sessions[model_name])
            with K.get_session().graph.as_default():
                for cycle in cycles:
                    prediction_set.loc[prediction_set.Trial == cycle, 'Torque'] = self.models[model_name].predict(
                        test_set[trials == cycle])[:, 0]
            self.model_predictions[model_name] = prediction_set

            K.clear_session()
        else:
            warnings.warn('Model name ' + model_name + ' not found in existing models or model sessions')

    def save_model(self, model, model_name):
        self.model_sessions[model_name] = K.get_session()
        model.save(self.cache_path + 'models/' + model_name + '.h5')
        self.models[model_name] = model
        self.model_histories[model_name] = pd.read_csv(self.cache_path + 'history/' + model_name + '.log',
                                                       sep=',',
                                                       engine='python')

    def load_models(self, model_name=None):
        models_paths = Path(self.cache_path + 'models/').glob('*.h5')
        for model_path in models_paths:
            if model_name is None:
                K.clear_session()
                self.models[model_path.stem] = load_model(str(model_path))
                self.model_sessions[model_path.stem] = K.get_session()
                K.clear_session()
            elif model_name == model_path.stem:
                K.clear_session()
                self.models[model_path.stem] = load_model(str(model_path))
                self.model_sessions[model_path.stem] = K.get_session()
                K.clear_session()
                break

        history_paths = Path(self.cache_path + 'history/').glob('*.log')
        for history_path in history_paths:
            try:
                self.model_histories[history_path.stem] = pd.read_csv(str(history_path), sep=',', engine='python')
            except pd.errors.EmptyDataError:
                print('Logger: ', history_path.stem, ' is empty')
                continue

    def plot_test(self, model_name, cycle_to_plot='random', lstm=False, lstm_look_back=3, use_train_set=False,
                  title=None, save_fig_as=None):
        if use_train_set:
            dataset = self.train_dataset.copy()
        else:
            dataset = self.test_dataset.copy()
        if cycle_to_plot == 'random':
            cycle_to_plot = [dataset.sample(n=1).Trial.to_string(index=False).strip()]
        elif cycle_to_plot == 'worst':
            cycle_to_plot = [cycle for cycle, _ in dataset.groupby('Trial')]
        elif not isinstance(cycle_to_plot, list):
            cycle_to_plot = [cycle_to_plot]

        prediction_mse = 0
        time_vec = None
        predictions = None
        labels = None

        for cycle in cycle_to_plot:
            test_cycle = dataset[dataset.Trial == cycle]
            test_time = test_cycle.pop('Time')

            if lstm:
                test_cycle, test_labels = gen_lstm_dataset(test_cycle, lstm_look_back)
                test_time = test_time[lstm_look_back-1:]
            else:
                test_cycle.drop(columns=['Time', 'Trial'], inplace=True, errors='ignore')
                test_labels = test_cycle.pop('Torque')

            K.set_session(self.model_sessions[model_name])
            with K.get_session().graph.as_default():
                test_prediction = self.models[model_name].predict(test_cycle)

            if lstm:
                test_prediction = test_prediction[:, 0]

            temp_mse = np.square(np.subtract(test_labels, test_prediction)).mean()
            if temp_mse > prediction_mse:
                cycle_name = cycle
                print(cycle_name)
                time_vec = test_time
                predictions = test_prediction
                labels = test_labels
                prediction_mse = temp_mse

        if title is None:
            title = cycle_name
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()
        fig.add_subplot(ax1)
        ax1.set_title(title)
        ax1.set_xlabel('gait cycle duration[s]')
        ax1.set_ylabel('normalized joint moment')
        ax1.plot(time_vec, labels, label='Test cycle')
        ax1.plot(time_vec, predictions, label='Prediction')
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
        train_labels = train_cycle.pop('Torque')
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
        ax1.plot(train_time, train_labels, label='Test cycle')
        ax1.plot(train_time, train_prediction, label='Prediction')
        ax1.legend()

        if save_fig_as is None:
            fig.show()
        else:
            fig.savefig('./figures/ann/' + save_fig_as + '.png', bbox_inches='tight')

    def get_test_prediction(self, model_name, cycle_to_predict='all', lstm=False, lstm_look_back=3):
        dataset = self.test_dataset.copy()
        if not cycle_to_predict == 'all':
            if cycle_to_predict == 'random':
                cycle_to_predict = [dataset.sample(n=1).Trial.to_string(index=False).strip()]
            elif cycle_to_predict == 'worst':
                cycle_to_predict = [cycle for cycle, _ in dataset.groupby('Trial')]
            elif not isinstance(cycle_to_predict, list):
                cycle_to_predict = [cycle_to_predict]

            test_cycle = dataset[dataset.Trial == cycle_to_predict]
        else:
            test_cycle = dataset.copy()

        if lstm:
            test_cycle, test_labels = gen_lstm_dataset(test_cycle, lstm_look_back)
        else:
            test_cycle.drop(columns=['Time', 'Trial'], inplace=True, errors='ignore')
            test_labels = test_cycle.pop('Torque')

        K.set_session(self.model_sessions[model_name])
        with K.get_session().graph.as_default():
            test_prediction = self.models[model_name].predict(test_cycle)

        if lstm:
            test_prediction = test_prediction[:, 0]

        if cycle_to_predict == 'all':
            # dataset['Torque'] = test_prediction
            df_to_return = []
            for _, df in dataset.groupby('Trial'):
                df_to_return.append(df[lstm_look_back-1:])
            df_to_return = pd.concat(df_to_return)
            df_to_return['Torque'] = test_prediction
            # df_to_return = dataset
        else:
            df_to_return = dataset[dataset.Trial == cycle_to_predict]
            df_to_return['Torque'] = test_prediction

        return df_to_return


def gen_lstm_dataset(df, look_back):
    """
    Prepares the dataset for training the LSTM depending on how many past timesteps should be used (i.e. look_back)
    :param pandas.DataFrame df: A DataFrame dataset with all the emg data, with dimensions (num_timesteps, num_features)
    :param int look_back: Number of past timesteps to look at for prediction
    :return: A numpy.array containing the LSTM prepared dataset with dimensions (num_timesteps, look_back, num_features)
    """
    df_copy = df.copy()

    inputs = list()
    labels = list()

    for _, trial_group in df_copy.groupby('Trial', sort=False):
        trial_group.drop(columns=['Time', 'Trial'], errors='ignore', inplace=True)
        group_labels = trial_group.pop('Torque')

        group_inputs = trial_group.to_numpy(dtype=float)
        group_labels = group_labels.to_numpy(dtype=float)

        for i in range(len(group_labels)):
            window_end = i + look_back
            if window_end > len(group_labels):
                break

            inputs.append(group_inputs[i:window_end, :])
            labels.append(group_labels[window_end-1])

    return np.array(inputs), np.array(labels)


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

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    def __init__(self, dataset=None, spec_cache_code=None, load_from_cache=None, load_models=None,
                 train_dataset=None, test_dataset=None, normalize_data=True):
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
            self.prepare_data(normalize=normalize_data)
        elif isinstance(train_dataset, pd.DataFrame) and isinstance(test_dataset, pd.DataFrame):
            warnings.warn('When assigning datasets directly, they are not normalized or split automatically.')
            self.cache_path = ''
            self.create_cache_dir(spec_cache_code)
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
            self.normalized_dataset = self.dataset
            self.save_datasets()
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
                raise ValueError('A cache path for ' + load_from_cache + ' was not found and no datasets given.')

    def create_train_and_test(self, frac=0.8, randomize_by_trial=True, validation_frac=0.2):
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
        train_dataset, test_dataset = split_df_by_frac(dataset, frac=0.8, randomize_by_trial=True)

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

    def train_lstm_w_mlp(self, initializer='glorot_uniform', optimizer='rmsprop', learning_rate=None, model_name='lstm',
                         num_nodes=32, epochs=100, val_split=0.2, early_stop_patience=None, dropout_rate=0.1,
                         mlp_dropout_rate=0.1, recurrent_dropout_rate=0.5, look_back=3, activation_func='relu',
                         tensorboard=True, keep_training_model=False, initial_epoch=0, batch_size_case=1):
        """
                Trains a DNN based on a number of recurrent LSTM based layers and a single Dense output layer for regression.
                The regression parameter represents the joint moment prediction from the emg input data.
                :param str initializer: the weight initializer used for the keras LSTM layer (default 'glorot_uniform').
                See https://keras.io/initializers/ \n
                :param str optimizer: the optimizer used for the training, limited to 'rmsprop' or 'adam' (default 'rmsprop').
                See https://keras.io/optimizers/ \n
                :param float learning_rate: the learning rate used for the optimizer. If None then the internal default of the optimizer is used
                :param str model_name: the name of the model as it will be saved in cache and self.models (default 'lstm').
                :param int num_nodes: the number of hidden nodes in each unfolded LSTM layer (each timestep)
                :param int epochs: number of epochs to run (default: 100). The number needs to be larger than initial_epoch (see initial_epoch).
                :param float val_split: a percentage of dataset used as validation set, can be from 0 to 1 (default: 0.2)
                :param int early_stop_patience: number of epochs used for the keras.callbacks.EarlyStopping, monitoring the validation loss (default: None)
                :param float dropout_rate: float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs (default: 0.1).
                :param float recurrent_dropout_rate: float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state (default: 0.3).
                :param int look_back: number of timesteps to include for each sample input to the LSTM (default: 3).
                :param str activation_func: the activation function used for the network (default: 'relu').
                :param bool tensorboard: if True then tensorboard callback is used (default: True).
                :param bool keep_training_model: if model_name already exists in cache and keep_training_model is True then training of the model continues from the point when it was cached (default: False).
                :param int initial_epoch: the epoch number to start from (default: 0). If restarting training from cached model then this value should typically be the next epoch number from where the cached model stopped.
                :param int batch_size_case: either 1 or 2 (default: 1). Case 1 makes all the cycles of equal length by zero padding and uses batch size as the length of these cycles
                :return:
                """
        x_train, y_train, longest_cycle = gen_lstm_dataset(self.train_dataset.copy(), look_back,
                                                           batch_size_case=batch_size_case)

        if batch_size_case == 1:
            batch_size = longest_cycle - look_back + 1
        elif batch_size_case == 2:
            batch_size = 1
        else:
            warnings.warn("batch_size_case should be either 1 or 2")
            return
        num_features = x_train.shape[2]

        K.clear_session()

        if model_name in self.models:
            if not keep_training_model:
                warnings.warn("Model name already exists! Either change the name or set keep_training_model=True")
                return
            else:
                if initial_epoch == 0:
                    warnings.warn("initial_epoch=0")

                K.set_session(self.model_sessions[model_name])
                with K.get_session().graph.as_default():
                    model = self.models[model_name]
        else:
            emg_t_past = keras.Input(shape=(look_back, num_features), name='emg_t_past')
            x = layers.LSTM(num_nodes, activation=activation_func, dropout=dropout_rate,
                            recurrent_dropout=recurrent_dropout_rate, name='LSTM_layer')(emg_t_past)

            emg_t_now = keras.Input(shape=(num_features,), name='emg_t_now')
            current_timestep_dropout = layers.Dropout(mlp_dropout_rate)(emg_t_now)
            current_timestep = layers.Dense(num_nodes, activation=activation_func,
                                            name='mlp_current_timestep')(current_timestep_dropout)

            x = layers.merge.concatenate([x, current_timestep])
            prediction = layers.Dense(1, name='output_layer')(x)
            model = keras.Model(inputs=[emg_t_past, emg_t_now], outputs=prediction, name=model_name)

            print(model.summary())
            keras.utils.plot_model(model, to_file=self.cache_path + 'diagrams/' + model_name + '.png')

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
        if tensorboard:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=self.cache_path + 'tensorboard/' + model_name,
                                                         histogram_freq=5, write_grads=True))
        callbacks.append(keras.callbacks.ModelCheckpoint(self.cache_path + 'models/' + model_name + '_best.h5',
                                                         save_best_only=True))

        with K.get_session().graph.as_default():
            model.fit({'emg_t_past': x_train, 'emg_t_now': x_train[:, -1]}, y_train, epochs=epochs,
                      initial_epoch=initial_epoch, batch_size=batch_size, verbose=1,
                      validation_split=val_split, callbacks=callbacks)

            self.save_model(model, model_name)

            try:
                model_copy = keras.Model(inputs=[emg_t_past, emg_t_now], outputs=prediction, name=model_name)
                model_copy.compile(loss='mean_squared_error', optimizer=optimizer,
                                   metrics=['mean_absolute_error', 'mean_squared_error'])
                self.save_model(model_copy, model_name + '_copy')
            except NameError as error:
                warnings.warn("Could not save model copy. Return NameError: " + str(error))
                return

        self.load_models(model_name=model_name + '_best')

        # TODO: Reconsider the use of model_copy, it is the same as model_best except maybe the session graph... but...
        # the session graph could allow one to keep training from the point of best result rather than most recent.
        K.set_session(self.model_sessions[model_name + '_best'])
        with K.get_session().graph.as_default():
            trained_weights = self.models[model_name + '_best'].get_weights()

        K.clear_session()

        K.set_session(self.model_sessions[model_name + '_copy'])
        with K.get_session().graph.as_default():
            model_copy = self.models[model_name + '_copy']
            model_copy.set_weights(trained_weights)
            self.save_model(model_copy, model_name + '_copy')

    def train_lstm(self, initializer='glorot_uniform', optimizer='rmsprop', learning_rate=None, model_name='lstm',
                   num_nodes=32, epochs=100, val_split=0.2, early_stop_patience=None, dropout_rate=0.1, l1_reg=0.01,
                   l2_reg=0.01, recurrent_dropout_rate=0.5, look_back=3, activation_func='hard_sigmoid',
                   recurrent_activation_func='hard_sigmoid', tensorboard=True, keep_training_model=False,
                   initial_epoch=0, batch_size_case=1, use_test_as_val=False):
        """
        Trains a DNN based on a number of recurrent LSTM based layers and a single Dense output layer for regression.
        The regression parameter represents the joint moment prediction from the emg input data.
        :param str initializer: the weight initializer used for the keras LSTM layer (default 'glorot_uniform').
        See https://keras.io/initializers/ \n
        :param str optimizer: the optimizer used for the training, limited to 'rmsprop' or 'adam' (default 'rmsprop').
        See https://keras.io/optimizers/ \n
        :param float learning_rate: the learning rate used for the optimizer. If None then the internal default of the optimizer is used
        :param str model_name: the name of the model as it will be saved in cache and self.models (default 'lstm').
        :param int num_nodes: the number of hidden nodes in each unfolded LSTM layer (each timestep)
        :param int epochs: number of epochs to run (default: 100). The number needs to be larger than initial_epoch (see initial_epoch).
        :param float val_split: a percentage of dataset used as validation set, can be from 0 to 1 (default: 0.2)
        :param int early_stop_patience: number of epochs used for the keras.callbacks.EarlyStopping, monitoring the validation loss (default: None)
        :param float dropout_rate: float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs (default: 0.1).
        :param float recurrent_dropout_rate: float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state (default: 0.3).
        :param int look_back: number of timesteps to include for each sample input to the LSTM (default: 3).
        :param str activation_func: the activation function used for the network (default: 'relu').
        :param bool tensorboard: if True then tensorboard callback is used (default: True).
        :param bool keep_training_model: if model_name already exists in cache and keep_training_model is True then training of the model continues from the point when it was cached (default: False).
        :param int initial_epoch: the epoch number to start from (default: 0). If restarting training from cached model then this value should typically be the next epoch number from where the cached model stopped.
        :param int batch_size_case: either 1 or 2 (default: 1). Case 1 makes all the cycles of equal length by zero padding and uses batch size as the length of these cycles
        :return:
        """
        x_train, y_train, longest_cycle = gen_lstm_dataset(self.train_dataset.copy(), look_back,
                                                           batch_size_case=batch_size_case)
        if use_test_as_val:
            x_val, y_val, _ = gen_lstm_dataset(self.test_dataset.copy(), look_back, batch_size_case=batch_size_case)
            validation_set = (x_val, y_val)
        else:
            validation_set = None

        if batch_size_case == 1:
            batch_size = longest_cycle - look_back + 1
        elif batch_size_case == 2:
            batch_size = 1
        else:
            warnings.warn("batch_size_case should be either 1 or 2")
            return
        num_features = x_train.shape[2]

        K.clear_session()

        if model_name in self.models:
            if not keep_training_model:
                warnings.warn("Model name already exists! Either change the name or set keep_training_model=True")
                return
            else:
                if initial_epoch == 0:
                    warnings.warn("initial_epoch=0")

                K.set_session(self.model_sessions[model_name])
                with K.get_session().graph.as_default():
                    model = self.models[model_name]
        else:
            emg_input = keras.Input(shape=(look_back, num_features), name='emg_input')
            x = layers.LSTM(num_nodes, activation=activation_func, recurrent_activation=recurrent_activation_func,
                            dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate,
                            kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                            name='LSTM_layer')(emg_input)
            prediction = layers.Dense(1, name='output_layer')(x)
            model = keras.Model(inputs=emg_input, outputs=prediction, name=model_name)

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
        if tensorboard:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=self.cache_path + 'tensorboard/' + model_name,
                                                         histogram_freq=5, write_grads=True))
            if model_name not in self.models:
                update_combined_bat(model_name, self.cache_path + 'tensorboard/')
        callbacks.append(keras.callbacks.ModelCheckpoint(self.cache_path + 'models/' + model_name + '_best.h5',
                                                         save_best_only=True))

        with K.get_session().graph.as_default():
            model.fit(x_train, y_train, epochs=epochs, initial_epoch=initial_epoch, batch_size=batch_size, verbose=1,
                      validation_split=val_split, validation_data=validation_set, callbacks=callbacks)

            self.save_model(model, model_name)

            try:
                model_copy = keras.Model(inputs=emg_input, outputs=prediction, name=model_name)
                model_copy.compile(loss='mean_squared_error', optimizer=optimizer,
                                   metrics=['mean_absolute_error', 'mean_squared_error'])
                self.save_model(model_copy, model_name + '_copy')
            except NameError as error:
                warnings.warn("Could not save model copy. Return NameError: " + str(error))
                return

        self.load_models(model_name=model_name + '_best')

        # TODO: Reconsider the use of model_copy, it is the same as model_best except maybe the session graph... but...
        # the session graph could allow one to keep training from the point of best result rather than most recent.
        K.set_session(self.model_sessions[model_name + '_best'])
        with K.get_session().graph.as_default():
            trained_weights = self.models[model_name + '_best'].get_weights()

        K.clear_session()

        K.set_session(self.model_sessions[model_name + '_copy'])
        with K.get_session().graph.as_default():
            model_copy = self.models[model_name + '_copy']
            model_copy.set_weights(trained_weights)
            self.save_model(model_copy, model_name + '_copy')

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

    def plot_model_diagram(self, model_name):
        K.set_session(self.model_sessions[model_name])
        with K.get_session().graph.as_default():
            print(self.models[model_name].summary())
            keras.utils.plot_model(self.models[model_name], to_file=self.cache_path + 'diagrams/' + model_name + '.png')

    def evaluate_model(self, model_name, lstm=False, lstm_look_back=3, lstm_w_mlp=False):
        test_dataset = self.test_dataset.copy()
        moment_avg = test_dataset.groupby('Time')['Torque'].mean()

        if lstm:
            test_dataset, test_labels, _ = gen_lstm_dataset(test_dataset, lstm_look_back)

            if lstm_w_mlp:
                test_dataset = {'emg_t_past': test_dataset, 'emg_t_now': test_dataset[:, -1]}
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
            os.mkdir(self.cache_path + 'diagrams/')
            os.mkdir(self.cache_path + 'tensorboard/')
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
        if '_copy' in model_name:
            model_hist_name = model_name.replace('_copy', '')
        else:
            model_hist_name = model_name
        self.model_histories[model_name] = pd.read_csv(self.cache_path + 'history/' + model_hist_name + '.log',
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

    def plot_test(self, model_name, cycle_to_plot='random', lstm=False, lstm_look_back=3, lstm_w_mlp=False,
                  use_train_set=False, title=None, save_fig_as=None):
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
            # test_cycle = test_cycle[test_cycle.Time >= 0.0]
            test_time = test_cycle.pop('Time')

            if lstm:
                test_cycle, test_labels, _ = gen_lstm_dataset(test_cycle, lstm_look_back)
                test_time = test_time[lstm_look_back-1:]

                if lstm_w_mlp:
                    test_cycle = {'emg_t_past': test_cycle, 'emg_t_now': test_cycle[:, -1]}
            else:
                test_cycle.drop(columns=['Time', 'Trial'], inplace=True, errors='ignore')
                test_labels = test_cycle.pop('Torque')

            K.set_session(self.model_sessions[model_name])
            with K.get_session().graph.as_default():
                test_prediction = self.models[model_name].predict(test_cycle, batch_size=1)

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

        xvec = np.linspace(0, 100, num=len(labels))
        # xvec = np.arange(0, 101)
        # labels = saf.resample_signal(labels, len(xvec))
        # predictions = saf.resample_signal(predictions, len(xvec))

        if title is None:
            title = cycle_name
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()
        fig.add_subplot(ax1)

        fmt = '%.0f%%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax1.xaxis.set_major_formatter(xticks)

        ax1.set_title(title)
        # ax1.set_xlabel('gait cycle duration[s]')
        # ax1.set_ylabel('normalized joint moment')
        ax1.plot(xvec, labels, label='Test cycle')
        ax1.plot(xvec, predictions, label='Prediction')
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
            test_cycle, test_labels, _ = gen_lstm_dataset(test_cycle, lstm_look_back, batch_size_case=0)
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


def gen_lstm_dataset(df, look_back, batch_size_case=1):
    """
    Prepares the dataset for training the LSTM depending on how many past timesteps should be used (i.e. look_back)
    :param pandas.DataFrame df: A DataFrame dataset with all the emg data, with dimensions (num_timesteps, num_features)
    :param int look_back: Number of past timesteps to look at for prediction
    :param int batch_size_case: if batch_size_case is 1 it indicates that the cycles should be zero padded as the batch size will be the length of the cycle, in which case they all need to be of equal length (default: 1)
    :return: A numpy.array containing the LSTM prepared dataset with dimensions (num_timesteps, look_back, num_features)
    """
    df_copy = df.copy()

    inputs = list()
    labels = list()

    longest_cycle = df_copy.groupby('Trial')['Torque'].count().max(level=0).max()

    for _, trial_group in df_copy.groupby('Trial', sort=False):
        trial_group.drop(columns=['Time', 'Trial'], errors='ignore', inplace=True)
        if batch_size_case == 1:
            trial_group = zero_pad_dataset(trial_group, longest_cycle)
        group_labels = trial_group.pop('Torque')

        group_inputs = trial_group.to_numpy(dtype=float)
        group_labels = group_labels.to_numpy(dtype=float)

        for i in range(len(group_labels)):
            window_end = i + look_back
            if window_end > len(group_labels):
                break

            inputs.append(group_inputs[i:window_end, :])
            labels.append(group_labels[window_end-1])

    return np.array(inputs), np.array(labels), longest_cycle


def zero_pad_dataset(df, target_len, smooth_transition=True):
    return_df = df.copy()
    return_df.drop(['Time', 'Trial'], axis=1, inplace=True, errors='ignore')
    rows, columns = return_df.shape
    lines_to_add = target_len - rows
    data_to_pad = np.zeros((lines_to_add, columns))
    if smooth_transition:
        first_line = return_df.iloc[[0]].to_numpy()
        current_row = np.true_divide(first_line, 2)
        i = len(data_to_pad) - 1
        while np.any(np.abs(current_row) > 0.0001) and i >= 0:
            data_to_pad[i] = current_row
            current_row = np.true_divide(current_row, 2)
            i = i - 1

    return_df = pd.concat([pd.DataFrame(data=data_to_pad, columns=return_df.columns), return_df],
                          ignore_index=True)
    return return_df


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


def split_df_by_frac(df, frac=0.8, randomize_by_trial=True):
    """ Splits a pandas.DataFrame into two random sets based of a given fraction.

    :param df: The DataFrame
    :param frac: The fraction of the DataFrame assigned to the first_dataset, default: 0.8.
    Note if randomize_by_trial=True then this is the fraction of trials/cycles used, where number of samples within
    trial/cycle may vary
    :param randomize_by_trial: If True then the split is done trial/cycle wise such that all samples within a trial
    or cycle are put together into a dataset, default: True
    :return: the two datasets, first_dataset which is the bigger, and second_dataset the smaller
    """
    dataset = df.copy()
    if randomize_by_trial:
        fast_cycles = dataset[dataset['Trial'].str.contains('fast', case=False)]
        slow_cycles = dataset[dataset['Trial'].str.contains('slow', case=False)]
        normal_cycles = dataset.drop(index=fast_cycles.index)
        normal_cycles = normal_cycles.drop(index=slow_cycles.index)

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

        first_dataset = pd.concat([fast_dataset, slow_dataset, normal_dataset])

        trial_groups = [df for _, df in first_dataset.groupby('Trial')]
        np.random.shuffle(trial_groups)
        first_dataset = pd.concat(trial_groups)
    else:
        first_dataset = dataset.sample(frac=frac, random_state=0)

    second_dataset = dataset.drop(first_dataset.index)

    return first_dataset, second_dataset


def update_combined_bat(model_name, base_path):
    if os.path.exists(base_path + 'combined.bat'):
        string_to_add = ',' + model_name + ':"' + base_path + model_name + '"'
        with open(base_path + 'combined.bat', 'a') as f:
            f.write(string_to_add)
    else:
        string_to_add = 'tensorboard --logdir=' + model_name + ':"' + base_path + model_name + '"'
        with open(base_path + 'combined.bat', 'w') as f:
            f.write(string_to_add)


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

import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

import data_manager as dm
import filters as filt
import stat_analysis_functions as saf

import ann_methods as ann

# my_ann = ann.ANN()
# my_ann.train_lstm(model_name='LSTM_lookback5_valsplit02')
# train_tuple, validation_tuple = ann.generate_lstm_samples(my_ann.train_dataset, look_back=5)

run_from_cache = False

# ----------------------------- Load data ------------------------------------------- #
my_dm = dm.DataManager(r'C:\Users\Gunnar\Google Drive\Gunnar\MSThesis\Data')
my_dm.update_data_structure()   # Uncomment if the data_structure.json needs updating
if not run_from_cache:
    my_dm.load_emg_and_torque('Subject01', '20190405', reload=False)
    my_dm.load_emg_and_torque('Subject01', '20190603', reload=False)
    my_dm.load_emg_and_torque('Subject02', '20190406', reload=False)
    my_dm.load_emg_and_torque('Subject02', '20190608', reload=False)
    my_dm.load_emg_and_torque('Subject06', '20190429', reload=False)
    my_dm.load_emg_and_torque('Subject06', '20190509', reload=False)
my_dm.load_pandas()

# ----------------------------- Setup filtering parameters -------------------------- #
emg_lowcut = 10             # The lowpass cutoff frequency for EMG
emg_highcut = 100           # The highpass cutoff frequency for EMG
torque_lowcut = 100         # The lowpass cutoff frequency for the torque
torque_filt_order = 5       # The order of the butterworth lowpass filter
smoothing_window_ms = 50    # The window for moving average in milliseconds
fs = 1000                   # Sampling frequency for analog data
fs_downsample = 10          # This value is used to "decrease the sampling frequency" of the emg by taking the mean,
#                           # i.e. new sampling frequency = fs/fs_mean_window
freq = 100                  # Sampling frequency of model param, i.e. moment labels, and thus the frequency to use

frames_to_add = 20.0        # Includes this number of frames before the heel strike in the data sample, if they exist

# ----------------------------- Filter data if necessary ----------------------------- #
if not run_from_cache:
    Subject01_20190405_filtered_df = filt.filter_emg(my_dm.list_of_pandas['20190405 Subject01 emg_raw_data'],
                                                     emg_lowcut, emg_highcut, smoothing_window_ms, fs,
                                                     fs_downsample=fs_downsample)
    Subject01_20190603_filtered_df = filt.filter_emg(my_dm.list_of_pandas['20190603 Subject01 emg_raw_data'],
                                                     emg_lowcut, emg_highcut, smoothing_window_ms, fs,
                                                     fs_downsample=fs_downsample)
    Subject02_20190406_filtered_df = filt.filter_emg(my_dm.list_of_pandas['20190406 Subject02 emg_raw_data'],
                                                     emg_lowcut, emg_highcut, smoothing_window_ms, fs,
                                                     fs_downsample=fs_downsample)
    Subject02_20190608_filtered_df = filt.filter_emg(my_dm.list_of_pandas['20190608 Subject02 emg_raw_data'],
                                                     emg_lowcut, emg_highcut, smoothing_window_ms, fs,
                                                     fs_downsample=fs_downsample)
    Subject06_20190429_filtered_df = filt.filter_emg(my_dm.list_of_pandas['20190429 Subject06 emg_raw_data'],
                                                     emg_lowcut, emg_highcut, smoothing_window_ms, fs,
                                                     fs_downsample=fs_downsample)
    Subject06_20190509_filtered_df = filt.filter_emg(my_dm.list_of_pandas['20190509 Subject06 emg_raw_data'],
                                                     emg_lowcut, emg_highcut, smoothing_window_ms, fs,
                                                     fs_downsample=fs_downsample)


# ----------------------------- Merge EMG and Torque --------------------------------- #
def merge_prepare_and_cut(df, session_subject_id, cache_dataframe=True):
    df_copy = df.copy()
    # Append the torque
    df_copy = df_copy.merge(my_dm.list_of_pandas[session_subject_id + ' torque_raw_data'][['Time', 'MomentX', 'Trial']],
                            on=['Time', 'Trial'])
    # Change the 'MomentX' column name to 'Torque'
    df_copy.rename(columns={'MomentX':'Torque'}, inplace=True)
    # Save to cache
    if cache_dataframe:
        # Set name of df
        df_copy.name = session_subject_id + ' filtered_data'
        my_dm.update_pandas(df_copy)
    # # Cut data to gait cycles
    # my_dm.cut_data_to_cycles(df_copy, session_subject_id + ' filtered_n_cut eq')


if not run_from_cache:
    merge_prepare_and_cut(Subject01_20190405_filtered_df, '20190405 Subject01', cache_dataframe=True)
    merge_prepare_and_cut(Subject01_20190603_filtered_df, '20190603 Subject01', cache_dataframe=True)
    merge_prepare_and_cut(Subject02_20190406_filtered_df, '20190406 Subject02', cache_dataframe=True)
    merge_prepare_and_cut(Subject02_20190608_filtered_df, '20190608 Subject02', cache_dataframe=True)
    merge_prepare_and_cut(Subject06_20190429_filtered_df, '20190429 Subject06', cache_dataframe=True)
    merge_prepare_and_cut(Subject06_20190509_filtered_df, '20190509 Subject06', cache_dataframe=True)

# ----------------------------- Select muscle set ------------------------------------ #
def update_df_w_muscle_set(full_set, new_muscle_set, df):
    return_df = df.copy()

    muscle_cut_set = [x for x in full_set if x not in new_muscle_set and x in return_df.columns]
    return_df.drop(muscle_cut_set, axis=1, inplace=True)

    if not all(muscle in return_df.columns for muscle in new_muscle_set):
        warnings.warn('Dataset "' + df.name + '" did not contain all muscles in the muscle set:\n' +
                      ', '.join(new_muscle_set))
        return df

    return return_df[['Time'] + new_muscle_set + ['Torque', 'Trial']]


muscle_full_set = ['GlutMax', 'RectFem', 'VasMed', 'VasLat', 'BicFem', 'Semitend', 'TibAnt', 'Soleus', 'GasMed',
                   'GasLat', 'PerLong', 'PerBrev']
muscle_set1 = ['GlutMax', 'RectFem', 'VasMed', 'VasLat', 'BicFem', 'Semitend', 'TibAnt', 'Soleus', 'GasMed', 'GasLat']
muscle_set2 = ['GlutMax', 'RectFem', 'VasMed', 'VasLat', 'BicFem', 'Semitend', 'TibAnt', 'GasMed']
muscle_set3 = ['RectFem', 'VasMed', 'VasLat', 'BicFem', 'Semitend', 'TibAnt', 'Soleus', 'GasMed', 'GasLat']

# ----------------------------- Prepare/merge/select data ---------------------------- #
if not run_from_cache:
    subject01_20190405_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190405 Subject01 filtered_data'])
    subject01_20190603_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190603 Subject01 filtered_data'])
    subject02_20190406_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190406 Subject02 filtered_data'])
    subject02_20190608_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190608 Subject02 filtered_data'])
    subject06_20190429_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190429 Subject06 filtered_data'])
    subject06_20190509_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190509 Subject06 filtered_data'])

    subject01_20190603_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190603 Subject01 filtered_data'])
    subject02_20190608_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190608 Subject02 filtered_data'])

    subject01_all_set1 = pd.concat([subject01_20190405_set1, subject01_20190603_set1], ignore_index=True)
    subject02_all_set1 = pd.concat([subject02_20190406_set1, subject02_20190608_set1], ignore_index=True)
    subject06_all_set0 = pd.concat([subject06_20190429_set0, subject06_20190509_set0], ignore_index=True)
    subject06_all_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1, subject06_all_set0)

    subject01_all_set2 = update_df_w_muscle_set(muscle_set1, muscle_set2, subject01_all_set1)
    subject02_all_set2 = update_df_w_muscle_set(muscle_set1, muscle_set2, subject02_all_set1)
    subject06_all_set2 = update_df_w_muscle_set(muscle_set1, muscle_set2, subject06_all_set1)

    subject01_all_set3 = update_df_w_muscle_set(muscle_set1, muscle_set3, subject01_all_set1)
    subject02_all_set3 = update_df_w_muscle_set(muscle_set1, muscle_set3, subject02_all_set1)
    subject06_all_set3 = update_df_w_muscle_set(muscle_set1, muscle_set3, subject06_all_set1)

    # Cuts the data to cycles and caches the dataframes with the given names
    my_dm.cut_data_to_cycles(subject01_all_set1, 'subject01_all_set1', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject02_all_set1, 'subject02_all_set1', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject06_all_set1, 'subject06_all_set1', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject01_all_set2, 'subject01_all_set2', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject02_all_set2, 'subject02_all_set2', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject06_all_set2, 'subject06_all_set2', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject01_all_set3, 'subject01_all_set3', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject02_all_set3, 'subject02_all_set3', add_time=frames_to_add / freq)
    my_dm.cut_data_to_cycles(subject06_all_set3, 'subject06_all_set3', add_time=frames_to_add / freq)

    # Without frames_to_add to use for the analysis
    my_dm.cut_data_to_cycles(subject01_all_set1, 'subject01_all_set1_analysis')
    my_dm.cut_data_to_cycles(subject02_all_set1, 'subject02_all_set1_analysis')
    my_dm.cut_data_to_cycles(subject06_all_set1, 'subject06_all_set1_analysis')
    my_dm.cut_data_to_cycles(subject01_all_set2, 'subject01_all_set2_analysis')
    my_dm.cut_data_to_cycles(subject02_all_set2, 'subject02_all_set2_analysis')
    my_dm.cut_data_to_cycles(subject06_all_set2, 'subject06_all_set2_analysis')
    my_dm.cut_data_to_cycles(subject01_all_set3, 'subject01_all_set3_analysis')
    my_dm.cut_data_to_cycles(subject02_all_set3, 'subject02_all_set3_analysis')
    my_dm.cut_data_to_cycles(subject06_all_set3, 'subject06_all_set3_analysis')

# ----------------------------- Create or load ANN instance -------------------------- #
if run_from_cache:
    # subject01_20190405_set1_ann = ann.ANN(load_from_cache='201907081800_subject01_20190405_set1')
    # subject01_20190603_set0_ann = ann.ANN(load_from_cache='201907081800_subject01_20190603_set0')
    # subject02_20190406_set1_ann = ann.ANN(load_from_cache='201907081800_subject02_20190406_set1')
    # subject02_20190608_set0_ann = ann.ANN(load_from_cache='201907081800_subject02_20190608_set0')
    # subject06_20190429_set0_ann = ann.ANN(load_from_cache='201907081800_subject06_20190429_set0')
    # subject06_20190509_set0_ann = ann.ANN(load_from_cache='201907081800_subject06_20190509_set0')

    subject01_all_set1_ann = ann.ANN(load_from_cache='201907081800_subject01_all_set1')
    subject02_all_set1_ann = ann.ANN(load_from_cache='201907081800_subject02_all_set1')
    subject06_all_set0_ann = ann.ANN(load_from_cache='201907081800_subject06_all_set0')
    subject06_all_set1_ann = ann.ANN(load_from_cache='201907081800_subject06_all_set1')

    subject01_all_set2_ann = ann.ANN(load_from_cache='201907081800_subject01_all_set2')
    subject02_all_set2_ann = ann.ANN(load_from_cache='201907081800_subject02_all_set2')
    subject06_all_set2_ann = ann.ANN(load_from_cache='201907081800_subject06_all_set2')

    subject01_all_set3_ann = ann.ANN(load_from_cache='201907081800_subject01_all_set3')
    subject02_all_set3_ann = ann.ANN(load_from_cache='201907081800_subject02_all_set3')
    subject06_all_set3_ann = ann.ANN(load_from_cache='201907081800_subject06_all_set3')
else:
    # subject01_20190405_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_20190405_set1'],
    #                                       spec_cache_code='subject01_20190405_set1')
    # subject01_20190603_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_20190603_set0'],
    #                                       spec_cache_code='subject01_20190603_set0')
    # subject02_20190406_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_20190406_set1'],
    #                                       spec_cache_code='subject02_20190406_set1')
    # subject02_20190608_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_20190608_set0'],
    #                                       spec_cache_code='subject02_20190608_set0')
    # subject06_20190429_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_20190429_set0'],
    #                                       spec_cache_code='subject06_20190429_set0')
    # subject06_20190509_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_20190509_set0'],
    #                                       spec_cache_code='subject06_20190509_set0')

    subject01_all_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_all_set1'],
                                     spec_cache_code='subject01_all_set1')
    subject01_all_set2_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_all_set2'],
                                     spec_cache_code='subject01_all_set2')
    subject01_all_set3_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_all_set3'],
                                     spec_cache_code='subject01_all_set3')

    subject02_all_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_all_set1'],
                                     spec_cache_code='subject02_all_set1')
    subject02_all_set2_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_all_set2'],
                                     spec_cache_code='subject02_all_set2')
    subject02_all_set3_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_all_set3'],
                                     spec_cache_code='subject02_all_set3')

    subject06_all_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_set0'],
                                     spec_cache_code='subject06_all_set0')
    subject06_all_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_set1'],
                                     spec_cache_code='subject06_all_set1')
    subject06_all_set2_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_set2'],
                                     spec_cache_code='subject06_all_set2')
    subject06_all_set3_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_set3'],
                                     spec_cache_code='subject06_all_set3')

# ----------------------------- Analyze ANN datasets (e.g. train and test) ----------- #
plot_pre_analysis = False
if plot_pre_analysis:
    def analyzing_plots(df, title_spec=None, save_fig_as_spec=''):
        saf.plot_moment_avg(df, title=title_spec, y_axis_range=(-0.6, 0.8), plot_font_size=20,
                            save_fig_as=save_fig_as_spec + '_moment_avg')
        saf.plot_muscle_average(df, ['RectFem', 'VasMed', 'VasLat'], y_axis_range=(0, 0.7), plot_font_size=20,
                                save_fig_as=save_fig_as_spec + '_knee_extensors_emg_avg')
        saf.plot_muscle_average(df, ['BicFem', 'Semitend'], y_axis_range=(0, 0.7), plot_font_size=20,
                                save_fig_as=save_fig_as_spec + '_knee_flexors_emg_avg')
        saf.plot_muscle_average(df, ['GlutMax'], y_axis_range=(0, 0.7), plot_font_size=20,
                                save_fig_as=save_fig_as_spec + '_hip_extensor_emg_avg')
        saf.plot_muscle_average(df, ['TibAnt', 'Soleus', 'GasMed', 'GasLat'], y_axis_range=(0, 0.7), plot_font_size=20,
                                save_fig_as=save_fig_as_spec + '_ankle_flexors_emg_avg')

        saf.plot_muscle_correlations(df, include_torque=True, save_fig_as=save_fig_as_spec + '_corr_heatmap')


    saf.plot_grid_emg_average([subject01_all_set1_ann.normalized_dataset, subject02_all_set1_ann.normalized_dataset,
                               subject06_all_set1_ann.normalized_dataset],
                              ['Knee Ext.', 'Knee Flex.', 'Hip Ext.', 'Ankle Flex.'],
                              ['Subject01', 'Subject02', 'Subject06'], y_axis_range=(0, 0.7), plot_font_size=12,
                              save_fig_as='all_subject_grid_emg_avg')

    subject01_axs = analyzing_plots(subject01_all_set1_ann.normalized_dataset, save_fig_as_spec='subject01_all_set1')
    subject02_axs = analyzing_plots(subject02_all_set1_ann.normalized_dataset, save_fig_as_spec='subject02_all_set1')
    subject06_axs = analyzing_plots(subject06_all_set1_ann.normalized_dataset, save_fig_as_spec='subject06_all_set1')


    saf.plot_moment_avg(subject01_all_set1_ann.train_dataset, title='Subject01 Knee Joint Moments',
                        ylabel='normalized moments', y_axis_range=(-0.6, 0.8), save_fig_as='subject01_all_set1_moment_avg')
    saf.plot_moment_avg(subject02_all_set1_ann.train_dataset, title='Subject02 Knee Joint Moments',
                        ylabel='normalized moments', y_axis_range=(-0.6, 0.8), save_fig_as='subject02_all_set1_moment_avg')
    saf.plot_moment_avg(subject06_all_set0_ann.train_dataset, title='Subject06 Knee Joint Moments',
                        ylabel='normalized moments', y_axis_range=(-0.6, 0.8), save_fig_as='subject06_all_set0_moment_avg')

    saf.plot_cycle_time_quartile(subject01_all_set1_ann.train_dataset, title='Gait cycle duration quartile',
                                 save_fig_as='subject01_all_set1_cycle_duration_quartile')
    saf.plot_cycle_time_quartile(subject02_all_set1_ann.train_dataset, title='Gait cycle duration quartile',
                                 save_fig_as='subject02_all_set1_cycle_duration_quartile')
    saf.plot_cycle_time_quartile(subject06_all_set0_ann.train_dataset, title='Gait cycle duration quartile',
                                 save_fig_as='subject06_all_set1_cycle_duration_quartile')

    # saf.plot_muscle_average(subject01_20190603_set0_ann.train_dataset, ['RectFem', 'VasMed', 'VasLat'], y_lim=(0, 0.99),
    #                         plot_max_emg=True)

    # analyzing_plots(subject01_20190405_set1_ann.train_dataset, title_spec='Subject01 20190405')
    # analyzing_plots(subject01_20190603_set0_ann.train_dataset, title_spec='Subject01 20190603')
    # analyzing_plots(subject02_20190406_set1_ann.train_dataset, title_spec='Subject02 20190406')
    # analyzing_plots(subject02_20190608_set0_ann.train_dataset, title_spec='Subject02 20190608')
    # analyzing_plots(subject06_20190429_set0_ann.train_dataset, title_spec='Subject06 20190429')
    # analyzing_plots(subject06_20190509_set0_ann.train_dataset, title_spec='Subject06 20190509')

    # norm_values = my_ann.get_norm_values()
    # normalized_dataset = filt.real_time_normalize(my_ann.dataset, norm_values)
    # saf.plot_muscle_correlations(normalized_dataset)

# ----------------------------- Train ANNs on datasets ------------------------------- #
train_models = False
if train_models:
    subject01_all_set1_ann.train_lstm(model_name='LSTM_adam_64_02_01_05_4_relu_bc1', optimizer='adam', num_nodes=64,
                                      epochs=150, early_stop_patience=30, val_split=0.2, dropout_rate=0.1,
                                      recurrent_dropout_rate=0.5, look_back=4, keep_training_model=True,
                                      initial_epoch=0, batch_size_case=1)
    subject02_all_set1_ann.train_lstm(model_name='LSTM_adam_64_02_01_05_4_relu_bc1', optimizer='adam', num_nodes=64,
                                      epochs=150, early_stop_patience=30, val_split=0.2, dropout_rate=0.1,
                                      recurrent_dropout_rate=0.5, look_back=4, keep_training_model=True,
                                      initial_epoch=0, batch_size_case=1)
    subject06_all_set1_ann.train_lstm(model_name='LSTM_adam_64_02_01_05_4_relu_bc1', optimizer='adam', num_nodes=64,
                                      epochs=150, early_stop_patience=30, val_split=0.2, dropout_rate=0.1,
                                      recurrent_dropout_rate=0.5, look_back=4, keep_training_model=True,
                                      initial_epoch=0, batch_size_case=1)

# ----------------------------- Evaluate trained models ------------------------------ #
subject06_LSTM_adam0001_64_02_00_4_relu_prediction = subject06_all_set1_ann.get_test_prediction(
    'LSTM_adam0001_64_02_00_4_relu', lstm=True, lstm_look_back=4)
# subject01_all_set1_ann.plot_test('LSTM_adam_64_02_03_3_relu_best', lstm=True, lstm_look_back=3, cycle_to_plot='worst')
# subject01_all_set1_ann.plot_test('LSTM_adam_64_02_03_4_relu_best', lstm=True, lstm_look_back=4, cycle_to_plot='worst')
# subject01_all_set1_ann.plot_test('LSTM_adam_64_01_03_4_relu_best', lstm=True, lstm_look_back=4, cycle_to_plot='worst')
# subject01_all_set2_ann.plot_test('LSTM_adam_32_02_03_4_relu_best', lstm=True, lstm_look_back=4, cycle_to_plot='worst')

# subject01_all_set1_ann.evaluate_model('LSTM_adam_32_02_03_3_relu_best', lstm=True, lstm_look_back=3)
# subject01_all_set1_ann.evaluate_model('LSTM_adam_64_02_03_3_relu_best', lstm=True, lstm_look_back=3)
# subject01_all_set1_ann.evaluate_model('LSTM_adam_64_02_03_4_relu_best', lstm=True, lstm_look_back=4)
# subject01_all_set1_ann.evaluate_model('LSTM_adam_32_02_03_4_relu_best', lstm=True, lstm_look_back=4)
# subject01_all_set2_ann.evaluate_model('LSTM_adam_32_02_03_4_relu_best', lstm=True, lstm_look_back=4)

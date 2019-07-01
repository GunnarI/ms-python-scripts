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

run_from_cache = True

# ----------------------------- Load data ------------------------------------------- #
my_dm = dm.DataManager(r'C:\Users\Gunnar\Google Drive\Gunnar\MSThesis\Data')
# my_dm.update_data_structure()   # Uncomment if the data_structure.json needs updating
if not run_from_cache:
    my_dm.load_emg_and_torque('Subject01', '20190405', reload=False)
    my_dm.load_emg_and_torque('Subject01', '20190603', reload=True)
    my_dm.load_emg_and_torque('Subject02', '20190406', reload=False)
    my_dm.load_emg_and_torque('Subject02', '20190608', reload=False)
    my_dm.load_emg_and_torque('Subject06', '20190429', reload=True)
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


# ----------------------------- Merge EMG and Torque and cut to gait cycles ---------- #
def merge_prepare_and_cut(df, session_subject_id, cache_dataframe=False):
    df_copy = df.copy()
    # Append the torque
    df_copy = df_copy.merge(my_dm.list_of_pandas[session_subject_id + ' torque_raw_data'][['Time', 'MomentX', 'Trial']],
                            on=['Time', 'Trial'])
    # Get column names and change the 'MomentX' column name to 'Torque'
    column_names = list(df_copy)
    # column_names['MomentX' in column_names] = 'Torque'
    df_copy.columns = [column_name.replace('MomentX', 'Torque') for column_name in column_names]
    # Save to cache
    if cache_dataframe:
        my_dm.add_pandas(df_copy, session_subject_id + ' filtered_data')
    # Cut data to gait cycles
    my_dm.cut_data_to_cycles(df_copy, session_subject_id + ' filtered_n_cut')


if not run_from_cache:
    merge_prepare_and_cut(Subject01_20190405_filtered_df, '20190405 Subject01', cache_dataframe=False)
    merge_prepare_and_cut(Subject01_20190603_filtered_df, '20190603 Subject01', cache_dataframe=False)
    merge_prepare_and_cut(Subject02_20190406_filtered_df, '20190406 Subject02', cache_dataframe=False)
    merge_prepare_and_cut(Subject02_20190608_filtered_df, '20190608 Subject02', cache_dataframe=False)
    merge_prepare_and_cut(Subject06_20190429_filtered_df, '20190429 Subject06', cache_dataframe=False)
    merge_prepare_and_cut(Subject06_20190509_filtered_df, '20190509 Subject06', cache_dataframe=False)


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
muscle_set2 = ['RectFem', 'VasMed', 'VasLat', 'BicFem', 'Semitend', 'TibAnt', 'GasMed']

# subject02_all_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
#                                             my_dm.list_of_pandas['20190406 Subject02 filtered'])
# subject06_all_wo_ss_fullset = update_df_w_muscle_set(muscle_full_set, muscle_full_set, subject06_all)
# subject06_all_wo_ss_set2 = update_df_w_muscle_set(muscle_full_set, muscle_set2, subject06_all)

# ----------------------------- Prepare/merge/select data ---------------------------- #
if not run_from_cache:
    subject01_20190405_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190405 Subject01 filtered_n_cut'])
    subject01_20190603_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190603 Subject01 filtered_n_cut'])
    subject02_20190406_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190406 Subject02 filtered_n_cut'])
    subject02_20190608_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190608 Subject02 filtered_n_cut'])
    subject06_20190429_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190429 Subject06 filtered_n_cut'])
    subject06_20190509_set0 = update_df_w_muscle_set(muscle_full_set, muscle_full_set,
                                                     my_dm.list_of_pandas['20190509 Subject06 filtered_n_cut'])

    subject01_20190603_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190603 Subject01 filtered_n_cut'])
    subject01_all_set1 = pd.concat([subject01_20190405_set1, subject01_20190603_set1], ignore_index=True)
    subject02_20190608_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1,
                                                     my_dm.list_of_pandas['20190608 Subject02 filtered_n_cut'])
    subject02_all_set1 = pd.concat([subject02_20190406_set1, subject02_20190608_set1], ignore_index=True)
    subject06_all_set0 = pd.concat([subject06_20190429_set0, subject06_20190509_set0], ignore_index=True)
    subject06_all_set1 = update_df_w_muscle_set(muscle_full_set, muscle_set1, subject06_all_set0)

# subject06_wo_start_and_stop = my_dm.list_of_pandas['20190509 Subject06 filtered_n_cut'][
#     ~my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('WalkStop') &
#     ~my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('StartWalk')]

# slow_walk_20190429_06, normal_walk_20190429_06, fast_walk_20190429_06 = ann.split_trials_by_duration(
#     my_dm.list_of_pandas['20190429 Subject06 filtered'])
# slow_walk_20190509_06 = my_dm.list_of_pandas['20190509 Subject06 filtered'][
#     my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('WalkSlow')]
# normal_walk_20190509_06 = my_dm.list_of_pandas['20190509 Subject06 filtered'][
#     ~my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('WalkSlow') &
#     ~my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('WalkFast') &
#     ~my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('WalkStop') &
#     ~my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('StartWalk')]
# fast_walk_20190509_06 = my_dm.list_of_pandas['20190509 Subject06 filtered'][
#     my_dm.list_of_pandas['20190509 Subject06 filtered'].Trial.str.contains('WalkFast')]
# subject06_all = pd.concat([my_dm.list_of_pandas['20190429 Subject06 filtered'],
#                            my_dm.list_of_pandas['20190509 06 wo_start_stop']], ignore_index=True)


# ----------------------------- Cache dataframes for future use ---------------------- #
if not run_from_cache:
    subject01_20190405_set1.name = 'subject01_20190405_set1'
    subject01_20190603_set0.name = 'subject01_20190603_set0'
    subject02_20190406_set1.name = 'subject02_20190406_set1'
    subject02_20190608_set0.name = 'subject02_20190608_set0'
    subject06_20190429_set0.name = 'subject06_20190429_set0'
    subject06_20190509_set0.name = 'subject06_20190509_set0'

    subject01_all_set1.name = 'subject01_all_set1'
    subject02_all_set1.name = 'subject02_all_set1'
    subject06_all_set0.name = 'subject06_all_set0'
    subject06_all_set1.name = 'subject06_all_set1'

    my_dm.update_pandas(subject01_20190405_set1)
    my_dm.update_pandas(subject01_20190603_set0)
    my_dm.update_pandas(subject02_20190406_set1)
    my_dm.update_pandas(subject02_20190608_set0)
    my_dm.update_pandas(subject06_20190429_set0)
    my_dm.update_pandas(subject06_20190509_set0)

    my_dm.update_pandas(subject01_all_set1)
    my_dm.update_pandas(subject02_all_set1)
    my_dm.update_pandas(subject06_all_set0)
    my_dm.update_pandas(subject06_all_set1)

# ----------------------------- Create or load ANN instance -------------------------- #
if run_from_cache:
    subject01_20190405_set1_ann = ann.ANN(load_from_cache='201906262150_subject01_20190405_set1')
    subject01_20190603_set0_ann = ann.ANN(load_from_cache='201906262245_subject01_20190603_set0')
    subject02_20190406_set1_ann = ann.ANN(load_from_cache='201906262150_subject02_20190406_set1')
    subject02_20190608_set0_ann = ann.ANN(load_from_cache='201906262150_subject02_20190608_set0')
    subject06_20190429_set0_ann = ann.ANN(load_from_cache='201906262150_subject06_20190429_set0')
    subject06_20190509_set0_ann = ann.ANN(load_from_cache='201906262150_subject06_20190509_set0')

    subject01_all_set1_ann = ann.ANN(load_from_cache='201906272111_subject01_all_set1')
    subject02_all_set1_ann = ann.ANN(load_from_cache='201906272111_subject02_all_set1')
    subject06_all_set0_ann = ann.ANN(load_from_cache='201906272111_subject06_all_set0')
    subject06_all_set1_ann = ann.ANN(load_from_cache='201907012053_subject06_all_set1')
else:
    subject01_20190405_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_20190405_set1'],
                                          spec_cache_code='subject01_20190405_set1')
    subject01_20190603_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_20190603_set0'],
                                          spec_cache_code='subject01_20190603_set0')
    subject02_20190406_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_20190406_set1'],
                                          spec_cache_code='subject02_20190406_set1')
    subject02_20190608_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_20190608_set0'],
                                          spec_cache_code='subject02_20190608_set0')
    subject06_20190429_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_20190429_set0'],
                                          spec_cache_code='subject06_20190429_set0')
    subject06_20190509_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_20190509_set0'],
                                          spec_cache_code='subject06_20190509_set0')

    subject01_all_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject01_all_set1'],
                                     spec_cache_code='subject01_all_set1')
    subject02_all_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject02_all_set1'],
                                     spec_cache_code='subject02_all_set1')
    subject06_all_set0_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_set0'],
                                     spec_cache_code='subject06_all_set0')
    subject06_all_set1_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_set1'],
                                     spec_cache_code='subject06_all_set1')
# my_ann = ann.ANN(dataset=my_dm.list_of_pandas['subject06_all_wo_ss_fullset'], spec_cache_code='06_all_wo_ss_fullset')
# my_ann = ann.ANN()


# ----------------------------- Analyze ANN datasets (e.g. train and test) ----------- #
def analyzing_plots(df, title_spec=None, save_fig_as_spec=''):
    saf.plot_moment_avg(df, title=title_spec, ylabel='normalized KJM', y_axis_range=(-0.6, 0.8),
                        save_fig_as=save_fig_as_spec + '_moment_avg')
    saf.plot_muscle_correlations(df, include_torque=True, save_fig_as=save_fig_as_spec + '_corr_heatmap')
    saf.plot_muscle_average(df, ['RectFem', 'VasMed', 'VasLat'], y_axis_range=(0, 0.99),
                            save_fig_as=save_fig_as_spec + '_emg_avg')


analyzing_plots(subject01_all_set1_ann.train_dataset, save_fig_as_spec='subject01_all_set1')
analyzing_plots(subject02_all_set1_ann.train_dataset, save_fig_as_spec='subject02_all_set1')
analyzing_plots(subject06_all_set1_ann.train_dataset, save_fig_as_spec='subject06_all_set1')

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

analyzing_plots(subject01_20190405_set1_ann.train_dataset, title_spec='Subject01 20190405')
analyzing_plots(subject01_20190603_set0_ann.train_dataset, title_spec='Subject01 20190603')
analyzing_plots(subject02_20190406_set1_ann.train_dataset, title_spec='Subject02 20190406')
analyzing_plots(subject02_20190608_set0_ann.train_dataset, title_spec='Subject02 20190608')
analyzing_plots(subject06_20190429_set0_ann.train_dataset, title_spec='Subject06 20190429')
analyzing_plots(subject06_20190509_set0_ann.train_dataset, title_spec='Subject06 20190509')

# norm_values = my_ann.get_norm_values()
# normalized_dataset = filt.real_time_normalize(my_ann.dataset, norm_values)
# saf.plot_muscle_correlations(normalized_dataset)

# ----------------------------- Train ANNs on datasets ------------------------------- #
# my_ann.plot_test('RNNTF_layers_64_64_dropout0005', teach_force=True)
# my_ann.train_rnntf(optimizer='rmsprop', model_name='RNNTF_layers_64_64_dropout0005_2',
#                    layers_nodes=[(64, 'relu'), (64, 'relu')], early_stop_patience=10, val_split=0.2,
#                    dropout_rates=[0.0, 0.5])
# my_ann.train_ann(model_name='MLP_test', model_type='mlp', epochs=10, dropout_rates=[0.3, 0.3],
#                  num_nodes=[32, 32], activation=['relu', 'relu'], name=['Dense1_32', 'Dense2_32'])
# my_ann.train_lstm(model_name='LSTM_nodes_32')

# ----------------------------- Evaluate trained models ------------------------------ #


# saf.plot_moment_avg(norm_normal_walk_20190429_06,
#                     plot_min_med_max=True,
#                     ylabel='normalized knee moment')
# saf.plot_moment_avg(norm_normal_walk_20190509_06,
#                     plot_min_med_max=True,
#                     ylabel='normalized knee moment')
# saf.plot_moment_w_muscle(norm_normal_walk_20190429_06,
#                          muscle_list=['BicFem', 'Semitend'],
#                          title='20190429 Average activity - Knee flexors',
#                          moment_label='normalized knee moment')
# saf.plot_moment_w_muscle(norm_normal_walk_20190429_06,
#                          muscle_list=['RectFem', 'VasMed', 'VasLat'],
#                          title='20190429 Average activity - Knee extensors',
#                          moment_label='normalized knee moment')
# saf.plot_moment_w_muscle(norm_normal_walk_20190509_06,
#                          muscle_list=['BicFem', 'Semitend'],
#                          title='20190509 Average activity - Knee flexors',
#                          moment_label='normalized knee moment')
# saf.plot_moment_w_muscle(norm_normal_walk_20190509_06,
#                          muscle_list=['RectFem', 'VasMed', 'VasLat'],
#                          title='20190509 Average activity - Knee extensors',
#                          moment_label='normalized knee moment')
#
#
# saf.plot_moment_avg(norm_normal_walk06,
#                     plot_min_med_max=True,
#                     ylabel='normalized knee moment',
#                     save_fig_as='20190429_06_moments_normal_walk')
# saf.plot_moment_avg(norm_slow_walk06,
#                     plot_min_med_max=True,
#                     ylabel='normalized knee moment',
#                     save_fig_as='20190429_06_moments_slow_walk')
# saf.plot_moment_avg(norm_fast_walk06,
#                     plot_min_med_max=True,
#                     ylabel='normalized knee moment',
#                     save_fig_as='20190429_06_moments_fast_walk')
# saf.plot_moment_w_muscle(norm_normal_walk06,
#                          muscle_list=['BicFem', 'Semitend'],
#                          title='Average activity - Knee flexors',
#                          moment_label='normalized knee moment',
#                          save_fig_as='norm_emg_knee_flex')
# saf.plot_moment_w_muscle(norm_normal_walk06,
#                          muscle_list=['RectFem', 'VasMed', 'VasLat'],
#                          title='Average activity - Knee extensors',
#                          moment_label='normalized knee moment',
#                          save_fig_as='norm_emg_knee_ext')
# saf.plot_moment_w_muscle(norm_normal_walk06,
#                          muscle_list=['GlutMax'],
#                          title='Average activity - Hip extensor',
#                          moment_label='normalized knee moment',
#                          save_fig_as='norm_emg_hip_ext')
# saf.plot_moment_w_muscle(norm_normal_walk06,
#                          muscle_list=['TibAnt', 'Soleus', 'GasMed', 'GasLat'],
#                          title='Average activity - Ankle dorsi/plantar flexors',
#                          moment_label='normalized knee moment',
#                          save_fig_as='norm_emg_ankle_flex')
# saf.plot_moment_w_muscle(norm_normal_walk06,
#                          muscle_list=['PerLong', 'PerBrev'],
#                          title='Average activity - Foot eversion',
#                          moment_label='normalized knee moment',
#                          save_fig_as='norm_emg_peroneus')

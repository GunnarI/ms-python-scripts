import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn import preprocessing


def butter_bandpass(lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return b, a


def butter_lowpass(data, lowcut, fs, order=4):
    b, a = butter(order, lowcut, btype='lowpass', fs=fs)
    return lfilter(b, a, data)


def fourier_trans(data):
    return np.fft.fft(data)


def t_vec_after_ma(n, t):
    t = t[int(np.floor(n / 2)): int(len(t) - (np.ceil((n / 2) - 1)))]
    return t


def get_max_emg_array(max_emg_values, max_emg_trial, emg_data, trial_id, trial_types=''):
    """Compare all values from new data set to previous maximum values and replace if they are larger

    :param max_emg_values: an array of previous max values, one value for each muscle
    :param max_emg_trial: a list of the names of the trials where the max emg was found
    :param emg_data: the emg data from a new trial to compare to the max values from previous trials
    :param trial_id: the name of the trial corresponding to the new emg data
    :param trial_types: the trial types to be included for the max emg (e.g. 'walk' looks only at trials that
    contain 'walk' in their trial_id
    :return: an updated max_emg_values array and corresponding max_emg_trial list
    """
    for line in emg_data:
        for i in range(len(max_emg_values)):
            if line[i] > max_emg_values[i] and trial_types in trial_id.lower():
                max_emg_values[i] = line[i]
                max_emg_trial[i] = trial_id

    return max_emg_values, max_emg_trial


def min_max_normalize_data(df, secondary_df=None, norm_emg=True, norm_torque=False, normalizer='z_score',
                           feature_range=(0, 1)):
    """Normalizes EMG data and/or torque data from pandas.DataFrame using sklearn.preprocessing.MinMaxScaler.
    If secondary_df is given then the scaling of the "primary" is done using the min and max from the "secondary"
    DataFrame. This is useful when scaling the test dataset.

    :param pandas.DataFrame df: The dataframe to scale
    :param pandas.DataFrame secondary_df: Dataframe to use for the min/max values, if None then the primary df is used
    (default: None)
    :param boolean norm_emg: If True then emg values in df are normalized (default: True)
    :param boolean norm_torque: If True then torque values in df are normalized (default: False)
    :param str normalizer: can be 'z_score' or 'min_max_scaler'
    :param tuple feature_range: a tuple (min, max) indicating the range for MinMaxScaler of normalizer='min_max_scaler'
    :return: Dataframe with normalized values
    """
    return_df = df.copy()
    col = list(df)
    if 'Time' in col:
        time = return_df.pop('Time')
    if 'Torque' in col:
        torque = return_df.pop('Torque')
    if 'Trial' in col:
        trial = return_df.pop('Trial')

    emg = return_df.to_numpy()

    if secondary_df is None:
        if norm_emg:
            if normalizer == 'z_score':
                emg_scaler = preprocessing.StandardScaler().fit(emg)
            elif normalizer == 'min_max_scaler':
                emg_scaler = preprocessing.MinMaxScaler(feature_range=feature_range).fit(emg)
        if norm_torque:
            torque_scaler = preprocessing.MinMaxScaler(
                feature_range=((torque.min()/torque.max()), 1)).fit(torque.to_numpy().reshape(-1, 1))
    else:
        scaling_df = secondary_df.copy()

        if 'Time' in col:
            scaling_df.pop('Time')
        if 'Torque' in col:
            s_torque = scaling_df.pop('Torque')
        if 'Trial' in col:
            scaling_df.pop('Trial')

        s_emg = scaling_df.to_numpy()

        if norm_emg:
            if normalizer == 'z_score':
                emg_scaler = preprocessing.StandardScaler().fit(s_emg)
            elif normalizer == 'min_max_scaler':
                emg_scaler = preprocessing.MinMaxScaler(feature_range=feature_range).fit(s_emg)
        if norm_torque:
            torque_scaler = preprocessing.MinMaxScaler(
                feature_range=((s_torque.min()/s_torque.max()), 1)).fit(s_torque.to_numpy().reshape(-1, 1))

    if norm_emg:
        return_df = pd.DataFrame(emg_scaler.transform(emg), columns=return_df.columns, index=return_df.index)
    if norm_torque:
        torque = torque_scaler.transform(torque.to_numpy().reshape(-1, 1))

    if 'Time' in col:
        return_df.insert(0, 'Time', time)
    if 'Torque' in col:
        return_df['Torque'] = torque
    if 'Trial' in col:
        return_df['Trial'] = trial

    return return_df

# TODO: implement real-time filtering (if time)
# def filter_emg_rt(channels, low_pass, high_pass, window)


def filter_emg(emg_df, lowcut, highcut, window, fs, fs_downsample=0):
    df_copy = emg_df.copy()

    column_names = [col_name for col_name in df_copy.columns if col_name not in ['Trial', 'Torque']]
    list_of_emg_filtered = []
    trial_cycle_names = []
    for group, df in df_copy.groupby('Trial'):
        t = df.pop('Time')
        trial = df.pop('Trial')
        num_emg = df.shape[1]
        t_short = t_vec_after_ma(window, t.to_numpy())
        if fs_downsample > 0:
            t_short = np.ceil(t_short[0::fs_downsample] * (fs / fs_downsample)) * (fs_downsample / fs)
            t_short = np.around(t_short, decimals=int((fs/fs_downsample)/10))
        else:
            t_short = np.around(t_short, decimals=int(fs/10))
        filtered_emg = np.zeros(shape=(len(t_short), num_emg + 1))
        filtered_emg[:, 0] = t_short

        for i, column in enumerate(df.to_numpy(dtype=float).T):
            column = noise_filter(column, lowcut, highcut, fs)
            column = demodulation(column)
            column, t = smoothing(column, t, window, fs)
            column = relinearization(column)
            if fs_downsample > 0:
                column = column[0::fs_downsample]
            filtered_emg[:, i + 1] = column

        list_of_emg_filtered.append(filtered_emg)
        trial_cycle_names.append([str(group)]*len(filtered_emg))

    emg_filtered_df = pd.DataFrame(data=np.concatenate(list_of_emg_filtered), columns=column_names, dtype=float)
    emg_filtered_df = emg_filtered_df.assign(Trial=np.concatenate(trial_cycle_names))
    old_name_split = emg_df.name.split()
    emg_filtered_df.name = old_name_split[0] + old_name_split[1] + ' filtered'

    return emg_filtered_df


def noise_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)     # lfilter is a causal forward-in-time filtering and can't be zero-phase
    return y


def rt_noise_filter(data, lowcut, highcut, fs, zi=None, order=5):
    """Example of use:
        result = zeros(data.size)
        zi = None
        for i, x in enumerate(data):
            result[i], zi = rt_noise_filter(x, 30, 200, 1000, zi)

    :param data:
    :param lowcut:
    :param highcut:
    :param fs:
    :param zi:
    :param order:
    :return:
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if zi is None:
        zi = lfilter_zi(b, a)
    y, zo = lfilter(b, a, [data], zi=zi)
    return y, zo

# def whitening


def demodulation(data):
    return np.square(data)


def smoothing(data, t, window, fs):
    n = int(fs * (window / 1000.0))
    data_sum = np.cumsum(np.insert(data, 0, 0))
    smoothed_data = (data_sum[n:] - data_sum[:-n]) / float(n)
    return smoothed_data, t_vec_after_ma(n, t)


def relinearization(data):
    return np.sqrt(data)

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


def save_np_dict_to_txt(dict_to_save, base_dir, data_fmt, headers=None):
    for key in dict_to_save:
        if headers:
            np.savetxt(base_dir + key + '.txt', dict_to_save[key], fmt=data_fmt,
                       header=' '.join(emg_id for emg_id in headers[key]), comments='')
        else:
            np.savetxt(base_dir + key + '.txt', dict_to_save[key], fmt=data_fmt)


def min_max_normalize_data(df, secondary_df=None, norm_emg=True, norm_torque=False):
    """Normalizes EMG data and/or torque data from pandas.DataFrame using sklearn.preprocessing.MinMaxScaler.
    If secondary_df is given then the scaling of the "primary" is done using the min and max from the "secondary"
    DataFrame. This is useful when scaling the test dataset.

    :param df: The dataframe to scale
    :param secondary_df: Dataframe to use for the min/max values, if None then the primary df is used (default: None)
    :param norm_emg: If True then emg values in df are normalized (default: True)
    :param norm_torque: If True then torque values in df are normalized (default: False)
    :return: Dataframe with normalized values
    """
    return_df = df.copy()
    col = return_df.columns
    if 'Time' in col:
        time = return_df.pop('Time')
    if 'Torque' in col:
        torque = return_df.pop('Torque')
    if 'Trial' in col:
        trial = return_df.pop('Trial')

    emg = return_df.values

    if secondary_df is None:
        if norm_emg:
            emg_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(emg)
        if norm_torque:
            min_torque = torque.min()
            max_torque = torque.max()
            if np.abs(min_torque) < max_torque:
                torque_scaler = preprocessing.MinMaxScaler(
                    feature_range=((min_torque/max_torque), 1)).fit(torque.values.reshape(-1, 1))
            elif min_torque < 0:
                torque_scaler = preprocessing.MinMaxScaler(
                    feature_range=(-1, (max_torque / np.abs(min_torque)))).fit(torque.values.reshape(-1, 1))
            else:
                torque_scaler = preprocessing.MinMaxScaler(
                    feature_range=(min_torque, 1)).fit(torque.values.reshape(-1, 1))

    else:
        scaling_df = secondary_df.copy()

        if 'Time' in col:
            scaling_df.pop('Time')
        if 'Torque' in col:
            s_torque = scaling_df.pop('Torque')
        if 'Trial' in col:
            scaling_df.pop('Trial')

        s_emg = scaling_df.values

        if norm_emg:
            emg_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(s_emg)
        if norm_torque:
            min_torque = s_torque.min()
            max_torque = s_torque.max()
            if np.abs(min_torque) < max_torque:
                torque_scaler = preprocessing.MinMaxScaler(
                    feature_range=((min_torque / max_torque), 1)).fit(s_torque.values.reshape(-1, 1))
            elif min_torque < 0:
                torque_scaler = preprocessing.MinMaxScaler(
                    feature_range=(-1, (max_torque / np.abs(min_torque)))).fit(s_torque.values.reshape(-1, 1))
            else:
                torque_scaler = preprocessing.MinMaxScaler(
                    feature_range=(min_torque, 1)).fit(s_torque.values.reshape(-1, 1))

            # torque_scaler = preprocessing.MinMaxScaler(
            #     feature_range=((s_torque.min()/s_torque.max()), 1)).fit(s_torque.values.reshape(-1, 1))

    if norm_emg:
        return_df = pd.DataFrame(emg_scaler.transform(emg), columns=return_df.columns, index=return_df.index)
    if norm_torque:
        torque = torque_scaler.transform(torque.values.reshape(-1, 1))

    if 'Time' in col:
        return_df.insert(0, 'Time', time)
    if 'Torque' in col:
        return_df['Torque'] = torque
    if 'Trial' in col:
        return_df['Trial'] = trial

    return return_df


# TODO: Test functionality!
def real_time_normalize(data, norm_values):
    if isinstance(data, pd.DataFrame) and isinstance(norm_values, dict):
        return_df = data.copy()
        for column in return_df:
            if column not in ['Time', 'Torque', 'Trial']:
                return_df[column] = normalize(return_df[column], norm_values['norm_min_values'][column],
                                         norm_values['norm_max_values'][column])
            elif column == 'Torque':
                min_torque = norm_values['norm_min_values'][column]
                max_torque = norm_values['norm_max_values'][column]

                if np.abs(min_torque) < max_torque:
                    scale_range = ((min_torque / max_torque), 1)
                elif min_torque < 0:
                    scale_range = (-1, (max_torque / np.abs(min_torque)))
                else:
                    scale_range = (min_torque, 1)

                return_df[column] = normalize(return_df[column], min_torque, max_torque, scale_range=scale_range)

        return return_df


def normalize(feature, min_value, max_value, scale_range=(0, 1)):
    feature_std = (feature - min_value) / (max_value - min_value)
    feature_scaled = feature_std * (scale_range[1] - scale_range[0]) + scale_range[0]

    return feature_scaled

# TODO: implement real-time filtering (if time)
# def filter_emg_rt(channels, low_pass, high_pass, window)


def filter_torque(torque_data_dict, filt_order, lowcut, fs, axis_of_focus=0, cut_time=False, lp_filter=False,
                  subject_id=None):
    torque_filt_data_dict = {}
    headers = {}
    b, a = butter(filt_order, lowcut, btype='lowpass', fs=fs)
    for key in torque_data_dict:
        if subject_id is None or subject_id in key:
            filtered_torque = np.zeros(shape=(len(torque_data_dict[key]["data"][:, 0]), 2))
            filtered_torque[:, 0] = torque_data_dict[key]["data"][:, 0]
            if lp_filter:
                filtered_torque[:, 1] = lfilter(b, a, torque_data_dict[key]["data"][:, axis_of_focus + 1])
            else:
                filtered_torque[:, 1] = torque_data_dict[key]["data"][:, axis_of_focus + 1]

            if cut_time and 'walk' in key.lower():
                t1 = torque_data_dict[key]["t1"]
                t2 = torque_data_dict[key]["t2"]
                filtered_torque = filtered_torque[(filtered_torque[:, 0] >= t1) & (filtered_torque[:, 0] <= t2), :]

            torque_filt_data_dict[key + ' filtered'] = filtered_torque
            headers[key + ' filtered'] = ['Time', 'Torque']

    # Save the filtered torque signals
    save_np_dict_to_txt(torque_filt_data_dict, './data/labels/', data_fmt='%f', headers=headers)


def filter_emg(emg_data_dict, lowcut, highcut, window, fs, cut_time=False, fs_downsample=0, subject_id=None):
    emg_filt_data_dict = {}
    emg_headers = {}
    for key in emg_data_dict:
        if subject_id is None or subject_id in key:
            num_emg = len(emg_data_dict[key]["data"][0, :]) - 1
            t = emg_data_dict[key]["data"][:, 0]
            t_short = t_vec_after_ma(window, t)
            if fs_downsample > 0:
                t_short = np.ceil(t_short[fs_downsample-1::fs_downsample]*(fs/fs_downsample))*(fs_downsample/fs)
            filtered_emg = np.zeros(shape=(len(t_short), num_emg + 1))
            filtered_emg[:, 0] = t_short
            for i, column in enumerate(emg_data_dict[key]["data"][:, 1:].T):
                column = noise_filter(column, lowcut, highcut, fs)
                column = demodulation(column)
                column, t = smoothing(column, t, window, fs)
                column = relinearization(column)
                if fs_downsample > 0:
                    column = column[0::fs_downsample]
                filtered_emg[:, i + 1] = column

            emg_filt_data_dict[key + ' filtered'] = filtered_emg

            emg_headers[key + ' filtered'] = emg_data_dict[key]["headers"]

            if cut_time and 'walk' in key.lower():
                t1 = emg_data_dict[key]["t1"]
                t2 = emg_data_dict[key]["t2"]
                emg_filt_data_dict[key + ' filtered'] = filtered_emg[
                                                        (filtered_emg[:, 0] >= t1) & (filtered_emg[:, 0] <= t2), :]

    # Save the filtered and normalized emg signals
    save_np_dict_to_txt(emg_filt_data_dict, './data/', data_fmt='%f', headers=emg_headers)


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

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


def get_max_emg_array(max_emg_values, max_emg_exercise, emg_data, exercise_id, exercise_types=''):
    """Compare all values from new data set to previous maximum values and replace if they are larger

    :param max_emg_values: an array of previous max values, one value for each muscle
    :param max_emg_exercise: a list of the names of the exercises where the max emg was found
    :param emg_data: the emg data from a new exercise to compare to the max values from previous exercises
    :param exercise_id: the name of the exercise corresponding to the new emg data
    :param exercise_types: the exercise types to be included for the max emg (e.g. 'walk' looks only at exercises that
    contain 'walk' in their exercise_id
    :return: an updated max_emg_values array and corresponding max_emg_exercise list
    """
    for line in emg_data:
        for i in range(len(max_emg_values)):
            if line[i] > max_emg_values[i] and exercise_types in exercise_id.lower():
                max_emg_values[i] = line[i]
                max_emg_exercise[i] = exercise_id

    return max_emg_values, max_emg_exercise


def save_np_dict_to_txt(dict_to_save, base_dir, data_fmt, headers=None):
    for key in dict_to_save:
        if headers:
            np.savetxt(base_dir + key + '.txt', dict_to_save[key], fmt=data_fmt,
                       header=' '.join(emg_id for emg_id in headers[key]), comments='')
        else:
            np.savetxt(base_dir + key + '.txt', dict_to_save[key], fmt=data_fmt)


def normalize_emg(emg_data_dict, max_emg_dict):
    for key in emg_data_dict:
        ids = key.split()

        emg_data_dict[key][:, 1:] = np.divide(
            emg_data_dict[key][:, 1:],
            max_emg_dict[ids[0] + ' ' + ids[1] + ' MaxEMG'])
    return emg_data_dict


def min_max_normalize_data(df, norm_emg=True, norm_torque=False):
    return_df = df.copy()
    torque = return_df.pop('Torque')
    emg = return_df.values
    # for i in range(x[:,1:1+num_features], 1):
    if norm_emg:
        return_df = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(emg),
                                 columns=return_df.columns)
    if norm_torque:
        torque = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    return_df['Torque'] = torque
    return return_df

# TODO: implement real-time filtering (if time)
# def filter_emg_rt(channels, low_pass, high_pass, window)


def filter_torque(torque_data_dict, filt_order, lowcut, fs, axis_of_focus=0, cut_time=False, lp_filter=False):
    torque_filt_data_dict = {}
    headers = {}
    b, a = butter(filt_order, lowcut, btype='lowpass', fs=fs)
    for key in torque_data_dict:
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


def filter_emg(emg_data_dict, lowcut, highcut, window, fs, cut_time=False, fs_mean_window=0):
    emg_filt_data_dict = {}
    emg_headers = {}
    # max_emg_values_dict = {}
    # max_emg_exercises_dict = {}
    for key in emg_data_dict:
        ids = key.split()
        num_emg = len(emg_data_dict[key]["data"][0, :]) - 1
        t = emg_data_dict[key]["data"][:, 0]
        t_short = t_vec_after_ma(window, t)
        if fs_mean_window > 0:
            t_short = t_short[fs_mean_window-1::fs_mean_window]
        filtered_emg = np.zeros(shape=(len(t_short), num_emg + 1))
        filtered_emg[:, 0] = t_short
        for i, column in enumerate(emg_data_dict[key]["data"][:, 1:].T):
            column = noise_filter(column, lowcut, highcut, fs)
            column = demodulation(column)
            column, t = smoothing(column, t, window, fs)
            column = relinearization(column)
            if fs_mean_window > 0:
                tail = - (column.size % fs_mean_window) if column.size % fs_mean_window > 0 else column.size
                column = column[:tail].reshape(-1, fs_mean_window).mean(axis=-1)
            filtered_emg[:, i + 1] = column

        emg_filt_data_dict[key + ' filtered'] = filtered_emg

        emg_headers[key + ' filtered'] = emg_data_dict[key]["headers"]

        # max_emg_key = ids[0] + ' ' + ids[1] + ' MaxEMG'

        # If dict key for specific trial does not exist then create key and initialize array/list values
        # if max_emg_key not in max_emg_values_dict:
        #     max_emg_values_dict[max_emg_key] = np.zeros(num_emg)
        #     max_emg_exercises_dict[max_emg_key] = [''] * num_emg

        # Update max emg values from relevant trial with values from this loops exercise
        # max_emg_values_dict[max_emg_key], max_emg_exercises_dict[max_emg_key] = get_max_emg_array(
        #     max_emg_values_dict[max_emg_key],
        #     max_emg_exercises_dict[max_emg_key],
        #     filtered_emg[:, 1:], ids[2], 'walk')

        if cut_time and 'walk' in key.lower():
            t1 = emg_data_dict[key]["t1"]
            t2 = emg_data_dict[key]["t2"]
            emg_filt_data_dict[key + ' filtered'] = filtered_emg[
                                                    (filtered_emg[:, 0] >= t1) & (filtered_emg[:, 0] <= t2), :]

    # Save the max emg values to a txt file
    # save_np_dict_to_txt(max_emg_values_dict, './data/', data_fmt='%f', headers=max_emg_exercises_dict)

    # Normalize the emg data
    # emg_filt_data_dict = normalize_emg(emg_filt_data_dict, max_emg_values_dict)

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

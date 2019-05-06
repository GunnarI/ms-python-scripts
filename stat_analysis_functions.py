import matplotlib.pyplot as plt
import numpy as np

import filters as filt


def get_min_max_cycles(filt_data_dict, subject):
    min_cycle = ''
    min_cycle_len = 1000
    max_cycle = ''
    max_cycle_len = 0
    for key in filt_data_dict:
        if 'concat' not in key and subject in key:
            if len(filt_data_dict[key].index) < min_cycle_len:
                min_cycle_len = len(filt_data_dict[key].index)
                min_cycle = key

            if len(filt_data_dict[key].index) > max_cycle_len:
                max_cycle_len = len(filt_data_dict[key].index)
                max_cycle = key

    return (min_cycle, min_cycle_len), (max_cycle, max_cycle_len)


def plot_moment_avg(filt_data_dict, subject, plot_min_and_max=False, plot_time_boxplot=False):
    min_cycle, max_cycle = get_min_max_cycles(filt_data_dict, subject)

    plt.figure()
    if plot_time_boxplot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        ax1 = plt.subplot()
    for key in filt_data_dict:
        if 'concat' in key and subject in key:
            moment_avg = filt_data_dict[key].groupby('Time')['Torque'].mean()
            moment_std = filt_data_dict[key].groupby('Time')['Torque'].std()
            time_vec = np.arange(0, max_cycle[1]) / 100

            cut_from_moment_len = -(len(moment_avg) - len(time_vec))
            cut_from_std_len = -(len(moment_avg) - len(time_vec))
            std_range = (moment_avg - moment_std, moment_avg + moment_std)
            std_range[0][np.isnan(std_range[0].values)] = 0
            std_range[1][np.isnan(std_range[1].values)] = 0

            ax1.fill_between(time_vec, std_range[0], std_range[1], alpha=0.2)
            ax1.plot(time_vec, moment_avg, label='Average')
            ax1.set_title(subject + ' - Knee joint moments')
            ax1.set_xlabel('gait cycle duration [s]')
            ax1.set_ylabel('joint moment [N.mm/kg]')

            if plot_time_boxplot:
                time_intervals = filt_data_dict[key].groupby('Exercise')['Time'].agg(np.ptp)
                ax2.boxplot(time_intervals, vert=False)

            break

    if plot_min_and_max:
        ax1.plot(np.arange(0, min_cycle[1]) / 100, filt_data_dict[min_cycle[0]]['Torque'],
                 label='Fastest cycle')
        ax1.plot(np.arange(0, max_cycle[1]) / 100, filt_data_dict[max_cycle[0]]['Torque'],
                 label='Slowest cycle')
    ax1.legend()


def plot_muscle_average(filt_data_dict, subject, muscle_list=None, plot_min_and_max=False,
                        title='EMG Muscle Activity'):
    min_cycle, max_cycle = get_min_max_cycles(filt_data_dict, subject)

    for key in filt_data_dict:
        if 'concat' in key and subject in key:
            df = filt.min_max_normalize_data(filt_data_dict[key])

            if not muscle_list:
                muscle_list = list(filt_data_dict[key])
                muscle_list.remove('Time')
                muscle_list.remove('Torque')
                muscle_list.remove('Exercise')

            num_plots = len(muscle_list)
            fig, axs = plt.subplots(num_plots, 1, sharex=True)
            fig.subplots_adjust(hspace=0.000)
            fig.suptitle(title)
            fig.text(0.04, 0.5, 'Normalized amplitdue [V/V]', va='center', rotation='vertical')
            for i, muscle in enumerate(muscle_list):
                emg_avg = df.groupby('Time')[muscle].mean()
                emg_std = df.groupby('Time')[muscle].std()
                time_vec = np.arange(0, max_cycle[1]) / 100

                std_range = (emg_avg - emg_std, emg_avg + emg_std)

                ax1 = plt.subplot(num_plots, 1, i+1)
                ax1.fill_between(time_vec, std_range[0], std_range[1], alpha=0.2)
                ax1.plot(time_vec, emg_avg)
                ax1.set_xlabel('gait cycle duration [s]')
                #ax1.set_ylabel('normalized amplitude [V/V]')

                if plot_min_and_max:
                    ax1.plot(np.arange(0, min_cycle[1]) / 100, filt_data_dict[min_cycle[0]][muscle],
                             label='Fastest cycle')
                    ax1.plot(np.arange(0, max_cycle[1]) / 100, filt_data_dict[max_cycle[0]][muscle],
                             label='Slowest cycle')

            break

    fig.show()


def plot_cycle_time_quartile(df):
    time_intervals = df.groupby('Exercise')['Time'].agg(np.ptp)
    plt.figure()
    plot_result = plt.boxplot(time_intervals)

    return plot_result

def plot_emg_torque(df, emg_to_plot):
    t = df['Time']

    for muscle in emg_to_plot:
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized EMG', color=color)
        ax1.plot(t, df[muscle])
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:red'
        ax2.set_ylabel('Torque', color=color)
        ax2.plot(t, df['Torque'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(muscle)
        plt.show()


def get_muscle_std(df, muscle_list=None):
    if not muscle_list:
        muscle_list = list(df)
        muscle_list.remove('Time')
        muscle_list.remove('Torque')
        muscle_list.remove('Exercise')

    muscle_list_std = np.zeros(len(muscle_list))
    for i, muscle in enumerate(muscle_list):
        muscle_list_std[i] = np.mean(df.groupby('Time')[muscle].std())

    return muscle_list_std

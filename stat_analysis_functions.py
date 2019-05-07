import matplotlib.pyplot as plt
import numpy as np


def get_min_med_max_cycles(df):
    grouped_by_cycle = df.groupby('Exercise').size()
    grouped_by_cycle = grouped_by_cycle.sort_values()

    min_cycle_len = grouped_by_cycle.iloc[0]
    min_cycle = grouped_by_cycle.index[0]

    median_index = np.floor(len(grouped_by_cycle.index)/2).astype('int')
    median_cycle_len = grouped_by_cycle.iloc[median_index]
    median_cycle = grouped_by_cycle.index[median_index]

    max_cycle_len = grouped_by_cycle.iloc[-1]
    max_cycle = grouped_by_cycle.index[-1]

    return (min_cycle, min_cycle_len), (median_cycle, median_cycle_len), (max_cycle, max_cycle_len)


def plot_moment_avg(df, plot_min_med_max=False, plot_time_boxplot=False):
    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df)

    plt.figure()
    if plot_time_boxplot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        ax1 = plt.subplot()

    moment_avg = df.groupby('Time')['Torque'].mean()
    moment_std = df.groupby('Time')['Torque'].std()
    time_vec = np.arange(0, max_cycle[1]) / 100

    std_range = (moment_avg - moment_std, moment_avg + moment_std)
    std_range[0][np.isnan(std_range[0].values)] = 0
    std_range[1][np.isnan(std_range[1].values)] = 0

    ax1.fill_between(time_vec, std_range[0], std_range[1], alpha=0.2)
    ax1.plot(time_vec, moment_avg, label='Average')
    ax1.set_title('Knee joint moments')
    ax1.set_xlabel('gait cycle duration [s]')
    ax1.set_ylabel('joint moment [N.mm/kg]')

    if plot_time_boxplot:
        time_intervals = df.groupby('Exercise')['Time'].agg(np.ptp)
        ax2.boxplot(time_intervals, vert=False)

    if plot_min_med_max:
        ax1.plot(df.loc[df['Exercise'] == min_cycle[0], 'Time'], df.loc[df['Exercise'] == min_cycle[0], 'Torque'],
                 label='Fastest cycle')
        ax1.plot(df.loc[df['Exercise'] == max_cycle[0], 'Time'], df.loc[df['Exercise'] == max_cycle[0], 'Torque'],
                 label='Slowest cycle')
        ax1.plot(df.loc[df['Exercise'] == med_cycle[0], 'Time'], df.loc[df['Exercise'] == med_cycle[0], 'Torque'],
                 label='Median cycle')
    ax1.legend()


def plot_muscle_average(df, muscle_list=None, plot_min_and_max=False,
                        title='EMG Muscle Activity'):
    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df)

    if not muscle_list:
        muscle_list = list(df)
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

        ax1 = plt.subplot(num_plots, 1, i + 1)
        ax1.fill_between(time_vec, std_range[0], std_range[1], alpha=0.2)
        ax1.plot(time_vec, emg_avg, label='Average')
        ax1.set_xlabel('gait cycle duration [s]')
        # ax1.set_ylabel('normalized amplitude [V/V]')

        if plot_min_and_max:
            ax1.plot(df.loc[df['Exercise'] == min_cycle[0], 'Time'], df.loc[df['Exercise'] == min_cycle[0], muscle],
                     label='Fastest cycle')
            ax1.plot(df.loc[df['Exercise'] == max_cycle[0], 'Time'], df.loc[df['Exercise'] == max_cycle[0], muscle],
                     label='Slowest cycle')
            ax1.plot(df.loc[df['Exercise'] == med_cycle[0], 'Time'], df.loc[df['Exercise'] == med_cycle[0], muscle],
                     label='Median cycle')
            ax1.legend()

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

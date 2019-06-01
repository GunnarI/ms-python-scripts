import matplotlib.pyplot as plt
import numpy as np


def get_min_med_max_cycles(df):
    grouped_by_cycle = df.groupby('Trial').size()
    grouped_by_cycle = grouped_by_cycle.sort_values()

    min_cycle_len = grouped_by_cycle.iloc[0]
    min_cycle = grouped_by_cycle.index[0]

    median_index = np.floor(len(grouped_by_cycle.index)/2).astype('int')
    median_cycle_len = grouped_by_cycle.iloc[median_index]
    median_cycle = grouped_by_cycle.index[median_index]

    max_cycle_len = grouped_by_cycle.iloc[-1]
    max_cycle = grouped_by_cycle.index[-1]

    return (min_cycle, min_cycle_len), (median_cycle, median_cycle_len), (max_cycle, max_cycle_len)


def get_muscle_std(df, muscle_list=None):
    if not muscle_list:
        muscle_list = list(df)
        muscle_list.remove('Time')
        muscle_list.remove('Torque')
        muscle_list.remove('Trial')

    muscle_list_std = np.zeros(len(muscle_list))
    for i, muscle in enumerate(muscle_list):
        muscle_list_std[i] = np.mean(df.groupby('Time')[muscle].std())

    return muscle_list_std


def plot_moment_avg(df, plot_min_med_max=False, title='Knee joint moments', ylabel=r'joint moment $[\frac{N.mm}{kg}]$',
                    save_fig_as=None):
    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df)
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot()
    fig.add_subplot(ax1)

    moment_avg = df.groupby('Time')['Torque'].mean()
    moment_std = df.groupby('Time')['Torque'].std()
    time_vec = np.arange(0, max_cycle[1]) / 100

    std_range = (moment_avg - moment_std, moment_avg + moment_std)
    std_range[0][np.isnan(std_range[0].values)] = 0
    std_range[1][np.isnan(std_range[1].values)] = 0

    ax1.fill_between(time_vec, std_range[0], std_range[1], alpha=0.2)
    ax1.plot(time_vec, moment_avg, label='Average')
    ax1.set_title(title)
    ax1.set_xlabel('gait cycle duration [s]')
    ax1.set_ylabel(ylabel)

    if plot_min_med_max:
        ax1.plot(df.loc[df['Trial'] == min_cycle[0], 'Time'], df.loc[df['Trial'] == min_cycle[0], 'Torque'],
                 label='Fastest cycle')
        ax1.plot(df.loc[df['Trial'] == max_cycle[0], 'Time'], df.loc[df['Trial'] == max_cycle[0], 'Torque'],
                 label='Slowest cycle')
        ax1.plot(df.loc[df['Trial'] == med_cycle[0], 'Time'], df.loc[df['Trial'] == med_cycle[0], 'Torque'],
                 label='Median cycle')
    ax1.legend()

    fig.add_subplot(ax1)
    if not save_fig_as:
        fig.show()
    else:
        fig.savefig('./figures/' + save_fig_as + '.png', bbox_inches='tight')

    return fig


def plot_muscle_average(df, muscle_list=None, plot_min_and_max=False, title='EMG Muscle Activity', save_fig_as=None):
    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df)

    if not muscle_list:
        muscle_list = list(df)
        muscle_list.remove('Time')
        muscle_list.remove('Torque')
        muscle_list.remove('Trial')

    num_plots = len(muscle_list)
    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(7, 1.7 * num_plots), squeeze=False)
    fig.subplots_adjust(hspace=0.04)
    fig.suptitle(title)
    fig.text(0.06, 0.5, 'Normalized amplitude [V/V]', ha='right', va='center', rotation='vertical')
    for i, muscle in enumerate(muscle_list):
        emg_avg = df.groupby('Time')[muscle].mean()
        emg_std = df.groupby('Time')[muscle].std()
        time_vec = np.arange(0, max_cycle[1]) / 100

        std_range = (emg_avg - emg_std, emg_avg + emg_std)

        axs[i, 0] = plt.subplot(num_plots, 1, i + 1)
        axs[i, 0].fill_between(time_vec, std_range[0], std_range[1], alpha=0.2)
        axs[i, 0].plot(time_vec, emg_avg, label='Average')
        axs[i, 0].set_xlabel('gait cycle duration [s]')

        if plot_min_and_max:
            axs[i, 0].plot(df.loc[df['Trial'] == min_cycle[0], 'Time'],
                           df.loc[df['Trial'] == min_cycle[0], muscle],
                           label='Fastest cycle')
            axs[i, 0].plot(df.loc[df['Trial'] == max_cycle[0], 'Time'],
                           df.loc[df['Trial'] == max_cycle[0], muscle],
                           label='Slowest cycle')
            axs[i, 0].plot(df.loc[df['Trial'] == med_cycle[0], 'Time'],
                           df.loc[df['Trial'] == med_cycle[0], muscle],
                           label='Median cycle')

        axs[i, 0].set_ylim(0, 0.99)
        axs[i, 0].set_yticks(np.arange(0, 1, 0.2))
        axs[i, 0].text(0.72, 0.95, muscle, size='large', ha='left', va='top', transform=axs[i, 0].transAxes)

    if plot_min_and_max:
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        # fig.legend((l1, l2, l3, l4), loc='upper right')

    if not save_fig_as:
        fig.show()
    else:
        fig.savefig('./figures/' + save_fig_as + '.png', bbox_inches='tight')


def plot_cycle_time_quartile(df, title='Gait cycle distribution', save_fig_as=None):
    time_intervals = df.groupby('Trial')['Time'].agg(np.ptp)
    fig = plt.figure(1, figsize=(7, 2))
    ax = fig.add_subplot(111)
    plot_result = ax.boxplot(time_intervals, whis='range', vert=False)
    ax.set_title(title)
    ax.set_xlabel('Cycle duration [s]')
    ax.set_yticklabels([])

    if not save_fig_as:
        fig.show()
    else:
        fig.savefig('./figures/' + save_fig_as + '.png', bbox_inches='tight')

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


def plot_moment_w_muscle(df, plot_trial='average', muscle_list=None, title='Moments and EMG',
                         moment_label=r'$\Delta$ joint moment $[\frac{N.mm}{kg.s}]$',
                         emg_label=r'normalized emg $[V/V]$', save_fig_as=None):
    if plot_trial == 'average' or plot_trial == 'mean':
        moments = df.groupby('Time')['Torque'].mean()
        time_vec = moments.index
    else:
        selected_trial = df.Trial.str.contains(plot_trial)
        moments = df.Torque[selected_trial]
        time_vec = df.Time[selected_trial]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    fig.suptitle(title)

    color = 'blue'
    ax1.set_xlabel('gait cycle duration[s]')
    ax1.set_ylabel(moment_label)
    ax1.plot(time_vec, moments, color=color, label='Joint moment')

    if muscle_list is not None:
        colors = ['red', 'green', 'cyan', 'yellow']
        ax2 = ax1.twinx()
        ax2.set_ylabel(emg_label)
        if plot_trial == 'average' or plot_trial == 'mean':
            for i, muscle in enumerate(muscle_list):
                emg_avg = df.groupby('Time')[muscle].mean()
                ax2.plot(time_vec, emg_avg.values, color=colors[i], label=muscle)
        else:
            for i, muscle in enumerate(muscle_list):
                emg = df[muscle][selected_trial]
                ax2.plot(time_vec, emg.values, color=colors[i], label=muscle)

    fig.legend()

    if not save_fig_as:
        fig.show()
    else:
        fig.savefig('./figures/' + save_fig_as + '.png', bbox_inches='tight')


def plot_moment_derivative(df, plot_trial='average', muscle_list=None):
    if plot_trial == 'average' or plot_trial == 'mean':
        moments = df.groupby('Time')['Torque'].mean()
        time_vec = moments.index
    else:
        selected_trial = df.Trial.str.contains(plot_trial)
        moments = df.Torque[selected_trial]
        time_vec = df.Time[selected_trial]

    moment_derivative = np.diff(moments)

    #plt.figure()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('gait cycle duration[s]')
    ax1.set_ylabel(r'$\Delta$ joint moment $[\frac{N.mm}{kg.s}]$')
    ax1.plot(time_vec[:-1], moment_derivative)

    if muscle_list is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'normalized emg $[V/V]$')
        if plot_trial == 'average' or plot_trial == 'mean':
            for i, muscle in enumerate(muscle_list):
                emg_avg = df.groupby('Time')[muscle].mean()
                ax2.plot(time_vec[:-1], emg_avg.values[:-1])
        else:
            for i, muscle in enumerate(muscle_list):
                emg = df[muscle][selected_trial]
                ax2.plot(time_vec[:-1], emg.values[:-1])

from deprecated import deprecated
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal as scisig


def get_min_med_max_cycles(df):
    df_copy = df.copy()
    grouped_by_cycle = df_copy.groupby('Trial').size()
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


def plot_muscle_correlations(df, method='pearson', title='Correlation'):
    muscle_df = df.copy()
    for column in muscle_df:
        if column in ['Time', 'Torque', 'Trial']:
            muscle_df.pop(column)

    correlation_matrix = muscle_df.corr(method=method)
    plt.figure()
    plt.title(title)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1)


def plot_moment_avg(df, plot_min_med_max=False, title='Knee joint moments', ylabel=r'joint moment $[\frac{N.mm}{kg}]$',
                    as_percentage=True, save_fig_as=None):
    if as_percentage:
        df_copy = set_gait_cycle_percentage(df)
    else:
        df_copy = df.copy()
    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df_copy)
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot()
    fig.add_subplot(ax1)

    if as_percentage:
        trial_groups = [trial for _, trial in df_copy.groupby('Trial')]
        xvec = np.arange(0, 101)
        num_steps = len(xvec)
        moments = np.zeros((len(trial_groups), num_steps))
        for i, df in enumerate(trial_groups):
            moments[i, :] = resample_signal(df.Torque, num_steps)
            if plot_min_med_max:
                min_cycle_xvec = xvec
                med_cycle_xvec = xvec
                max_cycle_xvec = xvec

                if df.iloc[0]['Trial'] == min_cycle[0]:
                    min_cycle_moments = moments[i, :]
                elif df.iloc[0]['Trial'] == med_cycle[0]:
                    med_cycle_moments = moments[i, :]
                elif df.iloc[0]['Trial'] == max_cycle[0]:
                    max_cycle_moments = moments[i, :]

        moment_avg = np.mean(moments, axis=0)
        moment_std = np.std(moments, axis=0)
    else:
        moment_avg = df_copy.groupby('Time')['Torque'].mean()
        moment_std = df_copy.groupby('Time')['Torque'].std()
        xvec = np.arange(0, max_cycle[1]) / 100

        if plot_min_med_max:
            min_cycle_xvec = df_copy.loc[df_copy['Trial'] == min_cycle[0], 'Time']
            med_cycle_xvec = df_copy.loc[df_copy['Trial'] == med_cycle[0], 'Time']
            max_cycle_xvec = df_copy.loc[df_copy['Trial'] == max_cycle[0], 'Time']

            min_cycle_moments = df_copy.loc[df_copy['Trial'] == min_cycle[0], 'Torque']
            med_cycle_moments = df_copy.loc[df_copy['Trial'] == med_cycle[0], 'Torque']
            max_cycle_moments = df_copy.loc[df_copy['Trial'] == max_cycle[0], 'Torque']

    std_range = (moment_avg - moment_std, moment_avg + moment_std)
    if not as_percentage:
        std_range[0][np.isnan(std_range[0].to_numpy())] = 0
        std_range[1][np.isnan(std_range[1].to_numpy())] = 0

    ax1.fill_between(xvec, std_range[0], std_range[1], alpha=0.2)
    ax1.plot(xvec, moment_avg, label='Average')
    ax1.set_title(title)
    if as_percentage:
        ax1.set_xlabel('Percentage of gait cycle [%]')
    else:
        ax1.set_xlabel('gait cycle duration [s]')
    ax1.set_ylabel(ylabel)

    if plot_min_med_max:
        print('The fastest cycles was: ' + min_cycle[0] + '\n\tTime steps: ' + str(min_cycle[1]))
        print('The slowest cycles was: ' + max_cycle[0] + '\n\tTime steps: ' + str(max_cycle[1]))
        ax1.plot(min_cycle_xvec, min_cycle_moments, label='Fastest cycle')
        ax1.plot(med_cycle_xvec, med_cycle_moments, label='Median cycle')
        ax1.plot(max_cycle_xvec, max_cycle_moments, label='Slowest cycle')
    ax1.legend()

    fig.add_subplot(ax1)
    if not save_fig_as:
        fig.show()
    else:
        fig.savefig('./figures/' + save_fig_as + '.png', bbox_inches='tight')

    return fig


def plot_muscle_average(df, muscle_list=None, plot_min_med_max=False, title='EMG Muscle Activity',
                        ylabel='EMG amplitude', save_fig_as=None, y_lim=None,
                        plot_max_emg=False, as_percentage=True):
    if as_percentage:
        df_copy = set_gait_cycle_percentage(df)
    else:
        df_copy = df.copy()

    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df_copy)

    if not muscle_list:
        muscle_list = list(df_copy)
        muscle_list.remove('Time')
        muscle_list.remove('Torque')
        muscle_list.remove('Trial')

    num_plots = len(muscle_list)
    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(7, 1.7 * num_plots), squeeze=False)
    fig.subplots_adjust(hspace=0.04)
    fig.suptitle(title)
    fig.text(0.06, 0.5, ylabel, ha='right', va='center', rotation='vertical')
    for i, muscle in enumerate(muscle_list):
        if as_percentage:
            trial_groups = [trial for _, trial in df_copy.groupby('Trial')]
            xvec = np.arange(0, 101)
            num_steps = len(xvec)
            emg_vec = np.zeros((len(trial_groups), num_steps))
            for j, df in enumerate(trial_groups):
                emg_vec[j, :] = resample_signal(df[muscle], num_steps)
                if plot_min_med_max:
                    min_cycle_xvec = xvec
                    med_cycle_xvec = xvec
                    max_cycle_xvec = xvec

                    if df.iloc[0]['Trial'] == min_cycle[0]:
                        min_cycle_emg = emg_vec[j, :]
                    elif df.iloc[0]['Trial'] == med_cycle[0]:
                        med_cycle_emg = emg_vec[j, :]
                    elif df.iloc[0]['Trial'] == max_cycle[0]:
                        max_cycle_emg = emg_vec[j, :]

                if plot_max_emg:
                    max_emg_xvec = xvec

                    if df.iloc[0]['Trial'] == df_copy.loc[df_copy[muscle].idxmax()]['Trial']:
                        max_emg = emg_vec[j, :]

            emg_avg = np.mean(emg_vec, axis=0)
            emg_std = np.std(emg_vec, axis=0)
        else:
            emg_avg = df_copy.groupby('Time')[muscle].mean()
            emg_std = df_copy.groupby('Time')[muscle].std()
            xvec = np.arange(0, max_cycle[1]) / 100

            if plot_min_med_max:
                min_cycle_xvec = df_copy.loc[df_copy['Trial'] == min_cycle[0], 'Time']
                med_cycle_xvec = df_copy.loc[df_copy['Trial'] == med_cycle[0], 'Time']
                max_cycle_xvec = df_copy.loc[df_copy['Trial'] == max_cycle[0], 'Time']

                min_cycle_emg = df_copy.loc[df_copy['Trial'] == min_cycle[0], muscle]
                med_cycle_emg = df_copy.loc[df_copy['Trial'] == med_cycle[0], muscle]
                max_cycle_emg = df_copy.loc[df_copy['Trial'] == max_cycle[0], muscle]

            if plot_max_emg:
                max_trial = df_copy.loc[df_copy[muscle].idxmax()]['Trial']
                max_emg_xvec = df_copy.loc[df_copy['Trial'] == max_trial, 'Time']
                max_emg = df_copy.loc[df_copy['Trial'] == max_trial, muscle]

        std_range = (emg_avg - emg_std, emg_avg + emg_std)

        axs[i, 0] = plt.subplot(num_plots, 1, i + 1)
        axs[i, 0].fill_between(xvec, std_range[0], std_range[1], alpha=0.2)
        axs[i, 0].plot(xvec, emg_avg, label='Average')
        if as_percentage:
            axs[i, 0].set_xlabel('Percentage of gait cycle [%]')
        else:
            axs[i, 0].set_xlabel('gait cycle duration [s]')

        if plot_min_med_max:
            axs[i, 0].plot(min_cycle_xvec, min_cycle_emg, label='Fastest cycle')
            axs[i, 0].plot(med_cycle_xvec, med_cycle_emg, label='Slowest cycle')
            axs[i, 0].plot(max_cycle_xvec, max_cycle_emg, label='Median cycle')

        if plot_max_emg:
            max_trial = df_copy.loc[df_copy[muscle].idxmax()]['Trial']
            print('The trial with maximum signal for muscle ' + muscle + ': ' + max_trial)
            axs[i, 0].plot(max_emg_xvec, max_emg, label='Muscle max value')

        if y_lim is None:
            axs[i, 0].set_ylim(0, 0.99)
        else:
            axs[i, 0].set_ylim(y_lim[0], y_lim[1])
        axs[i, 0].set_yticks(np.arange(y_lim[0], y_lim[1], 0.2))
        axs[i, 0].text(0.72, 0.95, muscle, size='large', ha='left', va='top', transform=axs[i, 0].transAxes)

    if plot_min_med_max:
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


def plot_emg_from_trial(df, muscle_list, trial_name):
    df_copy = df.copy()

    for group, df_i in df_copy.groupby('Trial'):
        if group == trial_name:
            t = df_i.pop('Time')
            for muscle in muscle_list:
                fig, ax1 = plt.subplots()
                color = 'tab:blue'
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Muscle signal', color=color)
                ax1.plot(t, df_i[muscle])
                ax1.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.title(muscle)
                plt.show()


def plot_moment_w_muscle(df, plot_trial='average', muscle_list=None, title='Moments and EMG',
                         moment_label=r'$\Delta$ joint moment $[\frac{N.mm}{kg.s}]$',
                         emg_label=r'normalized emg $[V/V]$', save_fig_as=None, as_percentage=True):
    if as_percentage:
        df_copy = set_gait_cycle_percentage(df)
    else:
        df_copy = df.copy()

    if plot_trial == 'average' or plot_trial == 'mean':
        if as_percentage:
            trial_groups = [trial for _, trial in df_copy.groupby('Trial')]
            xvec = np.arange(0, 101)
            num_steps = len(xvec)
            moments = np.zeros((len(trial_groups), num_steps))
            for i, df in enumerate(trial_groups):
                moments[i, :] = resample_signal(df.Torque, num_steps)
            moments = np.mean(moments, axis=0)
        else:
            moments = df.groupby('Time')['Torque'].mean()
            xvec = moments.index
    else:
        selected_trial = df.Trial.str.contains(plot_trial)
        moments = df.Torque[selected_trial]
        if as_percentage:
            xvec = np.linspace(0, 100, len(moments), endpoint=True)
        else:
            xvec = df.Time[selected_trial]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    fig.suptitle(title)

    color = 'blue'
    ax1.set_xlabel('gait cycle duration[s]')
    ax1.set_ylabel(moment_label)
    ax1.plot(xvec, moments, color=color, label='Joint moment')

    if muscle_list is not None:
        colors = ['red', 'green', 'cyan', 'yellow']
        ax2 = ax1.twinx()
        ax2.set_ylabel(emg_label)
        if plot_trial == 'average' or plot_trial == 'mean':
            for i, muscle in enumerate(muscle_list):
                if as_percentage:
                    emg_vec = np.zeros((len(trial_groups), num_steps))
                    for j, df in enumerate(trial_groups):
                        emg_vec[j, :] = resample_signal(df[muscle], num_steps)
                    emg_avg = np.mean(emg_vec, axis=0)
                else:
                    emg_avg = df.groupby('Time')[muscle].mean()
                    emg_avg = emg_avg.to_numpy()
                ax2.plot(xvec, emg_avg, color=colors[i], label=muscle)
        else:
            for i, muscle in enumerate(muscle_list):
                emg = df[muscle][selected_trial]
                if as_percentage:
                    xvec = np.linspace(0, 100, len(emg), endpoint=True)
                else:
                    xvec = df.Time[selected_trial]
                ax2.plot(xvec, emg.to_numpy(), color=colors[i], label=muscle)

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
                ax2.plot(time_vec[:-1], emg_avg.to_numpy()[:-1])
        else:
            for i, muscle in enumerate(muscle_list):
                emg = df[muscle][selected_trial]
                ax2.plot(time_vec[:-1], emg.to_numpy()[:-1])


def set_gait_cycle_percentage(df):
    return_df = df.copy()

    percentage_list = []
    for _, group in return_df.groupby('Trial'):
        len_cycle = group.count()['Time']
        for i in range(len_cycle):
            percentage_list.append((i/(len_cycle-1))*100)

    return_df['Percentage'] = percentage_list

    return return_df


def resample_signal(signal, new_sample_length):
    if len(signal) == new_sample_length:
        return signal
    else:
        return scisig.resample(signal, new_sample_length)

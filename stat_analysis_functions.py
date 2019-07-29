from deprecated import deprecated
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
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


def plot_muscle_correlations(df, method='pearson', include_torque=False, title=None, save_fig_as=None):
    muscle_df = df.copy()

    if include_torque:
        muscle_df.drop(columns=['Time', 'Trial'], errors='ignore', inplace=True)
    else:
        muscle_df.drop(columns=['Time', 'Torque', 'Trial'], errors='ignore', inplace=True)

    correlation_matrix = muscle_df.corr(method=method)
    plt.figure()
    if title is not None:
        plt.title(title)
    sns_plot = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", vmin=-1, vmax=1)

    fig = sns_plot.get_figure()

    if not save_fig_as:
        fig.show()
    else:
        save_image_as(fig, './figures/correlations/', save_fig_as)

    return fig


def plot_moment_avg(df, plot_min_med_max=False, plot_worst=False, title=None, xlabel=None,
                    ylabel=None, y_axis_range=None, plot_font_size=12, save_fig_as=None):
    df_copy = df.copy()
    df_copy = df_copy[df_copy.Time >= 0]
    df_copy.reset_index(drop=True, inplace=True)

    min_cycle, med_cycle, max_cycle = get_min_med_max_cycles(df_copy)
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot()
    fig.add_subplot(ax1)

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

    std_range = (moment_avg - moment_std, moment_avg + moment_std)

    if plot_worst:
        mse = 0
        for i, df in enumerate(trial_groups):
            temp_mse = (np.square(moment_avg - moments[i, :])).mean()
            if temp_mse > mse:
                mse = temp_mse
                worst_cycle = moments[i, :]
                worst_cycle_name = df.Trial.iloc[0]
                print(worst_cycle_name)

    ax1.fill_between(xvec, std_range[0], std_range[1], alpha=0.2)
    if plot_min_med_max:
        ax1.plot(xvec, moment_avg, label='Average')
    else:
        ax1.plot(xvec, moment_avg)
    if title is not None:
        ax1.set_title(title)
    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    if ylabel is not None:
        ax1.set_ylabel(ylabel)

    if y_axis_range is not None:
        ax1.set_ylim(y_axis_range[0], y_axis_range[1])

    if plot_font_size is not None:
        ax1.tick_params(labelsize=plot_font_size)

    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax1.xaxis.set_major_formatter(xticks)

    if plot_worst:
        print('The worst cycle was: ' + worst_cycle_name)
        ax1.plot(xvec, worst_cycle, label='Worst cycle')

    if plot_min_med_max:
        print('The fastest cycle was: ' + min_cycle[0] + '\n\tTime steps: ' + str(min_cycle[1]))
        print('The slowest cycle was: ' + max_cycle[0] + '\n\tTime steps: ' + str(max_cycle[1]))
        ax1.plot(min_cycle_xvec, min_cycle_moments, label='Fastest cycle')
        ax1.plot(med_cycle_xvec, med_cycle_moments, label='Median cycle')
        ax1.plot(max_cycle_xvec, max_cycle_moments, label='Slowest cycle')
        ax1.legend()

    fig.add_subplot(ax1)
    if not save_fig_as:
        fig.show()
    else:
        save_image_as(fig, './figures/moment_avg/', save_fig_as)

    return fig


def plot_muscle_average(df, muscle_list=None, ylabel=None, y_axis_range=None, plot_font_size=12, save_fig_as=None,):
    df_copy = df.copy()
    df_copy = df_copy[df_copy.Time >= 0]
    df_copy.reset_index(drop=True, inplace=True)

    if not muscle_list:
        muscle_list = list(df_copy)
        muscle_list.remove('Time')
        muscle_list.remove('Torque')
        muscle_list.remove('Trial')

    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot()

    if muscle_list is not None:
        colors = ['red', 'green', 'cyan', 'yellow']
        if ylabel is not None:
            ax1.set_ylabel(ylabel)
        for i, muscle in enumerate(muscle_list):
            trial_groups = [trial for _, trial in df_copy.groupby('Trial')]
            xvec = np.arange(0, 101)
            num_steps = len(xvec)
            emg_vec = np.zeros((len(trial_groups), num_steps))
            for j, df in enumerate(trial_groups):
                emg_vec[j, :] = resample_signal(df[muscle], num_steps)
            emg_avg = np.mean(emg_vec, axis=0)
            emg_std = np.std(emg_vec, axis=0)

            std_range = (emg_avg - emg_std, emg_avg + emg_std)

            ax1.fill_between(xvec, std_range[0], std_range[1], color=colors[i], alpha=0.2)
            ax1.plot(xvec, emg_avg, color=colors[i], label=muscle)

    if y_axis_range is not None:
        ax1.set_ylim(y_axis_range[0], y_axis_range[1])

    if plot_font_size is not None:
        ax1.tick_params(labelsize=plot_font_size)

    ax1.legend(fontsize=plot_font_size)

    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax1.xaxis.set_major_formatter(xticks)

    fig.add_subplot(ax1)
    if not save_fig_as:
        fig.show()
    else:
        save_image_as(fig, './figures/emg_avg/', save_fig_as)

    return fig


def plot_cycle_time_quartile(df, title='Gait cycle distribution', save_fig_as=None):
    time_intervals = df.groupby('Trial')['Time'].agg(np.ptp)
    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    # ax = fig.add_subplot(111)
    plot_result = ax.boxplot(time_intervals, whis='range', vert=False)
    ax.set_title(title)
    ax.set_xlabel('Cycle duration [s]')
    ax.set_yticklabels([])

    if not save_fig_as:
        fig.show()
    else:
        save_image_as(fig, './figures/cycle_time_quartiles/', save_fig_as)

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
                         emg_label=r'normalized emg $[V/V]$', save_fig_as=None):
    df_copy = set_gait_cycle_percentage(df)

    if plot_trial == 'average' or plot_trial == 'mean':
        trial_groups = [trial for _, trial in df_copy.groupby('Trial')]
        xvec = np.arange(0, 101)
        num_steps = len(xvec)
        moments = np.zeros((len(trial_groups), num_steps))
        for i, df in enumerate(trial_groups):
            moments[i, :] = resample_signal(df.Torque, num_steps)
        moments = np.mean(moments, axis=0)
    else:
        selected_trial = df.Trial.str.contains(plot_trial)
        moments = df.Torque[selected_trial]
        xvec = np.linspace(0, 100, len(moments), endpoint=True)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    fig.suptitle(title)

    color = 'blue'
    ax1.set_xlabel('Percentage of gait cycle [%]')
    ax1.set_ylabel(moment_label)
    ax1.plot(xvec, moments, color=color, label='Joint moment')

    if muscle_list is not None:
        colors = ['red', 'green', 'cyan', 'yellow']
        ax2 = ax1.twinx()
        ax2.set_ylabel(emg_label)
        if plot_trial == 'average' or plot_trial == 'mean':
            for i, muscle in enumerate(muscle_list):
                emg_vec = np.zeros((len(trial_groups), num_steps))
                for j, df in enumerate(trial_groups):
                    emg_vec[j, :] = resample_signal(df[muscle], num_steps)
                emg_avg = np.mean(emg_vec, axis=0)

                ax2.plot(xvec, emg_avg, color=colors[i], label=muscle)
        else:
            for i, muscle in enumerate(muscle_list):
                emg = df[muscle][selected_trial]

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
    return_df = return_df[return_df.Time >= 0]
    return_df.reset_index(drop=True, inplace=True)

    percentage_list = []
    for _, group in return_df.groupby('Trial'):
        len_cycle = group.count()['Time']
        for i in range(len_cycle):
            percentage_list.append((i/(len_cycle-1))*100)

    return_df['Percentage'] = percentage_list

    return return_df


def plot_grid_emg_average(dfs_list, row_names, col_names, muscle_lists=None, ylabel=None, y_axis_range=None,
                          plot_font_size=12, save_fig_as=None,):
    if muscle_lists is None:
        muscle_lists = [['RectFem', 'VasMed', 'VasLat'], ['BicFem', 'Semitend'], ['GlutMax'],
                        ['TibAnt', 'Soleus', 'GasMed', 'GasLat']]
    num_row = len(row_names)
    num_col = len(col_names)
    if len(dfs_list) != num_col:
        logging.error('The number of column names is not equal to number of subjects dataframes')
    if len(muscle_lists) != num_row:
        logging.error('The number of row names is not equal to number of muscle groups')

    fig, axes = plt.subplots(nrows=num_row, ncols=num_col, figsize=(11.69, 8.27), sharex=True, sharey=True,
                             squeeze=False)
    fig.subplots_adjust(hspace=0.04, wspace=0.04)

    for n, subject in enumerate(dfs_list):
        df_copy = subject.copy()

        for m, muscle_list in enumerate(muscle_lists):
            if muscle_list is not None:
                colors = ['red', 'green', 'cyan', 'yellow']
                if ylabel is not None:
                    axes[m, n].set_ylabel(ylabel)
                for i, muscle in enumerate(muscle_list):
                    trial_groups = [trial for _, trial in df_copy.groupby('Trial')]
                    xvec = np.arange(0, 101)
                    num_steps = len(xvec)
                    emg_vec = np.zeros((len(trial_groups), num_steps))
                    for j, df in enumerate(trial_groups):
                        emg_vec[j, :] = resample_signal(df[muscle], num_steps)
                    emg_avg = np.mean(emg_vec, axis=0)
                    emg_std = np.std(emg_vec, axis=0)

                    std_range = (emg_avg - emg_std, emg_avg + emg_std)

                    axes[m, n].fill_between(xvec, std_range[0], std_range[1], color=colors[i], alpha=0.2)
                    axes[m, n].plot(xvec, emg_avg, color=colors[i], label=muscle)

            if m == 0:
                axes[m, n].set_title(col_names[n], fontsize=plot_font_size)
            if n == 0:
                axes[m, n].set_ylabel(row_names[m], fontsize=plot_font_size)

            if y_axis_range is not None:
                axes[m, n].set_ylim(y_axis_range[0], y_axis_range[1])

            if plot_font_size is not None:
                axes[m, n].tick_params(labelsize=plot_font_size)

            if n == num_col - 1:
                axes[m, n].legend(fontsize=plot_font_size, loc='upper right')

            fmt = '%.0f%%'
            xticks = mtick.FormatStrFormatter(fmt)
            axes[m, n].xaxis.set_major_formatter(xticks)

    if not save_fig_as:
        fig.show()
    else:
        save_image_as(fig, './figures/emg_avg/', save_fig_as)


def resample_signal(signal, new_sample_length):
    if len(signal) == new_sample_length:
        return signal
    else:
        return scisig.resample(signal, new_sample_length)


def save_image_as(fig_object, directory, file_name):
    # directory = './figures/moment_avg/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig_object.savefig(directory + file_name + '.png', bbox_inches='tight')

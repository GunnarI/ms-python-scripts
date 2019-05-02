import matplotlib.pyplot as plt


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

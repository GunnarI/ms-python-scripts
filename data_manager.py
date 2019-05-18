import os
import sys
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd


class DataManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.emg_data_dict = {}
        self.torque_data_dict = {}
        self.filt_data_dict = {}
        self.filt_concat_data_dict = {}

        with open('./data_structure.json', 'r') as ds:
            self.data_struct = json.load(ds)

    def update_data_structure(self):
        subject_folders = Path(self.base_dir).glob('Subject*')
        subject_ids = get_list_of_subject_ids(self.data_struct)

        for folder in subject_folders:
            subject_id = folder.stem

            if subject_id not in subject_ids:
                self.data_struct.append(create_subject(self.base_dir, subject_id))
            else:
                i = subject_ids.index(subject_id)
                self.data_struct[i] = update_subject(self.base_dir + '\\' + subject_id, self.data_struct[i])

        with open('./data_structure.json', 'w') as ds:
            json.dump(self.data_struct, ds, indent=4)

    def load_emg_and_torque(self, subject_id, session_id, reload=False, load_filt=False):
        """Loads the raw emg data either from the original csv file or previously saved txt file.
        If txt file does not already exist then a new one is saved. All emg data (both from csv and txt files) is added
        to the DataManager.emg_data_dict dictionary, each trial with the key "<SessionID> <SubjectID> <TrialID>".
        Same goes for the torque data and DataManager.torque_data_dict.
        :param subject_id: the subject id from the data structure (e.g. "Subject01")
        :param session_id: the session id from the data structure (e.g. "20190405")
        :param reload: if True then the already saved txt files are overridden with values from the csv
        :param load_filt: if True then the already saved txt files with filtered data is loaded as Pandas.DataFrame
        :return: nothing, only updates the class variables DataManager.emg_data_dict and DataManager.torque_data_dict
        """
        for subject in self.data_struct:
            if subject["SubjectID"] == subject_id:
                for session in subject["Sessions"]:
                    if session["SessionID"] == session_id:
                        emg_device = session["EMGDevice"]
                        num_emg = session["NumEMG"]
                        frame_freq = session["FramesPerSec"]
                        analog_freq = session["AnalogFreq"]
                        emg_headers = ['Time']
                        for key in session["EMGProtocol"]:
                            emg_headers.append(session["EMGProtocol"][key])

                        for trial in session["Trials"]:
                            session_trial_id = session_id + ' ' + subject_id + ' ' + trial["TrialID"]
                            emg_array_dir = './data/' + session_trial_id + '.txt'

                            if 'walk' in trial["TrialID"].lower():
                                try:
                                    t1, t2, gait_cycles = get_gait_cycles(trial["File"])
                                except AssertionError as error:
                                    print(error)
                                    continue

                                # Load emg data
                                if os.path.isfile(emg_array_dir) and not reload:
                                    self.emg_data_dict[session_trial_id] = {"headers": emg_headers,
                                                                             "data": np.loadtxt(emg_array_dir),
                                                                             "t1": t1, "t2": t2,
                                                                             "gait_cycles": gait_cycles}
                                else:
                                    emg_data = get_emg_from_csv(trial["File"], emg_device, num_emg, frame_freq,
                                                                analog_freq)
                                    self.emg_data_dict[session_trial_id] = {"headers": emg_headers,
                                                                             "data": emg_data, "t1": t1, "t2": t2,
                                                                             "gait_cycles": gait_cycles}
                                    np.savetxt(emg_array_dir, emg_data, fmt='%f')

                                # Load torque data
                                torque_array_dir = './data/labels/' + session_trial_id + '.txt'
                                if os.path.isfile(torque_array_dir) and not reload:
                                    self.torque_data_dict[session_trial_id] = {"data": np.loadtxt(torque_array_dir),
                                                                                "t1": t1, "t2": t2,
                                                                                "gait_cycles": gait_cycles}
                                else:
                                    torque_data = get_torque_from_csv(trial["File"],
                                                                      subject_id + ':RKneeMoment',
                                                                      frame_freq)
                                    self.torque_data_dict[session_trial_id] = {"data": torque_data, "t1": t1, "t2": t2,
                                                                                "gait_cycles": gait_cycles}
                                    np.savetxt(torque_array_dir, torque_data, fmt='%f')

                                # Load filtered data
                                if load_filt:
                                    filt_emg_array_dir = './data/' + session_trial_id + ' filtered.txt'
                                    filt_torque_array_dir = './data/labels/' + session_trial_id + ' filtered.txt'
                                    if os.path.isfile(filt_emg_array_dir) and os.path.isfile(filt_torque_array_dir):
                                        for cycle in gait_cycles:
                                            pd_emg = pd.read_csv(filt_emg_array_dir, sep=' ').set_index(
                                                'Time').truncate(
                                                gait_cycles[cycle]['Start'], gait_cycles[cycle]['End'])
                                            pd_torque = pd.read_csv(filt_torque_array_dir, sep=' ').set_index(
                                                'Time').truncate(
                                                gait_cycles[cycle]['Start'], gait_cycles[cycle]['End'])
                                            self.filt_data_dict[session_trial_id + cycle] = pd_emg.join(
                                                pd_torque, how='outer').reset_index()

                                            df_to_add = self.filt_data_dict[session_trial_id + cycle].copy()
                                            df_to_add['Time'] = (
                                                    df_to_add['Time'] - gait_cycles[cycle]['Start']).round(decimals=2)
                                            df_to_add['Trial'] = trial['TrialID'] + cycle

                                            if session_id + ' ' + subject_id + ' filtered' \
                                                    not in self.filt_concat_data_dict:
                                                self.filt_concat_data_dict[
                                                    session_id + ' ' + subject_id + ' filtered'] = df_to_add
                                            elif not self.filt_concat_data_dict[
                                                session_id + ' ' + subject_id + ' filtered'
                                            ]['Trial'].str.contains(trial['TrialID'] + cycle).any():
                                                self.filt_concat_data_dict[
                                                    session_id + ' ' + subject_id + ' filtered'] = pd.concat(
                                                    [self.filt_concat_data_dict[session_id + ' ' + subject_id +
                                                                                ' filtered'], df_to_add],
                                                    ignore_index=True)
                            else:
                                if os.path.isfile(emg_array_dir) and not reload:
                                    self.emg_data_dict[session_trial_id] = {"headers": emg_headers,
                                                                             "data": np.loadtxt(emg_array_dir)}
                                else:
                                    emg_data = get_emg_from_csv(trial["File"],
                                                                emg_device, num_emg, frame_freq, analog_freq)
                                    self.emg_data_dict[session_trial_id] = {"headers": emg_headers, "data": emg_data}
                                    np.savetxt(emg_array_dir, emg_data, fmt='%f')

    def update_filt_data_dict(self):
        for key in self.torque_data_dict:
            ids = key.split()
            gait_cycle_dict = self.torque_data_dict[key]['gait_cycles']

            filt_emg_array_dir = './data/' + key + ' filtered.txt'
            filt_torque_array_dir = './data/labels/' + key + ' filtered.txt'
            if os.path.isfile(filt_emg_array_dir) and os.path.isfile(filt_torque_array_dir):
                # gait_cycle_dict = self.torque_data_dict[key]['gait_cycles']
                for cycle in gait_cycle_dict:
                    pd_emg = pd.read_csv(filt_emg_array_dir, sep=' ').set_index('Time').truncate(
                        gait_cycle_dict[cycle]['Start'], gait_cycle_dict[cycle]['End'])
                    pd_torque = pd.read_csv(filt_torque_array_dir, sep=' ').set_index('Time').truncate(
                        gait_cycle_dict[cycle]['Start'], gait_cycle_dict[cycle]['End'])
                    self.filt_data_dict[key + cycle] = pd_emg.join(
                        pd_torque, how='outer').reset_index()

            for cycle in gait_cycle_dict:
                df_to_add = self.filt_data_dict[key + cycle].copy()
                df_to_add['Time'] = (df_to_add['Time'] - gait_cycle_dict[cycle]['Start']).round(decimals=2)
                df_to_add['Trial'] = ids[2] + cycle

                if ids[0] + ' ' + ids[1] + ' filtered' not in self.filt_concat_data_dict:
                    self.filt_concat_data_dict[ids[0] + ' ' + ids[1] + ' filtered'] = df_to_add
                elif not self.filt_concat_data_dict[ids[0] + ' ' + ids[1] + ' filtered']['Trial'] \
                        .str.contains(ids[2] + cycle).any():
                    self.filt_concat_data_dict[ids[0] + ' ' + ids[1] + ' filtered'] = pd.concat(
                        [self.filt_concat_data_dict[ids[0] + ' ' + ids[1] + ' filtered'], df_to_add], ignore_index=True)


# Creates the new subject dictionary including all sessions and experiments
def create_subject(path, subject_id):
    sessions = []
    session_folders = Path(path + '\\' + subject_id).glob('*')
    for folder in session_folders:
        session_id = str(folder)
        sessions.append(create_session(session_id, subject_id))

    return {"SubjectID": subject_id, "Sessions": sessions}


def update_subject(path, subject):
    session_ids = get_list_of_session_ids(subject)
    session_folders = Path(path).glob('*')
    for folder in session_folders:
        session_id = folder.stem

        if session_id not in session_ids:
            subject["Sessions"].append(create_session(str(folder), subject["SubjectID"]))
        else:
            i = session_ids.index(session_id)
            subject["Sessions"][i]["Trials"] = \
                update_trials(str(folder) + '\\' + subject["SubjectID"], subject["Sessions"][i], subject["SubjectID"])

    return subject


def create_session(path, subject_id):
    with open(path + '\\session_config.json', 'r') as tc:
        session = json.load(tc)

    trials = []
    trial_files = Path(path + '\\' + subject_id).glob('*.csv')
    for file in trial_files:
        trial_name = file.stem.replace(subject_id + ' ', '')
        if 'walk' in trial_name.lower():
            ex_type = 'inverse_dynamics'
        else:
            ex_type = 'emg_only'
        trials.append({"TrialID": trial_name, "Type": ex_type, "File": str(file)})

    session['Trials'] = trials

    return session


def update_trials(path, session, subject_id):
    trial_ids = get_list_of_trial_ids(session)
    trial_files = Path(path).glob('*.csv')
    for file in trial_files:
        trial_id = file.stem.replace(subject_id + ' ', '')
        if trial_id not in trial_ids:
            if 'walk' in trial_id.lower():
                ex_type = 'inverse_dynamics'
            else:
                ex_type = 'emg_only'
            session["Trials"].append({"TrialID": trial_id, "Type": ex_type, "File": str(file)})

    return session["Trials"]


def get_list_of_subject_ids(data_structure):
    subject_ids = []
    for subject in data_structure:
        subject_ids.append(subject["SubjectID"])

    return subject_ids


def get_list_of_session_ids(subject):
    session_ids = []
    for session in subject["Sessions"]:
        session_ids.append(session["SessionID"])

    return session_ids


def get_list_of_trial_ids(session):
    trial_ids = []
    for trial in session["Trials"]:
        trial_ids.append(trial["TrialID"])

    return trial_ids


def get_subject_structure(data_structure, subject_id):
    for subject in data_structure:
        if subject["SubjectID"] == subject_id:
            return subject


def get_torque_from_csv(file, torque_id, frame_freq):
    if not os.path.exists(file):
        raise Exception('The file ' + file + ' could not be found!')

    try:
        f = open(file, 'r')
    except IOError:
        print('Could not read file: ', file)
        sys.exit()

    torque_data = []
    with f:
        fl = f.readline()
        while 'Model' not in fl:
            fl = f.readline()

        f.__next__()

        reader = csv.reader(f, delimiter=',')

        first_col = next(reader).index(torque_id)
        next(reader)
        next(reader)

        for line in reader:
            if len(line) < 10:
                f.close()
                break
            else:
                torque_data.append(line[:2] + line[first_col:first_col + 3])

    torque_data = np.array(torque_data, dtype=np.float)
    t = (torque_data[:, 0] / frame_freq)

    return np.concatenate((t.reshape(t.shape[0], 1), torque_data[:, 2:]), axis=1)


def get_emg_from_csv(file, emg_device, num_emg, frame_freq, analog_freq):
    if not os.path.exists(file):
        raise Exception('The file ' + file + ' could not be found!')

    try:
        f = open(file, 'r')
    except IOError:
        print('Could not read file: ', file)
        sys.exit()

    emg_data = []
    with f:
        fl = f.readline()
        while 'Devices' not in fl:
            fl = f.readline()

        f.__next__()

        reader = csv.reader(f, delimiter=',')

        devices = next(reader)
        emg_first_col = devices.index(emg_device)
        headers = next(reader)
        headers = headers[emg_first_col:emg_first_col+num_emg]
        next(reader)
        next(reader)

        for line in reader:
            if len(line) < num_emg + 1 or '' in line[emg_first_col:emg_first_col + num_emg]:
                f.close()
                break
            else:
                emg_data.append(line[:2] + line[emg_first_col:emg_first_col + num_emg])

    emg_data = np.array(emg_data, dtype=np.float)
    t = (emg_data[:, 0] + (emg_data[:, 1] / (analog_freq / frame_freq))) / frame_freq

    return np.concatenate((t.reshape(t.shape[0], 1), emg_data[:, 2:]), axis=1)


def get_gait_cycles(file):
    """Reads the time-frame of the gait cycles of the right leg from the .csv file exported from Nexus

    :param file: the full path to the csv file
    :return: the strings t1, t2 which are respectively the start and stop time of gait cycle activity of the right leg
    """
    if not os.path.exists(file):
        raise Exception('The file ' + file + ' could not be found!')

    try:
        f = open(file, 'r')
    except IOError:
        print('Could not read file: ', file)
        sys.exit()

    with f:
        reader = csv.reader(f, delimiter=',')
        row = next(reader)

        # Find events
        while 'Events' not in row:
            row = next(reader)

        next(reader)
        event_col_headers = next(reader)
        event_context_col = event_col_headers.index('Context')
        event_name_col = event_col_headers.index('Name')
        event_time_col = event_col_headers.index('Time (s)')

        right_fp_time = []
        time_value = 0
        row = next(reader)
        while 'General' == row[event_context_col]:
            if 'Right-FP' == row[event_name_col]:
                time_value = row[event_time_col]
                right_fp_time.append(time_value)
            row = next(reader)
            while row[event_time_col] == time_value:
                row = next(reader)

        assert (len(right_fp_time) > 0), 'No right foot FP event detected: ' + file

        while 'Right' != row[event_context_col]:
            row = next(reader)

        gait_cycles_name = []
        gait_cycles_time = []
        while 'Right' in row:
            gait_cycles_name.append(row[event_name_col])
            gait_cycles_time.append(row[event_time_col])
            row = next(reader)

        # Define cycle(s) into dictionary
        right_fp_time = np.array(right_fp_time, dtype=np.float)
        gait_cycles_time = np.array(gait_cycles_time, dtype=np.float)
        gait_cycle_dict = {}
        faulty_cycle_num = 0
        for i in range(len(right_fp_time)):
            items_before_fp_max, = np.where(gait_cycles_time < right_fp_time[i])
            items_after_fp_max, = np.where(gait_cycles_time > right_fp_time[i])
            if items_before_fp_max.size > 0 and items_after_fp_max.size > 0:
                faulty_cycle = not (
                        gait_cycles_name[items_before_fp_max[items_before_fp_max.size-1]] == 'Foot Strike' and
                        gait_cycles_name[items_after_fp_max[0]] == 'Foot Off' and
                        gait_cycles_name[items_after_fp_max[1]] == 'Foot Strike'
                )
            else:
                faulty_cycle = True

            if faulty_cycle:
                faulty_cycle_num = faulty_cycle_num + 1
                continue

            gait_cycle_dict['Cycle' + str(i + 1 - faulty_cycle_num)] = {
                'Start': np.around(gait_cycles_time[items_before_fp_max[len(items_before_fp_max)-1]], decimals=2),
                'End': np.around(gait_cycles_time[items_after_fp_max[1]], decimals=2)
            }

        assert (len(right_fp_time)-faulty_cycle_num > 0), 'All right foot cycles defective in: ' + file

        t1 = gait_cycle_dict['Cycle1']['Start']
        t2 = gait_cycle_dict['Cycle' + str(len(right_fp_time)-faulty_cycle_num)]['End']

    return t1, t2, gait_cycle_dict

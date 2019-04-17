import os
import sys
import csv
import json
from pathlib import Path
import numpy as np


class DataManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.emg_data_dict = {}
        self.torque_data_dict = {}

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

    def load_emg_and_torque(self, subject_id, trial_id, reload=False):
        """Loads the raw emg data either from the original csv file or previously saved txt file.
        If txt file does not already exist then a new one is saved. All emg data (both from csv and txt files) is added
        to the DataManager.emg_data_dict dictionary, each exercise with the key "<TrialID> <SubjectID> <ExerciseID>".
        Same goes for the torque data and DataManager.torque_data_dict.
        :param subject_id: the subject id from the data structure (e.g. "Subject01")
        :param trial_id: the trial id from the data structure (e.g. "20190405")
        :param reload: if True then the already saved txt files are overridden with values from the csv
        :return: nothing, only updates the class variables DataManager.emg_data_dict and DataManager.torque_data_dict
        """
        for subject in self.data_struct:
            if subject["SubjectID"] == subject_id:
                for trial in subject["Trials"]:
                    if trial["TrialID"] == trial_id:
                        emg_device = trial["EMGDevice"]
                        num_emg = trial["NumEMG"]
                        frame_freq = trial["FramesPerSec"]
                        analog_freq = trial["AnalogFreq"]

                        for exercise in trial["Exercises"]:
                            trial_exercise_id = trial_id + ' ' + subject_id + ' ' + exercise["ExerciseID"]
                            emg_array_dir = './data/' + trial_exercise_id + '.txt'

                            if 'walk' in exercise["ExerciseID"].lower():
                                t1, t2 = get_fp_time_frame(exercise["File"])

                                # Load emg data
                                if os.path.isfile(emg_array_dir) and not reload:
                                    self.emg_data_dict[trial_exercise_id] = np.loadtxt(emg_array_dir)
                                else:
                                    emg_data = get_emg_from_csv(exercise["File"], emg_device, num_emg, frame_freq,
                                                                analog_freq, t1, t2)
                                    self.emg_data_dict[trial_exercise_id] = emg_data
                                    np.savetxt(emg_array_dir, emg_data, fmt='%f')

                                # Load torque data
                                torque_array_dir = './data/labels/' + trial_exercise_id + '.txt'
                                if os.path.isfile(torque_array_dir) and not reload:
                                    self.torque_data_dict[trial_exercise_id] = np.loadtxt(torque_array_dir)
                                else:
                                    torque_data = get_torque_from_csv(exercise["File"],
                                                                      subject_id + ':RKneeMoment',
                                                                      ['X', 'Y', 'Z'],
                                                                      frame_freq, t1, t2)
                                    self.torque_data_dict[trial_exercise_id] = torque_data
                                    np.savetxt(torque_array_dir, torque_data, fmt='%f')
                            else:
                                if os.path.isfile(emg_array_dir) and not reload:
                                    self.emg_data_dict[trial_exercise_id] = np.loadtxt(emg_array_dir)
                                else:
                                    emg_data = get_emg_from_csv(exercise["File"],
                                                                emg_device, num_emg, frame_freq, analog_freq)
                                    self.emg_data_dict[trial_exercise_id] = emg_data
                                    np.savetxt(emg_array_dir, emg_data, fmt='%f')


# Creates the new subject dictionary including all trials and experiments
def create_subject(path, subject_id):
    trials = []
    trial_folders = Path(path + '\\' + subject_id).glob('*')
    for folder in trial_folders:
        trial_id = str(folder)
        trials.append(create_trial(trial_id, subject_id))

    return {"SubjectID": subject_id, "Trials": trials}


def update_subject(path, subject):
    trial_ids = get_list_of_trial_ids(subject)
    trial_folders = Path(path).glob('*')
    for folder in trial_folders:
        trial_id = folder.stem

        if trial_id not in trial_ids:
            subject["Trials"].append(create_trial(str(folder), subject["SubjectID"]))
        else:
            i = trial_ids.index(trial_id)
            subject["Trials"][i]["Exercises"] = \
                update_exercises(str(folder) + '\\' + subject["SubjectID"], subject["Trials"][i], subject["SubjectID"])

    return subject


def create_trial(path, subject_id):
    with open(path + '\\trial_config.json', 'r') as tc:
        trial = json.load(tc)

    exercises = []
    exercise_files = Path(path + '\\' + subject_id).glob('*.csv')
    for file in exercise_files:
        exercise_name = file.stem.replace(subject_id + ' ', '')
        if 'walk' in exercise_name.lower():
            ex_type = 'inverse_dynamics'
        else:
            ex_type = 'emg_only'
        exercises.append({"ExerciseID": exercise_name, "Type": ex_type, "File": str(file)})

    trial['Exercises'] = exercises

    return trial


def update_exercises(path, trial, subject_id):
    exercise_ids = get_list_of_exercise_ids(trial)
    exercise_files = Path(path).glob('*.csv')
    for file in exercise_files:
        exercise_id = file.stem.replace(subject_id + ' ', '')
        if exercise_id not in exercise_ids:
            if 'walk' in exercise_id.lower():
                ex_type = 'inverse_dynamics'
            else:
                ex_type = 'emg_only'
            trial["Exercises"].append({"ExerciseID": exercise_id, "Type": ex_type, "File": str(file)})

    return trial["Exercises"]


def get_list_of_subject_ids(data_structure):
    subject_ids = []
    for subject in data_structure:
        subject_ids.append(subject["SubjectID"])

    return subject_ids


def get_list_of_trial_ids(subject):
    trial_ids = []
    for trial in subject["Trials"]:
        trial_ids.append(trial["TrialID"])

    return trial_ids


def get_list_of_exercise_ids(trial):
    exercise_ids = []
    for exercise in trial["Exercises"]:
        exercise_ids.append(exercise["ExerciseID"])

    return exercise_ids


def get_subject_structure(data_structure, subject_id):
    for subject in data_structure:
        if subject["SubjectID"] == subject_id:
            return subject


def get_torque_from_csv(file, torque_id, axises, frame_freq, t1, t2):
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

        if t1 and t2:
            start = False
            stop = False
            for line in reader:
                if line[0] == t1 and not start:
                    start = True
                elif line[0] == t2 and start:
                    stop = True
                elif not start:
                    continue

                if len(line) < 10 or stop:
                    f.close()
                    break
                elif start:
                    torque_data.append(line[:2] + line[first_col:first_col + len(axises)])
        else:
            for line in reader:
                if len(line) < 10:
                    f.close()
                    break
                else:
                    torque_data.append(line[:2] + line[first_col:first_col + len(axises)])

    torque_data = np.array(torque_data, dtype=np.float)
    t = (torque_data[:, 0] / frame_freq)

    return np.concatenate((t.reshape(t.shape[0], 1), torque_data[:, 2:]), axis=1)


def get_emg_from_csv(file, emg_device, num_emg, frame_freq, analog_freq, t1=None, t2=None):
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
        # headers = next(reader)
        # headers = ['Time'] + headers[emg_first_col:emg_first_col+num_emg]
        next(reader)
        next(reader)

        if t1 and t2:
            start = False
            stop = False
            for line in reader:
                if line[0] == t1 and not start:
                    start = True
                elif line[0] == t2 and start:
                    stop = True
                elif not start:
                    continue

                if len(line) < num_emg + 1 or stop:
                    f.close()
                    break
                elif start:
                    emg_data.append(line[:2] + line[emg_first_col:emg_first_col + num_emg])
        else:
            for line in reader:
                if len(line) < num_emg + 1:
                    f.close()
                    break
                else:
                    emg_data.append(line[:2] + line[emg_first_col:emg_first_col + num_emg])

    emg_data = np.array(emg_data, dtype=np.float)
    t = (emg_data[:, 0] + (emg_data[:, 1] / (analog_freq / frame_freq))) / frame_freq

    return np.concatenate((t.reshape(t.shape[0], 1), emg_data[:, 2:]), axis=1)


def get_fp_time_frame(file, fp_slack=10**-6):
    """Checks the activity of the force plates and returns the time frame (start frame and end frame) where they are
    active

    :param file: the full path to the csv file
    :param fp_slack: sets the force plate activity threshold value (default 10^-6)
    :return: the strings t1, t2 which are respectively the start frame and the stop frame of force plate activity
    """
    if not os.path.exists(file):
        raise Exception('The file ' + file + ' could not be found!')

    try:
        f = open(file, 'r')
    except IOError:
        print('Could not read file: ', file)
        sys.exit()

    t1 = 0.0
    t2 = 0.0
    with f:
        fl = f.readline()
        while 'Devices' not in fl:
            fl = f.readline()

        f.__next__()

        reader = csv.reader(f, delimiter=',')

        devices = next(reader)
        fp_col = [i for i, s in enumerate(devices) if ' - Force' in s]
        next(reader)
        next(reader)

        start = False
        stop = False
        buffer = 0
        for line in reader:
            fp_val = np.fabs(np.array([line[i:i+3] for i in fp_col], dtype=np.float))

            if stop:
                f.close()
                break
            elif not start and np.any(fp_val > np.ones(fp_val.shape)*fp_slack):
                start = True
                t1 = line[0]
            elif start:
                if np.all(fp_val < np.ones(fp_val.shape)*fp_slack) and buffer > 100:
                    stop = True
                    t2 = str(int(line[0]) + (int(line[1]) > 0))
                else:
                    buffer = buffer + 1
    return t1, t2

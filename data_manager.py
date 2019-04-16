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

        with open('./data_structure.json','r') as ds:
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

    def load_emg(self, subject_id, trial_id, reload=False):
        """Loads the raw emg data either from the original csv file or previously saved txt file.
        If txt file does not already exist then a new one is saved. All emg data (both from csv and txt files) is added
        to the DataManager.emg_data_dict dictionary, each exercise with the key "<TrialID> <SubjectID> <ExerciseID>".
        :param subject_id: the subject id from the data structure (e.g. "Subject01")
        :param trial_id: the trial id from the data structure (e.g. "20190405")
        :param reload: if True then the already saved txt files are overridden with values from the csv
        :return: nothing, only updates the class variable DataManager.emg_data_dict
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
        while not 'Devices' in fl:
            fl = f.readline()

        f.__next__()

        reader = csv.reader(f, delimiter=',')

        first_col = next(reader).index(emg_device)
        headers = next(reader)
        headers = ['Time'] + headers[first_col:first_col+num_emg]
        next(reader)

        for line in reader:
            if len(line) < num_emg + 1:
                f.close()
                break
            else:
                emg_data.append(line[:2] + line[first_col:first_col+num_emg])

    emg_data = np.array(emg_data, dtype=np.float)
    t = (emg_data[:,0] + (emg_data[:,1] / (analog_freq / frame_freq))) / frame_freq

    return np.concatenate((t.reshape(t.shape[0],1), emg_data[:, 2:]), axis=1)

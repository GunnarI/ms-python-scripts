import os
import sys
import csv
import json
from pathlib import Path, PurePath
import numpy as np

class DataManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        with open('./data_structure.json','r') as ds:
            self.data_struct = json.load(ds)

    def update_data_structure(self):
        #with open('./data_structure.json','r+') as ds:
        #    update_ds = json.load(ds)

        subject_folders = Path(self.base_dir).glob('Subject*')
        subject_ids = get_list_of_subject_ids(self.data_struct)

        for folder in subject_folders:
            subject_id = folder.stem
            print(subject_id)

            if subject_id not in subject_ids:
                self.data_struct.append(create_subject(self.base_dir, subject_id))

        with open('./data_structure.json', 'w') as ds:
            json.dump(self.data_struct, ds, indent=4)

    def load_emg(self, subject_id, trial_id):
        for subject in self.data_struct:
            if subject["SubjectID"] == subject_id:
                for trial in subject["Trials"]:
                    if trial["TrialID"] == trial_id:
                        emg_device = trial["EMGDevice"]
                        num_emg = trial["NumEMG"]
                        frame_freq = trial["FramesPerSec"]
                        analog_freq = trial["AnalogFreq"]
                        for exercise in trial["Exercises"]:
                            self.exercise_list
                            get_emg_from_csv(exercise["ExerciseID"], emg_device, num_emg, frame_freq, analog_freq)


#Creates the new subject dictionary including all trials and experiments
def create_subject(path, subject_id):
    trials = []
    trial_folders = Path(path + '\\' + subject_id).glob('*')
    for folder in trial_folders:
        trial_id = str(folder)
        with open(trial_id + '\\trial_config.json', 'r') as tc:
            trial = json.load(tc)

        exercises = []
        exercise_files = Path(trial_id + '\\' + subject_id).glob('*.csv')
        for file in exercise_files:
            exercise_name = file.stem
            if 'walk' in exercise_name.lower():
                type = 'inverse_dynamics'
            else:
                type = 'emg_only'
            exercises.append({"ExerciseID": exercise_name, "Type": type, "File": str(file)})

        trial['Exercises'] = exercises
        trials.append(trial)

    return {"SubjectID": subject_id, "Trials": trials}

#def create_trial()

def get_list_of_subject_ids(data_structure):
    subject_ids = []
    for subject in data_structure:
        subject_ids.append(subject["SubjectID"])

    return subject_ids


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
            if not line:
                f.close()
                break
            else:
                emg_data.append(line[:2] + line[first_col:first_col+num_emg])

    emg_data = np.array(emg_data, dtype=np.float)
    t = (emg_data[:,0] + (emg_data[:,1] / (analog_freq / frame_freq))) / frame_freq

    #return headers, t, emg_data[:,2:]
    return t, emg_data[:, 2:]




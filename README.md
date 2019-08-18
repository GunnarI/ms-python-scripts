# Real-time joint moment prediction using ML models on sEMG
## Description
This repository offers an API for data processing and building a Neural Network (NN) for predicting joint moment from 
EMG data. The code is written and tested only for the case where:
1. Data was exported using 

## Notes and requirements
1. Python 3 needs to be installed. The code was written for Python 3.6.8 and has not been tested with any other version.
2. The tensorflow-gpu package was used. To run the code either the computer running the code needs to have a GPU 
supporting the package (see https://www.tensorflow.org/install/gpu) or use the regular tensorflow package instead.
3. Some of the data management and the NN model cache a lot of data for faster consecutive runs. A designated space of 
around 10Gb is recommended to be able to use caching without problems.
4. The code has only been tested on Windows 10.
5. Vicon Nexus 2.8.2 was used to export data to .csv files. The pipelines used in Vicon Nexus are under the 
nexus_pipelines subfolder. Pipelines that run python scripts need to reference the path to those scripts. The path
in the files in nexus_pipelines has been set to <your-system-specific-path-to-script>.
#### Data storage and structure
To be able to use the data management API (i.e. data_manager.py) the data exported from Vicon Nexus needs to be stored 
in a specific manner for each subject and session. The session_config.json file also needs to be created for each 
session.
Data Structure example:

    +- data_directory
    |   +- Subject01
    |   |   +- 20190601
    |   |   |   +- session_config.json
    |   |   |   +- Subject01
    |   |   |   |   +- Subject01 Walk01.csv
    |   |   |   |   +- Subject01 Walk02.csv
    |   |   |   |   +- ...
    |   |   +- 20190705
    |   |   |   +- session_config.json
    |   |   |   +- Subject01
    |   |   |   |   +- Subject01 Walk01.csv
    |   |   |   |   +- ...
    |   |   +- ...
    |   +- Subject02
    |   +- Subject03
    |   +- ...

Example for **session_config.json**:

    {
        "SessionID": "20190603",
        "FramesPerSec": 100,
        "AnalogFreq": 1000,
        "EMGDevice": "A-Myon EMG - Voltage",
        "NumEMG": 10,
        "EMGProtocol":
        {
            "EMG_A01": "GlutMax",
            "EMG_A02": "VasMed",
            "EMG_A03": "Semitend",
            "EMG_A04": "BicFem",
            "EMG_A05": "GasMed",
            "EMG_A06": "GasLat",
            "EMG_A07": "Soleus",
            "EMG_A08": "TibAnt",
            "EMG_B01": "RectFem",
            "EMG_B02": "VasLat"
        }
    }

## Installation
1. Create a directory on your computer where the project should be stored.
2. Download the project into your directory (use git clone https://github.com/GunnarI/ms-python-scripts.git OR download 
the zip)
3. Navigate to the root directory in the terminal (i.e. cd <path-to-the-directory-you-made>/ms-python-scripts/) and make 
sure python3 command can be used from there. Try running: *python3 --version*
4. Run the project-installation.sh bash script OR run the commands from within it, for a better overview. The commands are:
    * *python3 -m venv ./venv* &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;# To create a virtual environment
    * *source ./venv/Scripts/activate* &nbsp; &nbsp; &nbsp;# To activate the virtual environment
    * *pip install -r ./requirements.txt* &nbsp; &nbsp; &nbsp;# To install all required packages


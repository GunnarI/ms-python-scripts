import data_manager as dm
#import matplotlib as mpl
import matplotlib.pyplot as plt
import filters as filt

my_dm = dm.DataManager(r'C:\Users\Gunnar\Google Drive\MSThesis\Data')
my_dm.update_data_structure()
emg_data_dict = my_dm.load_emg('Subject01','20190405')


for column in emg_data_dict["20190405 Subject01 Walk07"][:,1:].T:
    plt.figure()
    plt.plot(emg_data_dict["20190405 Subject01 Walk07"][:,0],column)

    plt.plot(emg_data_dict["20190405 Subject01 Walk07"][:,0],filt.butter_bandpass_filter(column,20,200,1000,order=5))

plt.show()
import numpy as np
import scipy.signal
import mne

#filtering
def mne_filter(eeg_data):
    ch = ['O2','Oz','O1','P8','P4','Pz','P3','P7','T8','C4','Cz','C3','T7','F4','Fz','F3']
    info = mne.create_info(ch_names=ch,sfreq=125,ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    raw = raw.filter(l_freq=1, h_freq=35,fir_design='firwin',fir_window='hamming')
    raw = raw.notch_filter(freqs=50)
    return raw.get_data()

#compute_stft
def compute_stft(eeg_data):
    fs = 125
    noverlap = 0.5
    window_period = 2*125
    #input : eeg_data (ch,timepoins)
    freq, t, Amp = scipy.signal.stft(eeg_data,fs=fs,nperseg=window_period,noverlap=noverlap*window_period,boundary=None)
    Amp = np.abs(Amp)
    return freq, t, Amp
    
def stft_processed(freq,t,stft_Amp):
    ch_size,_,t_size = stft_Amp.shape
    processed_data = np.empty((ch_size,27,t_size))
    for i in range(ch_size):
        #4-30Hz
        for j in range(4,31):
            count_index = np.where((freq>=j)&(freq<j+1))
            freq_arr = np.empty((len(count_index[0]),10))#(2,timepoints)
            freq_arr = stft_Amp[i][count_index[0]]
            curent_freq = sum(freq_arr)/len(count_index[0])
            processed_data[i][j-4] = np.array(curent_freq)
    return processed_data
    
def pick_data(processed_data):
    ch,freq,t = processed_data.shape
    num = t-20
    #print(t,num)
    np_arr = np.empty((num,ch,freq,20))
    for i in range(num):
        np_arr[i] = processed_data[:,:,i:i+20]
    return np_arr
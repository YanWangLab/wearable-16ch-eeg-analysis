from preprocessing import * 
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, objects, labels):
        self.objects = objects
        self.labels = labels

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        timepoints = self.objects[idx]
        label = self.labels[idx]
        timepoints_tensor = torch.tensor(timepoints.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return timepoints_tensor, label_tensor
    
def data_process(eeg_data,grades):
    filted_data = mne_filter(eeg_data*1e3)
    freq, t, Amp = compute_stft(filted_data)
    processed_data = stft_processed(freq,t,Amp)
    picked_data = pick_data(processed_data)

    #grades
    grade_list = []
    if(isinstance(grades,np.ndarray)):
        #fn
        for i in range(grades.shape[0]):
            grade = int(grades[i])
            grade_list += [grade]*picked_data.shape[0]
    else:
        grade_list = [grades]*picked_data.shape[0]

    return picked_data.reshape(picked_data.shape[0], -1, picked_data.shape[-1]),grade_list
    
def split_one_class(objects, labels, train_ratio=0.7):
    objects = np.asarray(objects)
    labels = np.asarray(labels)
    train_num = int(len(objects) * train_ratio)
    x_train = objects[:train_num]
    y_train = labels[:train_num]
    x_valid = objects[train_num:]
    y_valid = labels[train_num:]
    return x_train, x_valid, y_train, y_valid

def main():
    fn_eeg = np.load(r'./data/train_data/segment_datas/find_numbers_segment.npy')
    rx_eeg = np.load(r'./data/train_data/segment_datas/relax_numbers_segment.npy')
    sl_eeg = np.array(pd.read_csv("./data/train_data/segment_datas/sleep_data.csv"))[:,1:].T
    fn_grades = np.atleast_1d(np.load(r'./data/train_data/segment_datas/fn_grades.npy'))
    fn_eeg_processed,fn_grade_list = data_process(fn_eeg,fn_grades)
    rx_eeg_processed,rx_grade_list = data_process(rx_eeg,20)
    sl_eeg_processed,sl_grade_list = data_process(sl_eeg,0)
    fn_x_train, fn_x_valid, fn_y_train, fn_y_valid = split_one_class(fn_eeg_processed,fn_grade_list,train_ratio=0.7)
    rx_x_train, rx_x_valid, rx_y_train, rx_y_valid = split_one_class(rx_eeg_processed,rx_grade_list,train_ratio=0.7)
    sl_x_train, sl_x_valid, sl_y_train, sl_y_valid = split_one_class(sl_eeg_processed,sl_grade_list,train_ratio=0.7)
    
    x_train = np.concatenate([fn_x_train, rx_x_train, sl_x_train],axis=0)
    y_train = np.concatenate([fn_y_train, rx_y_train, sl_y_train],axis=0)
    x_valid = np.concatenate([fn_x_valid, rx_x_valid, sl_x_valid],axis=0)
    y_valid = np.concatenate([fn_y_valid, rx_y_valid, sl_y_valid],axis=0)
    train_dataset = CustomDataset(x_train, y_train/100)
    valid_dataset = CustomDataset(x_valid, y_valid/100)
    torch.save(train_dataset, r"./data/train_data/processed/train_dataset.pt")
    torch.save(valid_dataset, r"./data/train_data/processed/valid_dataset.pt")

if __name__ == "__main__":
    main()
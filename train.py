import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LSTM_eeg
from build_dataset import CustomDataset
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def compute_r2(y_true, y_pred):
    y_true = torch.cat(y_true, dim=0).view(-1)
    y_pred = torch.cat(y_pred, dim=0).view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot.item() == 0:
        return float("nan")
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def train(model, train_data_loader,valid_data_loader, criterion, optimizer, num_epochs,device,patience=5,min_delta=1e-4):  
    best_valid_r2 = -float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss  = 0
        total = 0

        for features, labels in train_data_loader:
            features = features.to(device)
            labels = labels.to(device).view(-1)
            optimizer.zero_grad() 
            outputs = model(features).view(-1)
            loss = criterion(outputs, labels)  
            total += labels.size(0)  
            loss.backward()  
            optimizer.step()  
            total_loss  += loss.item()
        average_loss = total_loss / len(train_data_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')

        model.eval()
        y_true =[]
        y_pred = []
        with torch.no_grad():
            for features,labels in valid_data_loader:
                features = features.to(device)
                labels = labels.to(device).view(-1)
                outputs = model(features).view(-1)
                y_true.append(labels)
                y_pred.append(outputs)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_avg = y_true.mean()
        y = torch.pow(y_pred-y_true,2).sum()
        y_d = torch.pow(y_avg-y_true,2).sum()

        R_2 = 1 - y/y_d

        if R_2 > best_valid_r2 + min_delta:
            best_valid_r2 = R_2
            stop_count = 0
            torch.save(model.state_dict(), r'./checkpoints/train_demo.pth')
            print(f'New best model saved with best_valid_r2: {best_valid_r2*100}')
        else:
            stop_count+=1
            if stop_count>=patience:
                print(f"Early stopping")
                break
        

def main():
    setup_seed(40)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = torch.load(r"./data/train_data/processed/train_dataset.pt",weights_only=False)
    valid_dataset = torch.load(r"./data/train_data/processed/valid_dataset.pt",weights_only=False)
    train_data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset,batch_size=256,shuffle=False)
    model = LSTM_eeg(input_size=432, hidden_size=32, num_layers=3,output_size=1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model,train_data_loader,valid_data_loader,criterion,optimizer,1000,device)

if __name__ == "__main__":
    main()
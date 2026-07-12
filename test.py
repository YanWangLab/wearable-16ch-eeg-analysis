from model import LSTM_eeg
from preprocessing import * 
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM_eeg(input_size=16*27, hidden_size=32, num_layers=3, output_size=1).to(device)
    params = torch.load(r'./checkpoints/best_model.pth',map_location=device,weights_only=True,)
    model.eval()
    model.load_state_dict(params)
    #model.to(device)
    #load test_data
    test_data = np.load(r"./data/test_data/processed_data/test_data.npy")
    result = []
    with torch.no_grad():
        for i in range(test_data.shape[0]):
            output = model((torch.tensor(test_data[i,:,:,:]).reshape(16*27,20).T).unsqueeze(0).float().to(device))
            result.append(round((output*100).cpu().numpy()[0,0].tolist()))
        plt.plot(result)
        plt.xlabel("Time (sec)")
        plt.ylabel("Attention level")
        plt.savefig(r"./data/test_data/result/test_result.png")
    pd.DataFrame(result).to_csv(r"./data/test_data/result/test_result.csv")
if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LSTM_eeg(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size):
        super(LSTM_eeg, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.Hardtanh = nn.Hardtanh(min_val=0.0, max_val=1.0)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.Hardtanh(out)
        return out
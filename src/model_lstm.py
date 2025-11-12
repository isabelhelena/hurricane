import torch, torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=4, hidden=64, num_layers=1, out_dim=4, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

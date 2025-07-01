# soh_model.py
import torch
import torch.nn as nn
import joblib

class TemporalCNN(nn.Module):
    def __init__(self, in_features, window):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 16, kernel_size=2)
        self.flatten = nn.Flatten()
        conv_out_size = (window - 1 - 1) * 16
        self.fc = nn.Linear(conv_out_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)

def load_model_and_scalers(model_path, scaler_x_path, scaler_y_path):
    checkpoint = torch.load(model_path)
    model = TemporalCNN(in_features=checkpoint['input_features'],
                        window=checkpoint['window_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    window_size = checkpoint['window_size']
    input_dim = checkpoint['input_features']

    return model, scaler_X, scaler_y, window_size, input_dim

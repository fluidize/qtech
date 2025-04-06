import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from rich import print
from tqdm import tqdm
from model_tools import *

class SingleModel(nn.Module):
    def __init__(self, epochs=10, train: bool = True):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if train:
            self.epochs = epochs
            self.data = fetch_data("BTC-USDT", 80, "1min", 10, kucoin=True) #set train data here
            X, y, scalers = prepare_data(self.data, train_split=True) #train split to calculate dims

            input_dim = X.shape[1]
        else:
            input_dim = 33

        # MODEL ARCHITECTURE
        rnn_width = 256
        dense_width = 256
        self.rnn1 = nn.LSTM(input_dim, rnn_width, num_layers=3, bidirectional=True, dropout=0.2)
        self.mha = nn.MultiheadAttention(embed_dim=rnn_width*2, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(rnn_width*2, dense_width)
        self.fc2 = nn.Linear(dense_width, dense_width)
        self.output = nn.Linear(dense_width, 1)

    def forward(self, x):            
        x, _ = self.rnn1(x)
        # x, _ = self.mha(x, x, x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = self.output(x)
        return x

    def train_model(self, model, prompt_save=False, show_loss=False):
        X, y, scalers = prepare_data(self.data, train_split=True)

        model.to(self.DEVICE)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # Reshape y to (batch_size, 1) to remove reduction warning

        train_dataset = TensorDataset(X_tensor, y_tensor)
        
        batch_size = 64
        criterion = nn.MSELoss()
        lr = 1e-5
        l2_reg = 1e-5
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=l2_reg,
                               amsgrad=True,
                               )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        loss_history = []

        for epoch in range(model.epochs):
            model.train()
            total_loss = 0

            print()
            progress_bar = tqdm(total=len(train_loader), desc="EPOCH PROGRESS")
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() 
                progress_bar.update(1)
            progress_bar.close()

            print(f"\nEpoch {epoch + 1}/{model.epochs}, Train Loss: {total_loss:.4f}")
            loss_history.append(total_loss)

            scheduler.step(total_loss)

        if show_loss:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(loss_history))),
                    y=loss_history, mode='lines'
                )
            )
            fig.show()

        if ((input("Save? y/n ").lower() == 'y') if prompt_save else False):
            torch.save(model.state_dict(), input("Enter save path: "))
    
    def predict(self, model, data):
        """
        data should be unprepared with columns ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        X, y, scalers = prepare_data(data)
        
        # Convert to tensor and ensure proper shape and contiguity
        X = torch.tensor(X.values, dtype=torch.float32)
        X = X.contiguous()  # Ensure tensor is contiguous
        
        # Reshape for LSTM: (batch_size, seq_len, input_size)
        if len(X.shape) == 2:
            X = X.unsqueeze(1)  # Add sequence length dimension
            
        # Move to device
        X = X.to(self.DEVICE)

        model.eval()
        model.to(self.DEVICE)
        with torch.no_grad(), autocast('cuda'):
            yhat = model(X)
            yhat = yhat.cpu().numpy()
        
        temp_arr = np.zeros((len(yhat),4))
        temp_arr[:, 3] = yhat.squeeze()
        
        predictions = scalers['price'].inverse_transform(temp_arr)[:, 3]

        return predictions

def load_model(model_path: str):
    import torch
    import os
    
    # Create a new model instance
    model = SingleModel(train=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'),weights_only=True))
    model.eval()
    
    return model

if __name__ == "__main__":
    model = SingleModel(train=True, epochs=100)
    model.train_model(model, prompt_save=True, show_loss=True)
    predictions = model.predict(model, model.data)
    create_plot(model.data, predictions)

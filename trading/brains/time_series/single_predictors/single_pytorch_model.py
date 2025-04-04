import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
import pandas as pd
import numpy as np

from rich import print
from tqdm import tqdm
import os
from model_tools import fetch_data, prepare_data, create_plot

class SingleModel(nn.Module):
    def __init__(self, epochs=10):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[green]USING DEVICE: {self.DEVICE}[/green]")

        self.epochs = epochs

        self.data = fetch_data("BTC-USDT", 10, "1min", 0, kucoin=True) #set train data here
        X, y, scalers = prepare_data(self.data, train_split=True) #train split to calculate dims

        input_dim = X.shape[1]

        # MODEL ARCHITECTURE
        rnn_width = 128
        dense_width = 128
        self.em1 = nn.Embedding(num_embeddings=33, embedding_dim=32)
        self.rnn1 = nn.LSTM(input_dim, rnn_width, num_layers=2, bidirectional=True)
        self.mha = nn.MultiheadAttention(embed_dim=rnn_width*2, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(rnn_width*2, dense_width)
        self.fc2 = nn.Linear(dense_width, dense_width)
        self.output = nn.Linear(dense_width, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)  # Unpack the output and hidden state
        # x, _ = self.mha(x, x, x)
        x = torch.flatten(x, 1) #go thru dense layers
        x = self.fc1(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = self.output(x)
        return x

    def train_model(self, model, save_path="model.pth", prompt_save=False):
        X, y, scalers = prepare_data(self.data, train_split=True)

        model.to(self.DEVICE)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # Reshape y to (batch_size, 1) to remove reduction warning

        train_dataset = TensorDataset(X_tensor, y_tensor)
        
        batch_size = 64
        criterion = nn.MSELoss()
        lr = 1e-6
        l2_reg = 0
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=l2_reg,
                               amsgrad=True,
                               )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_loss = float('inf')
        for epoch in range(model.epochs):
            model.train()  # set model to training mode
            total_loss = 0

            print()
            progress_bar = tqdm(total=len(train_loader), desc="EPOCH PROGRESS")
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)

                optimizer.zero_grad()  # reset the gradients of all parameters before backpropagation
                outputs = model(X_batch)  # forward pass
                loss = criterion(outputs, y_batch)  # Calculate loss
                loss.backward()  # back pass
                optimizer.step()  # update weights

                total_loss += loss.item() 
                progress_bar.update(1)
            progress_bar.close()
            if total_loss < best_loss:
                best_loss = total_loss
                # Save model checkpoint here if needed

            print(f"\nEpoch {epoch + 1}/{model.epochs}, Train Loss: {total_loss:.4f}")

        # Prompt to save the model
        if ((input("Save? y/n ").lower() == 'y') if prompt_save else False):
            torch.save(model, save_path)
            print(f"Model saved to {save_path}")
    
    def predict(self, model , data):
        #input unprepared data
        X, y, scalers = prepare_data(data)
        X = torch.tensor(X.values, dtype=torch.float32).to(self.DEVICE)

        model.eval()
        model.to(self.DEVICE)
        with torch.no_grad(), autocast('cuda'): #automatically set precision since gpu is giving up
            yhat = model(X)
            yhat = yhat.cpu().numpy()
        
        temp_arr = np.zeros((len(yhat),4))
        temp_arr[:, 3] = yhat.squeeze()
        
        predictions = scalers['price'].inverse_transform(temp_arr)[:, 3]

        return predictions


if __name__ == "__main__":
    model = SingleModel(epochs=100)
    model.train_model(model, prompt_save=True)
    predictions = model.predict(model, model.data[:10000])
    create_plot(model.data, predictions)

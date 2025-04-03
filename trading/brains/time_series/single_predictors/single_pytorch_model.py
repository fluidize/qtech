import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from rich import print
from tqdm import tqdm

from price_data import fetch_data, prepare_data

class SingleModel(nn.Module):
    def __init__(self, batch_size=64, learning_rate=0.001, epochs=10):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[green]USING DEVICE: {self.DEVICE}[/green]")
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.data = fetch_data("BTC-USDT", 10, "1min", 0, kucoin=True) #set train data here
        X, y, scalers = prepare_data(self.data, train_split=True) #train split to calculate dims

        input_dim = X.shape[1]

        # MODEL ARCHITECTURE
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train_model(self, model, save_path, prompt_save=False):
        X, y, scalers = prepare_data(self.data, train_split=True)

        model.to(self.DEVICE)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # Reshape y to (batch_size, 1) to remove reduction warning

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

        for epoch in range(model.epochs):
            model.train()  # set model to training mode
            total_loss = 0

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
        with torch.no_grad():
            yhat = model(X)
            yhat = yhat.cpu().numpy()
        
        temp_arr = np.zeros((len(yhat),4))
        temp_arr[:, 3] = yhat.squeeze()
        
        predictions = scalers['price'].inverse_transform(temp_arr)[:, 3]

        return predictions


if __name__ == "__main__":
    model = SingleModel(batch_size=128, learning_rate=0.0005, epochs=1)
    model.train_model(model, "trained_model.pth")
    predictions = model.predict(model, fetch_data("BTC-USDT", 10, "1min", 0, kucoin=True))
    print(predictions)

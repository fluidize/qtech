import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from price_data import fetch_data, prepare_data


class SimpleNN(nn.Module):
    def __init__(self, batch_size=64, learning_rate=0.001, epochs=10):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Prepare data
        self.X, self.y = prepare_data(fetch_data("SOL-USDT", 10, "1min", 0, kucoin=True), train_split=True)
        
        # Print some data info
        print(self.X.head())
        print(self.X.shape)
        
        # Input dimension is based on the number of features
        input_dim = self.X.shape[1]

        # Define network layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train_model(self, model, save_path):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(DEVICE)

        # Convert the data to tensors
        X_tensor = torch.tensor(self.X.values, dtype=torch.float32)
        y_tensor = torch.tensor(self.y.values, dtype=torch.float32)

        # Create a DataLoader for the training set
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()  # Using MSELoss for regression
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

        for epoch in range(model.epochs):
            model.train()  # Set model to training mode
            total_loss = 0

            # Training loop
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                optimizer.zero_grad()  # Zero the gradients
                outputs = model(X_batch)  # Forward pass
                loss = criterion(outputs, y_batch)  # Calculate loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                total_loss += loss.item()  # Accumulate total loss for this epoch

            # Print training loss for each epoch
            print(f"Epoch {epoch + 1}/{model.epochs}, Train Loss: {total_loss:.4f}")

        # Prompt to save the model
        if input("Save? y/n ").lower() == 'y':
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":
    model = SimpleNN(batch_size=128, learning_rate=0.0005, epochs=5)
    model.train_model(model, "trained_model.pth")

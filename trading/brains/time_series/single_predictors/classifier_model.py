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

import sys
sys.path.append(r"trading")
import model_tools as mt

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights

class ClassifierModel(nn.Module):
    def __init__(self, ticker: str = None, chunks: int = None, interval: str = None, age_days: int = None, epochs=10, train: bool = True):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs

        if train:
            self.data = mt.fetch_data(ticker, chunks, interval, age_days, kucoin=True)
            X, y = mt.prepare_data_classifier(self.data, train_split=True)
            input_dim = X.shape[1]
        else:
            input_dim = 33

        # Enhanced MODEL ARCHITECTURE
        rnn_width = 256
        dense_width = 256
        
        # Feature processing
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, rnn_width, num_layers=2, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(rnn_width*2, rnn_width, num_layers=2, bidirectional=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = Attention(rnn_width*2)
        
        # Dense layers with residual connections
        self.fc1 = nn.Linear(rnn_width*2, dense_width)
        self.fc2 = nn.Linear(dense_width, dense_width)
        self.fc3 = nn.Linear(dense_width, dense_width)
        
        # Output layer
        self.output = nn.Linear(dense_width, 3)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        # Feature processing
        x = self.batch_norm(x)
        
        # LSTM layers
        lstm_out1, _ = self.lstm1(x)  # Add sequence length dimension
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Attention mechanism
        # attention_weights = self.attention(lstm_out2)
        # attended = torch.sum(attention_weights * lstm_out2, dim=1)
        
        # Dense layers with residual connections
        x = self.fc1(lstm_out2)  # Use attended instead of lstm_out2
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        residual = x
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Add residual connection
        x = x + residual
        
        # Output layer
        x = self.output(x)
        return x

    def train_model(self, model, prompt_save=False, show_loss=False):
        X, y = mt.prepare_data_classifier(self.data, train_split=True)
        print("Class distribution:", y.value_counts())

        model.to(self.DEVICE)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        # Calculate class weights for imbalanced data
        class_counts = torch.bincount(y_tensor)
        class_weights = 1. / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.DEVICE)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        
        batch_size = 32  # Reduced batch size for better generalization
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        lr = 1e-4  # Reduced learning rate
        l2_reg = 1e-5
        optimizer = optim.AdamW(model.parameters(),
                              lr=lr,
                              weight_decay=l2_reg,
                              amsgrad=True)
        
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=lr,
                                                epochs=model.epochs,
                                                steps_per_epoch=(len(train_dataset)//batch_size)+1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        loss_history = []
        best_loss = float('inf')

        print(y)

        for epoch in range(model.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            print()
            progress_bar = tqdm(total=len(train_loader), desc="EPOCH PROGRESS")
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                progress_bar.update(1)
            progress_bar.close()

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            print(f"\nEpoch {epoch + 1}/{model.epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            loss_history.append(epoch_loss)

            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')

        if show_loss:
            fig = mt.loss_plot(loss_history)
            fig.show()

        if ((input("Save? y/n ").lower() == 'y') if prompt_save else False):
            torch.save(model.state_dict(), input("Enter save path: "))
    
    def predict(self, model, data):
        X, y = mt.prepare_data_classifier(data, train_split=True)
        
        X = torch.tensor(X.values, dtype=torch.float32).contiguous().to(self.DEVICE)

        model.eval()
        model.to(self.DEVICE)
        with torch.no_grad(), autocast('cuda'):
            logits = model(X)
            logits = logits.cpu()
        probabilities = torch.softmax(logits, dim=1)
        predictions = np.argmax(probabilities, axis=1)

        return predictions
    
    def prediction_plot(self, data, predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual'))
        
        up_x = []
        up_y = []
        down_x = []
        down_y = []
        for i in range(len(predictions)):
            if predictions[i] == 2:  # Up prediction
                up_x.append(data.index[i])
                up_y.append(data['Close'][i])
            elif predictions[i] == 0:  # Down prediction
                down_x.append(data.index[i])
                down_y.append(data['Close'][i])

        fig.add_trace(go.Scatter(x=up_x, y=up_y, mode='markers', name='Up Predictions', 
                                marker=dict(color='green', size=8, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=down_x, y=down_y, mode='markers', name='Down Predictions', 
                                marker=dict(color='red', size=8, symbol='triangle-down')))
        fig.show()

def load_model(model_path: str):
    import torch
    import os
    
    model = ClassifierModel(train=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'),weights_only=True))
    model.eval()
    
    return model

if __name__ == "__main__":
    model = ClassifierModel(ticker="SOL-USDT", chunks=3, interval="1min", age_days=10, epochs=50)
    model.train_model(model, prompt_save=False, show_loss=True)
    predictions = model.predict(model, model.data)
    print(predictions)
    model.prediction_plot(model.data, predictions)

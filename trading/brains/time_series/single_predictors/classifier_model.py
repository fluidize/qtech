import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from rich import print
from tqdm import tqdm

import sys
sys.path.append(r"trading")
import model_tools as mt

class FeatureSelectionCallback:
    """Callback for feature importance-based selection during training"""
    def __init__(self, X_train, y_train, feature_names, top_n=20):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.top_n = top_n
        self.important_features = None
        
    def get_important_features(self):
        import lightgbm as lgb
        start_time = time.time()
        gbm = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            force_col_wise=True,
            verbose=-1
        )
        gbm.fit(self.X_train, self.y_train)
        end_time = time.time()
        importances = gbm.feature_importances_
        indices = np.argsort(importances)[::-1]

        self.important_features = [self.feature_names[i] for i in indices[:self.top_n]]
        importance_values = [importances[i] for i in indices[:self.top_n]]
        
        print(f"Top features selected by importance ({end_time - start_time:.2f}s):")
        for i, (feature, importance) in enumerate(zip(self.important_features, importance_values)):
            print(f"{i+1}. {feature}: {importance:.2f}")
            
        return self.important_features

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

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.LayerNorm(out_features)
        self.lin2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.LayerNorm(out_features)
        
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        residual = x
        
        out = self.lin1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        
        out = self.lin2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.leaky_relu(out, 0.2)
        
        return out

class ClassifierModel(nn.Module):
    def __init__(self, ticker: str = None, chunks: int = None, interval: str = None, age_days: int = None, epochs=10, train: bool = True, pct_threshold=0.01, lagged_length=10, input_dim=None, use_feature_selection: bool = True):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.epochs = epochs
        self.pct_threshold = pct_threshold
        self.feature_selector = None
        self.lagged_length = lagged_length

        if train:
            self.data = mt.fetch_data(ticker, chunks, interval, age_days, kucoin=True)
            X, y = mt.prepare_data_classifier(self.data, pct_threshold=self.pct_threshold, lagged_length=lagged_length)
            feature_names = X.columns.tolist()
            
            if use_feature_selection:
                self.feature_selector = FeatureSelectionCallback(X.values, y.values, feature_names, top_n=30)
                important_feature_names = self.feature_selector.get_important_features()
                X = X[important_feature_names]
            else:
                print("Feature selection is skipped.")
            
            self.input_dim = X.shape[1]
        else:
            self.input_dim = input_dim
            if self.input_dim is None:
                raise ValueError("Input dimension must be provided if model is loaded.")
        
        self.hidden_dim = 512
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,  # Use input_dim as the input size
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        self.attention = Attention(self.hidden_dim * 2)
        self.res_block1 = ResidualBlock(self.hidden_dim * 2, self.hidden_dim)
        self.res_block2 = ResidualBlock(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim)        
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 3)
        
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.batch_norm = nn.BatchNorm1d(self.input_dim)
        self.instance_norm = nn.InstanceNorm1d(self.input_dim)

        self.dropout = nn.Dropout(0.2)
        
        self._init_weights()

    def forward(self, x):
        x = self.batch_norm(x)
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        
        # attention_weights = self.attention(lstm_out)
        # context = torch.sum(attention_weights * lstm_out, dim=1)
        
        x = self.res_block1(lstm_out)
        # x = self.res_block2(x)
        
        # x = F.leaky_relu(self.fc_out(x), 0.2)

        x = self.fc1(lstm_out)
        x = self.fc2(x)

        x = self.output(x)
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def train_model(self, model, prompt_save=False, show_loss=False):
        X, y = mt.prepare_data_classifier(self.data, pct_threshold=self.pct_threshold, lagged_length=self.lagged_length)
        print(X)
        
        if self.feature_selector and self.feature_selector.important_features:
            X = X[self.feature_selector.important_features]
            self.input_dim = X.shape[1]
        
        # Print class distribution
        print("Class distribution: \n", y.value_counts())
        total_samples = len(y)
        for class_label, count in y.value_counts().items():
            percentage = (count/total_samples) * 100
            print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        print(f"Feature count: {X.shape[1]}")
        
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_val = X_val.apply(pd.to_numeric, errors='coerce')
        
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        model = model.to(self.DEVICE)
        
        class_counts = np.bincount(y_train)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.DEVICE)
        
        print("Class weights:", class_weights.cpu().numpy())
        
        batch_size = 128
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        base_lr = 1e-3
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            amsgrad=True,
            weight_decay=1e-6
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        model = model.to(self.DEVICE)

        best_state_dict = None
        best_val_loss = float('inf')
        
        for epoch in range(model.epochs):
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            print()
            progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{model.epochs}")
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)
                
                optimizer.zero_grad()
                
                outputs = model(X_batch)

                # L1 and L2 reg
                l1_lambda = 1e-6
                l2_lambda = 1e-6

                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

                loss = criterion(outputs, y_batch) + l1_lambda * l1_norm + l2_lambda * l2_norm
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()

                total_train_loss += loss.item()
                
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()
                
                progress_bar.update(1)
            
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            
            lr_scheduler.step()
            progress_bar.close()
            
            train_loss = total_train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            
            print(f"\nEpoch {epoch + 1}/{model.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()

        if show_loss:
            fig = self.loss_plot(train_loss_history, train_acc_history, val_loss_history, val_acc_history)
            fig.show()
        
        if ((input("Save? y/n ").lower() == 'y') if prompt_save else False):
            save_path = f"{self.ticker}_{self.interval}_{self.pct_threshold}_{self.lagged_length}.pth"
            torch.save({
                'model_state_dict': best_state_dict,
                'selected_features': self.feature_selector.important_features if self.feature_selector else None  # Save selected features
            }, save_path)
        
        return model
    
    def prediction_plot(self, data, predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
        
        confidence_threshold = 0.9
        high_confidence_mask = np.zeros(len(predictions), dtype=bool)
        
        if hasattr(self, 'prediction_probabilities'):
            max_probs = np.max(self.prediction_probabilities, axis=1)
            high_confidence_mask = max_probs > confidence_threshold
        
        up_x, up_y = [], []
        down_x, down_y = [], []
        hold_x, hold_y = [], []
        
        hc_up_x, hc_up_y = [], []
        hc_down_x, hc_down_y = [], []
        
        for i in range(len(predictions)):
            if predictions[i] == 2:  # Buy signal
                if high_confidence_mask[i]:
                    hc_up_x.append(data.index[i])
                    hc_up_y.append(data['Close'][i])
                else:
                    up_x.append(data.index[i])
                    up_y.append(data['Close'][i])
            elif predictions[i] == 0:  # Sell signal
                if high_confidence_mask[i]:
                    hc_down_x.append(data.index[i])
                    hc_down_y.append(data['Close'][i])
                else:
                    down_x.append(data.index[i])
                    down_y.append(data['Close'][i])
            else:  # Hold signal
                hold_x.append(data.index[i])
                hold_y.append(data['Close'][i])
        
        fig.add_trace(go.Scatter(x=up_x, y=up_y, mode='markers', name='Buy', 
                                marker=dict(color='green', size=8, symbol='triangle-up', opacity=0.6)))
        
        fig.add_trace(go.Scatter(x=down_x, y=down_y, mode='markers', name='Sell', 
                                marker=dict(color='red', size=8, symbol='triangle-down', opacity=0.6)))
        
        if hasattr(self, 'prediction_probabilities'):
            fig.add_trace(go.Scatter(x=hc_up_x, y=hc_up_y, mode='markers', name='High Conf Buy', 
                                    marker=dict(color='green', size=10, symbol='triangle-up', line=dict(width=2, color='white'))))
            
            fig.add_trace(go.Scatter(x=hc_down_x, y=hc_down_y, mode='markers', name='High Conf Sell', 
                                    marker=dict(color='red', size=10, symbol='triangle-down', line=dict(width=2, color='white'))))
        
        fig.update_layout(
            title='Price Action with Model Predictions',
            xaxis_title='Time',
            yaxis_title='Price',
            template="plotly_dark"
        )
        
        fig.show()
        return fig

    def loss_plot(self, train_loss, train_acc=None, val_loss=None, val_acc=None):
        """
        Create a plot showing training and validation loss and accuracy over epochs.
        
        Args:
            train_loss (list): List of training loss values
            train_acc (list, optional): List of training accuracy values
            val_loss (list, optional): List of validation loss values
            val_acc (list, optional): List of validation accuracy values
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(train_loss) + 1)),
                y=train_loss,
                mode='lines',
                name='Training Loss',
                line=dict(color='red')
            )
        )
        
        if val_loss is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(val_loss) + 1)),
                    y=val_loss,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red', dash='dash')
                )
            )
        
        if train_acc is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(train_acc) + 1)),
                    y=train_acc,
                    mode='lines',
                    name='Training Accuracy',
                    line=dict(color='green'),
                    yaxis='y2'
                )
            )
        
        if val_acc is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(val_acc) + 1)),
                    y=val_acc,
                    mode='lines',
                    name='Validation Accuracy',
                    line=dict(color='green', dash='dash'),
                    yaxis='y2'
                )
            )
        
        fig.update_layout(
            title='Training and Validation Metrics',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            yaxis2=dict(
                title='Accuracy (%)',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            template="plotly_dark",
            showlegend=True
        )
        
        return fig

def load_model(model_path: str, input_dim: int = None):
    model = ClassifierModel(train=False, input_dim=input_dim)  # Initialize the model
    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.selected_features = checkpoint['selected_features']  # Load selected features
    
    return model #return raw class

if __name__ == "__main__":
    model = ClassifierModel(ticker="BTC-USDT", chunks=1, interval="1min", age_days=0, epochs=50, pct_threshold=0.1, lagged_length=20, use_feature_selection=True)
    model = model.train_model(model, prompt_save=True, show_loss=True)
    with torch.no_grad():
        predictions = model(mt.prepare_data_classifier(model.data, pct_threshold=model.pct_threshold, lagged_length=model.lagged_length))
        
    model.prediction_plot(model.data, predictions)

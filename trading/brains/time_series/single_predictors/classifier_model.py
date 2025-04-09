import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier

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
        # Use Random Forest for feature selection
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        self.important_features = [self.feature_names[i] for i in indices[:self.top_n]]
        importance_values = [importances[i] for i in indices[:self.top_n]]
        
        print("Top features selected by importance:")
        for i, (feature, importance) in enumerate(zip(self.important_features, importance_values)):
            print(f"{i+1}. {feature}: {importance:.4f}")
            
        return self.important_features

class ClassifierModel(nn.Module):
    def __init__(self, ticker: str = None, chunks: int = None, interval: str = None, age_days: int = None, epochs=10, train: bool = True, pct_threshold=0.01, use_feature_selection: bool = True):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.pct_threshold = pct_threshold
        self.feature_selector = None
        
        # Dynamically determine input dimension from data
        if train:
            self.data = mt.fetch_data(ticker, chunks, interval, age_days, kucoin=True)
            X, y = mt.prepare_data_classifier(self.data, train_split=True, pct_threshold=self.pct_threshold, lagged_length=10)
            
            feature_names = X.columns.tolist()
            
            # Conditional feature selection
            if use_feature_selection:
                self.feature_selector = FeatureSelectionCallback(X.values, y.values, feature_names, top_n=30)
                important_feature_names = self.feature_selector.get_important_features()
                X = X[important_feature_names]
            else:
                print("Feature selection is skipped.")
            
            input_dim = X.shape[1]
        else:
            input_dim = 10  # Set a default input dimension for inference
        
        self.input_dim = input_dim
        self.hidden_dim = 128  # Reduced hidden dimension
        
        # Removed feature extractor
        # Directly using LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,  # Use input_dim directly
            hidden_size=self.hidden_dim,
            num_layers=1,  # Reduced to 1 layer
            batch_first=True
        )
        
        self.fc_block = nn.Sequential( #shrinking layers
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.GELU()
        )

        self.output = nn.Linear(self.hidden_dim // 8, 3)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_val_accuracy = 0.0

    def forward(self, x):
        batch_size = x.size(0)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        x = lstm_out.squeeze(1)
        
        x = self.fc_block(x)
        x = self.output(x)
        return x

    def train_model(self, model, prompt_save=False, show_loss=False, val_split=0.2):
        X, y = mt.prepare_data_classifier(self.data, train_split=True, pct_threshold=self.pct_threshold, lagged_length=10)
        
        if self.feature_selector and self.feature_selector.important_features:
            X = X[self.feature_selector.important_features]
        
        print("Class distribution: \n", y.value_counts())
        print(f"Feature count: {X.shape[1]}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values, 
            test_size=val_split, 
            random_state=42,
            stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        model.to(self.DEVICE)
        
        class_counts = np.bincount(y_train)
        class_weights = 1. / class_counts.astype(np.float32)
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, device=self.DEVICE)
        
        batch_size = 64
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        base_lr = 3e-4
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=1e-4,
            amsgrad=True
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # First restart after 10 epochs
            T_mult=2,  # Double the interval after each restart
            eta_min=1e-3  # Minimum learning rate
        )
        
        scaler = GradScaler()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        patience = 15
        
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
                
                # Use mixed precision for faster training
                with autocast('cuda'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_train_loss += loss.item()
                
                with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()
                
                progress_bar.update(1)
            
            progress_bar.close()
            
            model.eval()
            total_val_loss = 0
            val_correct = 0
            val_total = 0
            
            val_class_correct = [0, 0, 0]  # for classes 0, 1, 2
            val_class_total = [0, 0, 0]
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    total_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
                    
                    for i in range(3):
                        mask = (y_batch == i)
                        val_class_total[i] += mask.sum().item()
                        if mask.sum() > 0:
                            val_class_correct[i] += (predicted[mask] == i).sum().item()
            
            train_loss = total_train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            class_accuracies = [100 * correct / total if total > 0 else 0 for correct, total in zip(val_class_correct, val_class_total)]
            
            print(f"\nEpoch {epoch+1}/{model.epochs} (LR: {current_lr:.6f})")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Class Accuracies: Sell: {class_accuracies[0]:.2f}%, Hold: {class_accuracies[1]:.2f}%, Buy: {class_accuracies[2]:.2f}%")
            
            if val_acc > model.best_val_accuracy:
                model.best_val_accuracy = val_acc
                model.patience_counter = 0
                torch.save(model.state_dict(), 'best_acc_model.pth')
                print("✓ Saved new best accuracy model")
            elif val_loss < model.best_val_loss:
                model.best_val_loss = val_loss
                model.patience_counter = 0
                torch.save(model.state_dict(), 'best_loss_model.pth')
                print("✓ Saved new best loss model")
            else:
                model.patience_counter += 1
            
            if model.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if show_loss:
            fig = make_loss_plot(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
            fig.show()
        
        if ((input("Save? y/n ").lower() == 'y') if prompt_save else False):
            save_path = input("Enter save path: ")
            torch.save(model.state_dict(), save_path)
        
        # Load the best model for evaluation
        try:
            model.load_state_dict(torch.load('best_acc_model.pth'))
            print("Loaded best accuracy model for evaluation")
        except:
            try:
                model.load_state_dict(torch.load('best_loss_model.pth'))
                print("Loaded best loss model for evaluation")
            except:
                print("No saved model found, using current model state")
        
        return model
    
    def predict(self, model, data):
        X, y = mt.prepare_data_classifier(data, train_split=True, pct_threshold=self.pct_threshold, lagged_length=10)
        
        # Use the same features as during training if feature selection was performed
        if self.feature_selector and self.feature_selector.important_features:
            # Check if all important features exist in X
            available_features = set(X.columns)
            required_features = set(self.feature_selector.important_features)
            missing_features = required_features - available_features
            
            if missing_features:
                print(f"Warning: Missing {len(missing_features)} features that were used during training.")
                available_important_features = [f for f in self.feature_selector.important_features if f in available_features]
                X = X[available_important_features]
            else:
                X = X[self.feature_selector.important_features]
        
        X = torch.tensor(X.values, dtype=torch.float32).contiguous().to(self.DEVICE)
        
        model.eval()
        model.to(self.DEVICE)
        
        # Process in batches to avoid memory issues with large datasets
        batch_size = 256
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad(), autocast('cuda'):
            for batch in dataloader:
                batch_X = batch[0]
                outputs = model(batch_X)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_probabilities.append(probabilities.cpu().numpy())
        
        # Store probabilities for potential confidence-based filtering
        self.prediction_probabilities = np.vstack(all_probabilities)
        
        return np.array(all_predictions)
    
    def prediction_plot(self, data, predictions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
        
        # Filter predictions by confidence if available
        confidence_threshold = 0.6
        high_confidence_mask = np.zeros(len(predictions), dtype=bool)
        
        if hasattr(self, 'prediction_probabilities'):
            # Find predictions with high confidence
            max_probs = np.max(self.prediction_probabilities, axis=1)
            high_confidence_mask = max_probs > confidence_threshold
        
        # Separate by prediction type and confidence
        up_x, up_y = [], []
        down_x, down_y = [], []
        hold_x, hold_y = [], []
        
        # High confidence predictions
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
        
        # Regular signals
        fig.add_trace(go.Scatter(x=up_x, y=up_y, mode='markers', name='Buy', 
                                marker=dict(color='green', size=8, symbol='triangle-up', opacity=0.6)))
        
        fig.add_trace(go.Scatter(x=down_x, y=down_y, mode='markers', name='Sell', 
                                marker=dict(color='red', size=8, symbol='triangle-down', opacity=0.6)))
        
        # High confidence signals (if available)
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

def make_loss_plot(train_loss, val_loss=None, train_acc=None, val_acc=None):
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss During Training", "Accuracy During Training"))
    
    # Plot training loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_loss) + 1)),
            y=train_loss,
            mode='lines',
            name='Training Loss'
        ),
        row=1, col=1
    )
    
    # Plot validation loss if provided
    if val_loss is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_loss) + 1)),
                y=val_loss,
                mode='lines',
                name='Validation Loss'
            ),
            row=1, col=1
        )
    
    # Plot accuracy
    if train_acc is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(train_acc) + 1)),
                y=train_acc,
                mode='lines',
                name='Training Accuracy'
            ),
            row=2, col=1
        )
    
    if val_acc is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(val_acc) + 1)),
                y=val_acc,
                mode='lines',
                name='Validation Accuracy'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=800,
        template="plotly_dark"
    )
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    
    return fig

def load_model(model_path: str, pct_threshold=0.01):
    model = ClassifierModel(train=False, pct_threshold=pct_threshold)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device('cuda'), weights_only=True)
    
    # Get input dimension from first layer weights
    if 'feature_extractor.0.weight' in state_dict:
        input_dim = state_dict['feature_extractor.0.weight'].shape[1]
        model.input_dim = input_dim
        print(f"Model loaded with input dimension: {input_dim}")
        
        # Reinitialize model with correct input dim
        model = ClassifierModel(train=False, pct_threshold=pct_threshold)
        model.input_dim = input_dim
        
        # Update feature extraction layer
        model.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, model.hidden_dim),
            nn.ReLU()  # Changed to ReLU for simplicity
        )
    
    # Now load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

if __name__ == "__main__":
    model = ClassifierModel(ticker="SOL-USDT", chunks=1, interval="1min", age_days=0, epochs=100, pct_threshold=0.01, use_feature_selection=False)
    model = model.train_model(model, prompt_save=False, show_loss=True)
    predictions = model.predict(model, model.data)
    print(predictions)
    model.prediction_plot(model.data, predictions)

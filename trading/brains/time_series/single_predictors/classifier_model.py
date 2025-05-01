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

class FeatureSelector:
    """Callback for feature importance-based selection during training"""
    def __init__(self, X_train, y_train, feature_names, importance_threshold=5, max_features=None):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.importance_threshold = importance_threshold
        self.max_features = max_features
        self.important_features = None
        
    def get_important_features(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        import numpy as np
        start_time = time.time()

        # Train LightGBM for its feature importance
        gbm = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            max_depth=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            lambda_l1=0.1,
            lambda_l2=0.1,
            verbose=-1,
            random_state=42
        )
        gbm.fit(self.X_train, self.y_train)
        lgb_importances = gbm.feature_importances_

        # Train RandomForest for its feature importance
        RFClassifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        RFClassifier.fit(self.X_train, self.y_train)
        rf_importances = RFClassifier.feature_importances_

        combined_importances = (np.array(rf_importances) + np.array(lgb_importances)) / 2
        indices = np.argsort(combined_importances)[::-1]  # reverse order

        features = [self.feature_names[i] for i in indices]
        importance_values = [combined_importances[i] for i in indices]
        
        candidate_features = [f for f, imp in zip(features, importance_values) if imp >= self.importance_threshold]
        if self.max_features and len(candidate_features) > self.max_features:
            candidate_features = candidate_features[:self.max_features]
            
        # Check for harmful feature correlations
        X_candidates = self.X_train[candidate_features]
        correlation_matrix = X_candidates.corr().abs()
        
        # Remove highly correlated features (>0.95)
        to_drop = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] > 0.95:
                    colname = correlation_matrix.columns[i]
                    to_drop.add(colname)
        
        candidate_features = [f for f in candidate_features if f not in to_drop]
        print(f"Removed {len(to_drop)} highly correlated features")
        
        # Incremental feature validation
        final_features = []
        best_score = 0
        scaler = StandardScaler()
        
        # First, evaluate baseline model with most important feature
        X_scaled = scaler.fit_transform(self.X_train[[candidate_features[0]]])
        baseline_model = lgb.LGBMClassifier(random_state=42)
        baseline_score = cross_val_score(baseline_model, X_scaled, self.y_train, cv=3, scoring='accuracy').mean()
        final_features = [candidate_features[0]]
        best_score = baseline_score
        
        print(f"Starting with feature: {candidate_features[0]}, baseline score: {baseline_score:.4f}")
        
        # Now test adding each feature incrementally
        for feature in candidate_features[1:]:
            current_features = final_features + [feature]
            X_scaled = scaler.fit_transform(self.X_train[current_features])
            
            model = lgb.LGBMClassifier(random_state=42)
            score = cross_val_score(model, X_scaled, self.y_train, cv=3, scoring='accuracy').mean()
            
            # Only keep features that improve or maintain performance 
            # (allowing small degradation of up to 0.5% to avoid overfitting)
            if score >= best_score - 0.005:
                final_features.append(feature)
                if score > best_score:
                    best_score = score
                print(f"Added feature: {feature}, new score: {score:.4f}")
            else:
                print(f"Rejected feature: {feature}, score: {score:.4f} vs best: {best_score:.4f}")
                
        end_time = time.time()
        self.important_features = final_features
        
        # Print final features with their importance values
        filtered_importances = []
        for feature in self.important_features:
            idx = features.index(feature)
            imp = importance_values[idx]
            filtered_importances.append(imp)
        
        print(f"\nFinal features selected ({end_time - start_time:.2f}s):")
        for i, (feature, importance) in enumerate(zip(self.important_features, filtered_importances)):
            feature_idx = self.feature_names.index(feature)
            feature_max_value = max(self.X_train.iloc[:, feature_idx])
            feature_min_value = min(self.X_train.iloc[:, feature_idx])
            print(f"{i+1}. {feature}: {importance}")
            
        return self.important_features

class NormalizationBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__() 
        self.normalizer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.normalizer(x)
        return x

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
    def __init__(self, ticker: str = None, chunks: int = None, interval: str = None, age_days: int = None, epochs=10, train: bool = True, lagged_length=20, use_feature_selection: bool = True, importance_threshold: int = 30, max_features: int = 50, input_dim=None):
        super().__init__()

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ticker = ticker
        self.chunks = chunks
        self.interval = interval
        self.age_days = age_days
        self.epochs = epochs
        self.selected_features = None
        self.lagged_length = lagged_length
        self.importance_threshold = importance_threshold
        self.max_features = max_features
        self.input_dim = input_dim  # Allow input_dim to be set directly
        self.hidden_dim = 256
        

        if train:
            self.data = mt.fetch_data(ticker, chunks, interval, age_days, kucoin=True)
            X, y = mt.prepare_data_classifier(self.data, lagged_length=lagged_length)
            feature_names = X.columns.tolist()
            
            if use_feature_selection:
                self.feature_selector = FeatureSelector(X_train=X, y_train=y, feature_names=feature_names, 
                                                      importance_threshold=importance_threshold, max_features=max_features)
                self.selected_features = self.feature_selector.get_important_features()
                X = X[self.selected_features]
            else:
                print("Feature selection is skipped.")
            
            self.input_dim = X.shape[1]

        # Only initialize layers if input_dim is known
        if self.input_dim is not None:
            self._build_model()
        else:
            print("Warning: input_dim is None. Model layers will not be initialized.")
    
    def _build_model(self):
        """Build the model layers using the input_dim"""
        # Use LayerNorm - batch independent and effective for time series data
        self.normalization_block = NormalizationBlock(self.input_dim, self.input_dim, hidden_dim=128)

        self.lstm = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        self.res_block1 = ResidualBlock(self.hidden_dim * 2, self.hidden_dim)
        self.res_block2 = ResidualBlock(self.hidden_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 2)
        
        self._init_weights()

    def forward(self, x):
        x_norm = self.normalization_block(x)

        lstm_out, _ = self.lstm(x_norm)
        
        x = self.res_block1(lstm_out)
        x = self.res_block2(x)
        
        # Use training=self.training to respect model mode
        x = F.dropout(self.fc1(x), 0.1, training=self.training)
        x = F.dropout(self.fc2(x), 0.1, training=self.training)
            
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
        X, y = mt.prepare_data_classifier(self.data, lagged_length=self.lagged_length)
        
        if self.feature_selector and self.feature_selector.important_features:
            X = X[self.feature_selector.important_features]
            self.input_dim = X.shape[1]
        
        total_samples = len(y)
        for class_label, count in y.value_counts().items():
            percentage = (count/total_samples) * 100
            print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
        
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
        
        batch_size = 64
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
            
            progress_bar = tqdm(total=len(train_loader), 
                              desc=f"Epoch {epoch+1}/{model.epochs}",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                              leave=False,
                              position=0,
                              dynamic_ncols=True)
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)
                
                optimizer.zero_grad()
                
                outputs = model(X_batch)

                # L1 and L2 regularization
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
                
                train_acc = 100 * train_correct / train_total
                avg_loss = total_train_loss / (progress_bar.n + 1)
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{train_acc:.2f}%',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                progress_bar.update(1)
            progress_bar.close()
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
            
            print(f"Epoch {epoch + 1}/{model.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                best_train_acc = train_acc
                best_val_acc = val_acc

        if show_loss:
            fig = self.loss_plot(train_loss_history, train_acc_history, val_loss_history, val_acc_history)
            fig.show()
        
        if prompt_save:
            save_prompt = input(f"Save? y/n [Best val loss: {best_val_loss:.4f}, Val acc: {best_val_acc:.2f}%] ")
            should_save = save_prompt.lower() == 'y'
        else:
            should_save = True
        
        if should_save:
            save_path = f"{self.ticker}_{self.interval}_{self.lagged_length}_{len(self.selected_features)}features.pth"
            torch.save({
                'model_state_dict': best_state_dict,
                'selected_features': self.selected_features if self.selected_features else None,
                'input_dim': self.input_dim,
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'train_acc': best_train_acc,
                'feature_selection_params': {
                    'importance_threshold': self.importance_threshold,
                    'max_features': self.max_features
                }
            }, save_path)
            print(f"Model saved to {save_path}")
        
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

def load_model(model_path: str, input_dim=None, verbose: bool = False):
    checkpoint = torch.load(model_path)
    
    checkpoint_input_dim = checkpoint.get('input_dim')
    
    effective_input_dim = checkpoint_input_dim if checkpoint_input_dim is not None else input_dim
    
    if effective_input_dim is None:
        raise ValueError("Input dimension could not be determined from checkpoint and wasn't provided. Please specify input_dim.")
    
    # If we have feature selection params, use them
    feature_selection_params = checkpoint.get('feature_selection_params', {})
    importance_threshold = feature_selection_params.get('importance_threshold', 10)
    max_features = feature_selection_params.get('max_features', 50)
    
    model = ClassifierModel(
        train=False, 
        input_dim=effective_input_dim,
        importance_threshold=importance_threshold,
        max_features=max_features
    )
    
    model.selected_features = checkpoint.get('selected_features')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if verbose:
        print(f"Model loaded with input_dim: {effective_input_dim}")
        print(f"Selected features: {len(model.selected_features) if model.selected_features else 'None'}")
        
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
            if 'val_acc' in checkpoint:
                print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model

if __name__ == "__main__":
    model = ClassifierModel(ticker="BTC-USDT", chunks=50, interval="5min", age_days=0, epochs=50, lagged_length=20, use_feature_selection=True)
    model = model.train_model(model, prompt_save=True, show_loss=False)

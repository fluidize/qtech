import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from trading.model_tools import fetch_data
from loss_functions import IntervalLoss
from main import PriceDataset
import matplotlib.pyplot as plt

EPOCHS = 500
DEVICE = 'cuda'
DATA = {
    "symbol": "SOL-USDT",
    "days":180,
    "interval": "30m",
    "age_days": 0,
    "data_source": "binance",
    "cache_expiry_hours": 999,
    "verbose": True
}
class DirectionalConfidencePredictor(nn.Module):
    def __init__(self, input_dim, dropout=0.03):
        super().__init__()
        self.input_dim = input_dim
        
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main_network(x) #y2 = upper bound, y1 = lower bound
        return x.squeeze()

class TensorSubset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

data = fetch_data(**DATA)
full_dataset = PriceDataset(data, shift=10)
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, shuffle=False)
train_dataset = TensorSubset(full_dataset.X[train_idx], full_dataset.y[train_idx])
val_dataset = TensorSubset(full_dataset.X[val_idx], full_dataset.y[val_idx])

directional_confidence_model = DirectionalConfidencePredictor(input_dim=full_dataset.X.shape[1]).to(DEVICE)
directional_confidence_optimizer = optim.Adam(directional_confidence_model.parameters(), weight_decay=1e-5)
directional_confidence_scheduler = optim.lr_scheduler.CosineAnnealingLR(directional_confidence_optimizer, T_max=EPOCHS, eta_min=1e-8)
directional_confidence_loss_fn = IntervalLoss()

best_val_loss = float('inf')
best_model_state = None
train_losses = []
val_losses = []

for i in range(EPOCHS):
    directional_confidence_model.train()
    directional_confidence_optimizer.zero_grad()
    yhat = directional_confidence_model(train_dataset.X.to(DEVICE))
    loss = directional_confidence_loss_fn(yhat, train_dataset.y.to(DEVICE))
    loss.backward()
    directional_confidence_optimizer.step()
    directional_confidence_scheduler.step()
    train_losses.append(loss.item())

    directional_confidence_model.eval()
    with torch.no_grad():
        val_yhat = directional_confidence_model(val_dataset.X.to(DEVICE))
        val_loss = directional_confidence_loss_fn(val_yhat, val_dataset.y.to(DEVICE)).item()
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = directional_confidence_model.state_dict().copy()

    print(f"Epoch {i+1}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

if best_model_state is not None:
    directional_confidence_model.load_state_dict(best_model_state)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

directional_confidence_model.eval()
with torch.no_grad():
    test_y = directional_confidence_model(val_dataset.X.to(DEVICE)).detach().cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(val_dataset.y.numpy(), label='True')
plt.plot(test_y[:,1], label='Predicted Upper Bound')
plt.plot(test_y[:,0], label='Predicted Lower Bound')
plt.plot(test_y[:,1] - test_y[:,0], label='Spread')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('True vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
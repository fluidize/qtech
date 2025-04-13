import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from PIL import Image
import io
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

model_path = r"trading\brains\time_series\single_predictors"
sys.path.append(model_path)
from classifier_model import ClassifierModel

model = ClassifierModel(ticker="SOL-USDT", chunks=1, interval="1min", age_days=0, epochs=1, pct_threshold=0.1, lagged_length=5, use_feature_selection=False)
dummy_input = torch.randn(1, 57)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Log the model graph
writer.add_graph(model, dummy_input)

# Close the writer
writer.close()

print("Model graph has been logged to TensorBoard.")
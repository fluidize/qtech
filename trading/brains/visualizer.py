import torch
from torchviz import make_dot
import sys
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

model_path = r"trading\brains\time_series\single_predictors"
sys.path.append(model_path)
from classifier_model import ClassifierModel

model = ClassifierModel(ticker="SOL-USDT", chunks=1, interval="1min", age_days=0, epochs=100, pct_threshold=0.1, lagged_length=5, use_feature_selection=False)
dummy_input = torch.randn(1, 57)
output = model(dummy_input)

dot = make_dot(output, params=dict(model.named_parameters()))
dot = dot.pipe(format='png')
image = Image.open(io.BytesIO(dot))
image.show()
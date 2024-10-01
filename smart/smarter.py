import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

traindf = pd.read_csv("train_features.csv")
traindfout = pd.read_csv("train_targets.csv")

X = traindf.drop(["match_id_hash"], inplace=False, axis=1)
y = traindfout["radiant_win"]

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())
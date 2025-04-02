import tensorflow as tf
import keras
from keras import layers
from keras import ops
import pandas as pd
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


inputs = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(filters=16, kernel_size=(7,7), activation="relu")(inputs)
x = layers.BatchNormalization()(x) 

x = layers.Dense(256)(x)
x = layers.LeakyReLU(negative_slope=0.1)(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU(negative_slope=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=x)

model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"]
    )

model.fit(X_train, y_train, batch_size=32, epochs=10)

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
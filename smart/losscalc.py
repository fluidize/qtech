import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

traindf = pd.read_csv("train_features.csv")
traindfout = pd.read_csv("train_targets.csv")

X = traindf.drop(["match_id_hash"], inplace=False, axis=1)
y = traindfout["radiant_win"]

print()

alpha_values = []
loss_values = []
cross_val_scores = []

for iter_alpha in range(110, 115, 1):
    alpha_values.append(iter_alpha/100)
    model = MLPClassifier(solver="adam", alpha=iter_alpha/100, max_iter=1000)
    print('fitting...')
    model.fit(X,y)
    #low loss + low valid score = overfit
    bestloss = model.best_loss_
    loss_values.append(bestloss)
    print('validating...')
    cross_val_scores.append(cross_val_score(model, X, y, cv=5))

for x in range(len(alpha_values)):
    print(f"alpha: {alpha_values[x]}, loss: {loss_values[x]} validation_avg: {cross_val_scores[x].mean()}, val_std: {cross_val_scores[x].std()}")

# plt.plot(alpha_values, loss_values)
# plt.plot(alpha_values, validscore_values)
# plt.show()

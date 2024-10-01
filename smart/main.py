import pandas as pd
from sklearn.neural_network import MLPClassifier
import math
import numpy as np

traindf = pd.read_csv("train_features.csv")
traindfout = pd.read_csv("train_targets.csv")

X = traindf.drop(["match_id_hash"], inplace=False, axis=1)
y = traindfout["radiant_win"]
#print(X)

model = MLPClassifier(solver="adam", alpha=1.1, max_iter=1000, verbose=True)


print('fitting...')
model.fit(X,y)

nnloss = model.best_loss_
print(nnloss)

testdf = pd.read_csv("test_features.csv")
test_matchidhash = testdf["match_id_hash"]


Xtest = testdf.drop(["match_id_hash"], inplace=False, axis=1)

Xtest_proba = model.predict_proba(Xtest)[:, 1]

results = pd.DataFrame(test_matchidhash, columns=["match_id_hash"])
results["radiant_win_prob"] = Xtest_proba

print(results)
results.to_csv("prediction.csv", index=False)
print("DONE")

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("trading")
import model_tools as mt

# Define the model
model = lgb.LGBMClassifier(verbose=-1)

# Define the parameter grid
param_grid = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'min_data_in_leaf': [20, 30, 50],
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.8, 1.0],
    'lambda_l1': [0, 0.1],
    'lambda_l2': [0, 0.1]
}

X, y = mt.prepare_data_classifier_test(mt.fetch_data(ticker="BTC-USDT", chunks=10, interval="1min", age_days=0), lagged_length=20, train_split=True)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X, y)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)
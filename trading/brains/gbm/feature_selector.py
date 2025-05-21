import pandas as pd
import numpy as np
import time
import os
from rich import print
from typing import List, Optional
from rich.console import Console
from rich.table import Table

# Set environment variable to suppress joblib/loky warning about CPU count
os.environ["LOKY_MAX_CPU_COUNT"] = "24"

class SimpleFeatureSelector:
    """
    Simple feature selector using only LightGBM feature importances.
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, feature_names: List[str], importance_threshold: float = 5, max_features: Optional[int] = -1):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.importance_threshold = importance_threshold
        self.max_features = max_features
        self.important_features = None
        self.console = Console()

    def get_important_features(self) -> List[str]:
        import lightgbm as lgb
        import numpy as np
        start_time = time.time()
        gbm = lgb.LGBMClassifier(
            objective='multiclass',
            metric='multi_logloss',
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
        indices = np.argsort(lgb_importances)[::-1]
        features = [self.feature_names[i] for i in indices]
        importance_values = [lgb_importances[i] for i in indices]
        candidate_features = [f for f, imp in zip(features, importance_values) if imp >= self.importance_threshold]
        initial_feature_count_after_threshold = len(candidate_features)
        self.console.print(f"[bold cyan]Features above threshold {self.importance_threshold}: {initial_feature_count_after_threshold} of {len(features)}[/bold cyan]")
        if self.max_features != -1 and len(candidate_features) > self.max_features:
            candidate_features = candidate_features[:self.max_features]
            self.console.print(f"[bold yellow]Limited to top {self.max_features} features[/bold yellow]")
        end_time = time.time()
        self.important_features = candidate_features
        table = Table(title="SimpleFeatureSelector: Selected Features")
        table.add_column("#", justify="right")
        table.add_column("Feature")
        table.add_column("Importance")
        for i, feature in enumerate(self.important_features):
            idx = features.index(feature)
            imp = importance_values[idx]
            table.add_row(str(i+1), feature, str(round(imp, 4)))
        self.console.print(table)
        self.console.print(f"[green]Selection completed in {end_time - start_time:.2f}s[/green]")
        self.console.print(f"[bold]Selected {len(self.important_features)} features from original {len(self.feature_names)}[/bold]")
        return self.important_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.important_features is None:
            raise ValueError("Call get_important_features() first")
        return X[self.important_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.X_train = X
        self.y_train = y
        self.feature_names = X.columns.tolist()
        self.get_important_features()
        return self.transform(X)

class AdvancedFeatureSelector:
    """
    Feature importance-based selector using ensemble methods (LGBM + RandomForest + correlation + greedy selection)
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, feature_names: List[str], importance_threshold: float = 5, max_features: Optional[int] = -1):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.importance_threshold = importance_threshold
        self.max_features = max_features
        self.important_features = None
        self.console = Console()

    def get_important_features(self) -> List[str]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import MinMaxScaler
        import lightgbm as lgb
        import numpy as np
        start_time = time.time()
        gbm = lgb.LGBMClassifier(
            objective='multiclass',
            metric='multi_logloss',
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
        RFClassifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        RFClassifier.fit(self.X_train, self.y_train)
        rf_importances = RFClassifier.feature_importances_
        combined_importances = (np.array(rf_importances) + np.array(lgb_importances)) / 2
        indices = np.argsort(combined_importances)[::-1]
        features = [self.feature_names[i] for i in indices]
        importance_values = [combined_importances[i] for i in indices]
        candidate_features = [f for f, imp in zip(features, importance_values) if imp >= self.importance_threshold]
        initial_feature_count_after_threshold = len(candidate_features)
        self.console.print(f"[bold cyan]Features above threshold {self.importance_threshold}: {initial_feature_count_after_threshold} of {len(features)}[/bold cyan]")
        if self.max_features != -1 and len(candidate_features) > self.max_features:
            candidate_features = candidate_features[:self.max_features]
            self.console.print(f"[bold yellow]Limited to top {self.max_features} features[/bold yellow]")
        X_candidates = self.X_train[candidate_features]
        correlation_matrix = X_candidates.corr().abs()
        to_drop = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] > 0.95:
                    to_drop.add(correlation_matrix.columns[i])
        candidate_features = [f for f in candidate_features if f not in to_drop]
        self.console.print(f"[magenta]Removed {len(to_drop)} highly correlated features, {len(candidate_features)} remaining[/magenta]")
        final_features = []
        best_score = 0
        scaler = MinMaxScaler()
        X_scaled_array = scaler.fit_transform(self.X_train[[candidate_features[0]]])
        X_scaled = pd.DataFrame(X_scaled_array, columns=[candidate_features[0]], index=self.X_train.index)
        baseline_model = lgb.LGBMClassifier(random_state=42)
        baseline_score = cross_val_score(baseline_model, X_scaled, self.y_train, cv=3, scoring='accuracy').mean()
        final_features = [candidate_features[0]]
        best_score = baseline_score
        self.console.print(f"[bold green]Starting with feature: {candidate_features[0]}, baseline score: {baseline_score:.4f}[/bold green]")
        for feature in candidate_features[1:]:
            current_features = final_features + [feature]
            X_scaled_array = scaler.fit_transform(self.X_train[current_features])
            X_scaled = pd.DataFrame(X_scaled_array, columns=current_features, index=self.X_train.index)
            model = lgb.LGBMClassifier(random_state=42)
            score = cross_val_score(model, X_scaled, self.y_train, cv=3, scoring='accuracy').mean()
            if score >= best_score - (0.005 * best_score):
                final_features.append(feature)
                if score > best_score:
                    best_score = score
                self.console.print(f"[green]Added feature: {feature}, new score: {score:.4f}[/green]")
            else:
                self.console.print(f"[red]Rejected feature: {feature}, score: {score:.4f} vs best: {best_score:.4f}[/red]")
        end_time = time.time()
        self.important_features = final_features
        filtered_importances = []
        for feature in self.important_features:
            idx = features.index(feature)
            imp = importance_values[idx]
            filtered_importances.append(imp)
        table = Table(title="AdvancedFeatureSelector: Selected Features")
        table.add_column("#", justify="right")
        table.add_column("Feature")
        table.add_column("Importance")
        for i, (feature, importance) in enumerate(zip(self.important_features, filtered_importances)):
            table.add_row(str(i+1), feature, str(round(importance, 4)))
        self.console.print(table)
        selection_method = f"importance threshold ({self.importance_threshold})"
        if self.max_features != -1:
            selection_method += f" and max features limit ({self.max_features})"
        self.console.print(f"[bold]Selection criteria: {selection_method}[/bold]")
        self.console.print(f"[bold]Selected {len(self.important_features)} features from original {len(self.feature_names)}[/bold]")
        self.console.print(f"[green]Selection completed in {end_time - start_time:.2f}s[/green]")
        return self.important_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.important_features is None:
            raise ValueError("Call get_important_features() first")
        return X[self.important_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.X_train = X
        self.y_train = y
        self.feature_names = X.columns.tolist()
        self.get_important_features()
        return self.transform(X)

if __name__ == "__main__":
    import sys
    sys.path.append("trading")
    import model_tools as mt
    X, y = mt.prepare_data_classifier(mt.fetch_data(ticker="BTC-USDT", chunks=5, interval="1min", age_days=60, kucoin=True), lagged_length=5, extra_features=False, elapsed_time=False)
    print("\n[bold underline]SimpleFeatureSelector[/bold underline]")
    simple_selector = SimpleFeatureSelector(X, y, X.columns.tolist())
    simple_features = simple_selector.get_important_features()
    print(simple_features)
    print("\n[bold underline]AdvancedFeatureSelector[/bold underline]")
    advanced_selector = AdvancedFeatureSelector(X, y, X.columns.tolist())
    advanced_features = advanced_selector.get_important_features()
    print(advanced_features)

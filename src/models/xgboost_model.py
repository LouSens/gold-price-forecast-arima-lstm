"""XGBoost regressor on engineered lag/rolling features, predicting next-day log return."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import XGBConfig
from ..evaluation import rmse
from ..features import returns_to_price


class XGBReturnForecaster:
    name = "XGBoost"

    def __init__(self, config: XGBConfig | None = None, seed: int = 42):
        self.config = config or XGBConfig()
        self.seed = seed
        self.model_: XGBRegressor | None = None
        self.best_params_: dict | None = None
        self.tuning_results_: pd.DataFrame | None = None

    def _make(self, **params) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.config.n_estimators,
            early_stopping_rounds=self.config.early_stopping_rounds,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            **params,
        )

    def fit(self, X_train, y_train, X_val, y_val, close_val, target_close_val) -> "XGBReturnForecaster":
        """Grid-search on the validation set; selection criterion is validation price RMSE."""
        grid = self.config.param_grid
        keys = list(grid)
        records, best_score = [], np.inf
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            model = self._make(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            price_pred = returns_to_price(close_val, model.predict(X_val))
            score = rmse(target_close_val, price_pred)
            # best_iteration is zero-based: the model predicts with best_iteration + 1 boosting rounds
            records.append({**params, "best_iteration": model.best_iteration,
                            "boosting_rounds": model.best_iteration + 1, "val_rmse": score})
            if score < best_score:
                best_score, self.model_, self.best_params_ = score, model, params
        self.tuning_results_ = pd.DataFrame(records).sort_values("val_rmse").reset_index(drop=True)
        return self

    def predict_return(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict(X)

    def predict_close(self, X, close) -> np.ndarray:
        return returns_to_price(close, self.predict_return(X))

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=feature_names).sort_values(ascending=False)

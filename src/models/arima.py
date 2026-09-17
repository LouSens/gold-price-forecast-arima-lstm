"""ARIMA with ADF-selected differencing and AIC grid search over (p, q).

Parameters are estimated on the training period only. Test forecasts are
one-step-ahead: the fitted parameters are re-applied (not re-estimated) to the
full series, so the forecast for ``t+1`` conditions only on observations up to ``t``.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ..config import ArimaConfig
from ..preprocessing import select_differencing_order


class ArimaForecaster:
    name = "ARIMA"

    def __init__(self, config: ArimaConfig | None = None):
        self.config = config or ArimaConfig()
        self.d_: int | None = None
        self.order_: tuple[int, int, int] | None = None
        self.result_ = None
        self.search_results_: pd.DataFrame | None = None

    def _transform(self, close: pd.Series) -> np.ndarray:
        values = close.to_numpy(dtype=float)
        return np.log(values) if self.config.use_log else values

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return np.exp(values) if self.config.use_log else values

    def fit(self, close_train: pd.Series) -> "ArimaForecaster":
        cfg = self.config
        y = self._transform(close_train)
        self.d_ = select_differencing_order(pd.Series(y), max_d=cfg.max_d, alpha=cfg.adf_alpha)

        records, best = [], None
        for p, q in itertools.product(cfg.p_values, cfg.q_values):
            order = (p, self.d_, q)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(y, order=order).fit()
            except Exception as exc:  # noqa: BLE001 - some orders legitimately fail to converge
                records.append({"order": order, "aic": np.nan, "bic": np.nan, "error": str(exc)})
                continue
            records.append({"order": order, "aic": res.aic, "bic": res.bic, "error": None})
            if best is None or res.aic < best.aic:
                best = res
        if best is None:
            raise RuntimeError("No ARIMA order could be fitted.")

        self.result_ = best
        self.order_ = tuple(best.model.order)
        self.search_results_ = pd.DataFrame(records).sort_values("aic").reset_index(drop=True)
        return self

    def predict_close(self, close_full: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
        """One-step-ahead forecasts of ``Close[t+1]`` for each origin ``t``."""
        if self.result_ is None:
            raise RuntimeError("Call fit() first.")
        y_full = self._transform(close_full)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.result_.apply(y_full)
            in_sample = np.asarray(applied.predict(start=0, end=len(y_full) - 1))
        positions = close_full.index.get_indexer(origins)
        if (positions < 0).any() or (positions + 1 >= len(y_full)).any():
            raise KeyError("Every origin needs a following observation inside close_full.")
        return self._inverse(in_sample[positions + 1])

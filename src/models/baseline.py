"""Naive random-walk baseline: tomorrow's close equals today's close."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveForecaster:
    name = "Naive"

    def fit(self, *_, **__) -> "NaiveForecaster":
        return self

    def predict_close(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["close"].to_numpy(dtype=float)

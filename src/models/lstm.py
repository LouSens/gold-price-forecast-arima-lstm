"""PyTorch LSTM on sliding windows of standardised log returns.

The scaler (mean/std) is fitted on training-period returns only. Hyperparameters
(window length, hidden units, layers) are chosen by validation price RMSE, with
early stopping on validation loss.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import LSTMConfig
from ..evaluation import rmse
from ..features import make_sequences, returns_to_price


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    device: str,
    seed: int,
) -> tuple[nn.Module, pd.DataFrame]:
    """Adam + MSE with early stopping; restores the best validation weights."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,  # windows are self-contained; shuffling them does not leak future data
        generator=generator,
    )
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait, history = np.inf, None, 0, []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-6:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


class LSTMForecaster:
    name = "LSTM"

    def __init__(self, config: LSTMConfig | None = None, device: str = "cpu", seed: int = 42):
        self.config = config or LSTMConfig()
        self.device = device
        self.seed = seed
        self.mu_: float | None = None
        self.sigma_: float | None = None
        self.model_: LSTMRegressor | None = None
        self.best_params_: dict | None = None
        self.history_: pd.DataFrame | None = None
        self.tuning_results_: pd.DataFrame | None = None
        self._scaled_returns: pd.Series | None = None

    def _scale(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mu_) / self.sigma_

    def _predict_scaled(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(X).to(self.device)).cpu().numpy()

    def fit(self, log_returns: pd.Series, train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> "LSTMForecaster":
        cfg = self.config
        train_returns = log_returns.loc[: train_frame.index[-1]].dropna()
        self.mu_, self.sigma_ = float(train_returns.mean()), float(train_returns.std())
        self._scaled_returns = (log_returns - self.mu_) / self.sigma_

        records, best_score = [], np.inf
        for window, hidden, layers in itertools.product(cfg.windows, cfg.hidden_sizes, cfg.num_layers):
            torch.manual_seed(self.seed)
            X_tr, m_tr = make_sequences(self._scaled_returns, train_frame.index, window)
            X_va, m_va = make_sequences(self._scaled_returns, val_frame.index, window)
            y_tr = self._scale(train_frame["target_return"].to_numpy()[m_tr])
            y_va = self._scale(val_frame["target_return"].to_numpy()[m_va])

            model = LSTMRegressor(1, hidden, layers, cfg.dropout)
            model, history = train_model(
                model, X_tr, y_tr, X_va, y_va,
                batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
                max_epochs=cfg.max_epochs, patience=cfg.patience,
                device=self.device, seed=self.seed,
            )
            model.eval()
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(X_va).to(self.device)).cpu().numpy()
            pred_price = returns_to_price(val_frame["close"].to_numpy()[m_va], pred_scaled * self.sigma_ + self.mu_)
            score = rmse(val_frame["target_close"].to_numpy()[m_va], pred_price)
            records.append({
                "window": window, "hidden_size": hidden, "num_layers": layers,
                "epochs": len(history), "best_val_loss": history["val_loss"].min(), "val_rmse": score,
            })
            if score < best_score:
                best_score = score
                self.model_, self.history_ = model, history
                self.best_params_ = {"window": window, "hidden_size": hidden, "num_layers": layers}

        self.tuning_results_ = pd.DataFrame(records).sort_values("val_rmse").reset_index(drop=True)
        return self

    def predict_close(self, frame: pd.DataFrame) -> np.ndarray:
        """Next-day close for each origin in ``frame``; NaN where history is shorter than the window."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        X, mask = make_sequences(self._scaled_returns, frame.index, self.best_params_["window"])
        out = np.full(len(frame), np.nan)
        if mask.any():
            pred_return = self._predict_scaled(X) * self.sigma_ + self.mu_
            out[mask] = returns_to_price(frame["close"].to_numpy()[mask], pred_return)
        return out

    def save(self, path) -> None:
        torch.save(
            {"state_dict": self.model_.state_dict(), "params": self.best_params_,
             "dropout": self.config.dropout, "mu": self.mu_, "sigma": self.sigma_},
            path,
        )

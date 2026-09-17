"""Central, typed configuration for the forecasting pipeline.

Every tunable used by the pipeline lives here so that experiments are
reproducible and the notebook, CLI and tests agree on the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "raw" / "xauusd_daily.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures" / "pipeline"

    def ensure(self) -> None:
        for d in (self.processed_dir, self.models_dir, self.results_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    # Yahoo Finance gold futures; used only when downloading is explicitly requested.
    ticker: str = "GC=F"
    start_date: str = "2015-01-01"
    end_date: str = "2026-04-30"
    train_frac: float = 0.80
    val_frac: float = 0.10  # test = remainder


@dataclass
class FeatureConfig:
    n_lags: int = 10
    windows: tuple[int, ...] = (5, 10, 20)


@dataclass
class ArimaConfig:
    p_values: tuple[int, ...] = (0, 1, 2, 3)
    q_values: tuple[int, ...] = (0, 1, 2, 3)
    max_d: int = 2
    use_log: bool = True
    adf_alpha: float = 0.05


@dataclass
class XGBConfig:
    param_grid: dict = field(
        default_factory=lambda: {
            "max_depth": [2, 3, 5],
            "learning_rate": [0.01, 0.05],
            "min_child_weight": [1, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }
    )
    n_estimators: int = 2000
    early_stopping_rounds: int = 50


@dataclass
class LSTMConfig:
    windows: tuple[int, ...] = (30, 60)
    hidden_sizes: tuple[int, ...] = (32, 64)
    num_layers: tuple[int, ...] = (1, 2)
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10


@dataclass
class Config:
    seed: int = 42
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    arima: ArimaConfig = field(default_factory=ArimaConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)

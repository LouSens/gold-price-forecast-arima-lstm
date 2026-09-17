"""End-to-end CLI pipeline: load -> clean -> EDA figures -> features -> split -> train -> evaluate.

Usage::

    python -m src.pipeline                         # uses data/raw/xauusd_daily.csv
    python -m src.pipeline --data path/to/file.csv
    python -m src.pipeline --skip-lstm --cpu       # quick run without the neural net
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import visualization as viz  # noqa: E402
from .config import Config  # noqa: E402
from .data_loader import read_price_csv, resolve_data_path  # noqa: E402
from .evaluation import compare_models, diebold_mariano, rmse  # noqa: E402
from .features import build_feature_frame, feature_columns  # noqa: E402
from .models import ArimaForecaster, LSTMForecaster, NaiveForecaster, XGBReturnForecaster  # noqa: E402
from .preprocessing import add_returns, adf_test, chronological_split, clean_prices, filter_date_range  # noqa: E402
from .utils import get_device, get_logger, set_seed, timer  # noqa: E402

log = get_logger()


def run(cfg: Config, data_path: Path | None = None, skip_lstm: bool = False, prefer_gpu: bool = True) -> pd.DataFrame:
    set_seed(cfg.seed)
    paths = cfg.paths
    paths.ensure()
    fig = paths.figures_dir

    # 1. Load & clean -----------------------------------------------------------------
    source = resolve_data_path(data_path or paths.raw_data)
    log.info("Loading %s", source)
    raw = read_price_csv(source)
    df, quality = clean_prices(raw)
    df = filter_date_range(df, cfg.data.start_date, cfg.data.end_date)
    log.info("Quality report: %s", quality.as_dict())
    df = add_returns(df)
    df.to_csv(paths.processed_dir / "xauusd_clean.csv")

    # 2. EDA figures --------------------------------------------------------------------
    viz.plot_price_history(df, fig / "01_price_history.png")
    viz.plot_returns(df["log_return"], fig / "02_returns_volatility.png")
    viz.plot_seasonal_profile(df, fig / "03_seasonal_profile.png")
    viz.plot_acf_pacf(np.log(df["Close"]), title="log(Close)", path=fig / "04_acf_pacf_level.png")
    viz.plot_acf_pacf(df["log_return"], title="log return", path=fig / "05_acf_pacf_return.png")
    stationarity = {"log_close": adf_test(np.log(df["Close"])), "log_return": adf_test(df["log_return"])}

    # 3. Features & split ---------------------------------------------------------------
    frame = build_feature_frame(df, cfg.features.n_lags, cfg.features.windows)
    train, val, test = chronological_split(frame, cfg.data.train_frac, cfg.data.val_frac)
    feats = feature_columns(frame)
    log.info("Rows train/val/test = %d/%d/%d; %d features", len(train), len(val), len(test), len(feats))
    viz.plot_split(df["Close"], train.index[-1], val.index[-1], fig / "06_split.png")

    y_true, current = test["target_close"].to_numpy(), test["close"].to_numpy()
    predictions, val_predictions, timings = {}, {}, {}

    # 4. Models -------------------------------------------------------------------------
    with timer("Naive", log) as t:
        predictions["Naive"] = NaiveForecaster().predict_close(test)
        val_predictions["Naive"] = NaiveForecaster().predict_close(val)
    timings["Naive"] = t["seconds"]

    with timer("ARIMA", log) as t:
        arima = ArimaForecaster(cfg.arima).fit(df["Close"].loc[: train.index[-1]])
        predictions[f"ARIMA{arima.order_}"] = arima.predict_close(df["Close"], test.index)
        val_predictions[f"ARIMA{arima.order_}"] = arima.predict_close(df["Close"], val.index)
    timings["ARIMA"] = t["seconds"]
    arima.search_results_.to_csv(paths.results_dir / "arima_grid_search.csv", index=False)
    (paths.results_dir / "arima_summary.txt").write_text(str(arima.result_.summary()))

    with timer("XGBoost", log) as t:
        xgb = XGBReturnForecaster(cfg.xgb, cfg.seed).fit(
            train[feats], train["target_return"], val[feats], val["target_return"],
            val["close"], val["target_close"],
        )
        predictions["XGBoost"] = xgb.predict_close(test[feats], test["close"])
        val_predictions["XGBoost"] = xgb.predict_close(val[feats], val["close"])
    timings["XGBoost"] = t["seconds"]
    xgb.tuning_results_.to_csv(paths.results_dir / "xgboost_tuning.csv", index=False)
    xgb.model_.save_model(paths.models_dir / "xgboost.json")
    viz.plot_feature_importance(xgb.feature_importance(feats), path=fig / "08_xgb_feature_importance.png")

    if not skip_lstm:
        device = get_device(prefer_gpu)
        log.info("Training LSTM on %s", device)
        with timer("LSTM", log) as t:
            lstm = LSTMForecaster(cfg.lstm, device=device, seed=cfg.seed).fit(df["log_return"], train, val)
            predictions["LSTM"] = lstm.predict_close(test)
            val_predictions["LSTM"] = lstm.predict_close(val)
        timings["LSTM"] = t["seconds"]
        lstm.tuning_results_.to_csv(paths.results_dir / "lstm_tuning.csv", index=False)
        lstm.save(paths.models_dir / "lstm.pt")
        viz.plot_learning_curve(lstm.history_, fig / "07_lstm_learning_curve.png")

    # 5. Evaluation ---------------------------------------------------------------------
    # Test metrics are reported; the model is *selected* on validation RMSE so the test set never influences the choice.
    metrics = compare_models(current, y_true, predictions)
    metrics.insert(0, "Val RMSE", [rmse(val["target_close"], val_predictions[n]) for n in metrics.index])
    metrics = metrics.sort_values("Val RMSE")
    best_model = metrics["Val RMSE"].idxmin()
    metrics["Fit+predict time (s)"] = [timings[n.split("(")[0]] for n in metrics.index]
    naive = predictions["Naive"]
    dm = {n: diebold_mariano(y_true, p, naive) for n, p in predictions.items() if n != "Naive"}
    metrics["DM p-value vs Naive"] = [dm.get(n, {}).get("p_value", np.nan) for n in metrics.index]

    metrics.round(4).to_csv(paths.results_dir / "metrics.csv")
    (paths.results_dir / "metrics.md").write_text(metrics.round(4).to_markdown())
    # A forecast made at origin t predicts the close of the next trading day (the target date).
    target_dates = df.index[df.index.get_indexer(test.index) + 1]
    pd.DataFrame({"origin_date": test.index, "target_date": target_dates, "actual_next_close": y_true, **predictions}).to_csv(
        paths.results_dir / "test_predictions.csv", index=False
    )
    (paths.results_dir / "run_summary.json").write_text(json.dumps({
        "data_source": str(source),
        "quality": {k: str(v) for k, v in quality.as_dict().items()},
        "rows": {"clean": len(df), "train": len(train), "val": len(val), "test": len(test)},
        "origin_periods": {name: [str(s.index[0].date()), str(s.index[-1].date())] for name, s in
                           (("train", train), ("val", val), ("test", test))},
        "target_periods": {name: [str(df.index[df.index.get_loc(s.index[0]) + 1].date()),
                                  str(df.index[df.index.get_loc(s.index[-1]) + 1].date())] for name, s in
                           (("train", train), ("val", val), ("test", test))},
        "stationarity": stationarity,
        "selection_criterion": "validation RMSE",
        "best_model": best_model,
        "arima_order": arima.order_,
        "xgboost_best_params": xgb.best_params_,
        "xgboost_boosting_rounds": int(xgb.model_.best_iteration) + 1,
        "lstm_best_params": None if skip_lstm else lstm.best_params_,
    }, indent=2, default=float))

    viz.plot_predictions(target_dates, y_true, predictions, path=fig / "09_test_predictions.png")
    viz.plot_predictions(target_dates, y_true, predictions, last_n=60, path=fig / "10_test_predictions_zoom.png")
    viz.plot_residuals(target_dates, y_true, predictions, fig / "11_residuals.png")
    viz.plot_metric_comparison(metrics, fig / "12_metric_comparison.png")

    log.info("\n%s", metrics.round(4).to_string())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="CSV path (default: data/raw/xauusd_daily.csv)")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip the LSTM for a fast CPU-only run")
    parser.add_argument("--cpu", action="store_true", help="Force the LSTM to train on CPU")
    parser.add_argument("--start", default=None, help="Override analysis start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override analysis end date (YYYY-MM-DD)")
    args = parser.parse_args()

    cfg = Config()
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end
    run(cfg, data_path=args.data, skip_lstm=args.skip_lstm, prefer_gpu=not args.cpu)


if __name__ == "__main__":
    main()

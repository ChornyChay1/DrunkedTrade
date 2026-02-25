"""
ML baseline: Random Walk and Ridge/Linear Regression on lags.
Target: next-period return r_{t+1} = (close_{t+1} - close_t) / close_t.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Default path to CSV (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "BTCUSDT_3months_20260223_213051.csv"
TEST_FRAC = 0.2
N_LAGS = 10  # number of lag features based on past 1-step returns


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load CSV, sort by time, ensure datetime column."""
    df = pd.read_csv(csv_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute feature set and target for LinearRegression.

    Target:
        next-period return: r_{t+1} = (close_{t+1} - close_t) / close_t

    Features (all основаны на прошлых/текущих доходностях, без заглядывания в будущее):
        1.  r_1       — 1-шаговая доходность (close_t / close_{t-1} - 1)
        2.  r_5       — доходность за последние 5 шагов (close_t / close_{t-5} - 1)
        3.  r_10      — доходность за последние 10 шагов
        4.  r_20      — доходность за последние 20 шагов
        5.  cumret_5  — сумма 1-шаговых доходностей за окно 5
        6.  cumret_20 — сумма 1-шаговых доходностей за окно 20
        7.  mean_5    — средняя 1-шаговая доходность за окно 5
        8.  mean_20   — средняя 1-шаговая доходность за окно 20
        9.  vol_5     — стандартное отклонение 1-шаговых доходностей за окно 5
        10. vol_20    — стандартное отклонение 1-шаговых доходностей за окно 20
        11. vol_60    — стандартное отклонение 1-шаговых доходностей за окно 60
        12. zscore_20 — z-score текущей 1-шаговой доходности относительно окна 20
    """
    close = df["close"].astype(float)

    # Target: следующая доходность
    target = (close.shift(-1) - close) / close

    # 1-шаговые (прошлые) доходности: r_t = (close_t / close_{t-1} - 1)
    ret1 = close.pct_change(1)

    # Доходности за N шагов
    r_1 = ret1
    r_5 = close.pct_change(5)
    r_10 = close.pct_change(10)
    r_20 = close.pct_change(20)

    # Кумулятивные доходности как сумма 1-шаговых доходностей на окне
    cumret_5 = ret1.rolling(window=5, min_periods=5).sum()
    cumret_20 = ret1.rolling(window=20, min_periods=20).sum()

    # Скользящие средние и волатильности 1-шаговых доходностей
    mean_5 = ret1.rolling(window=5, min_periods=5).mean()
    mean_20 = ret1.rolling(window=20, min_periods=20).mean()
    vol_5 = ret1.rolling(window=5, min_periods=5).std()
    vol_20 = ret1.rolling(window=20, min_periods=20).std()
    vol_60 = ret1.rolling(window=60, min_periods=60).std()

    # Z-score текущей доходности относительно окна 20
    zscore_20 = (ret1 - mean_20) / vol_20

    # Базовые engineered-признаки
    feature_dict: dict[str, pd.Series] = {
        "r_1": r_1,
        "r_5": r_5,
        "r_10": r_10,
        "r_20": r_20,
        "cumret_5": cumret_5,
        "cumret_20": cumret_20,
        "mean_5": mean_5,
        "mean_20": mean_20,
        "vol_5": vol_5,
        "vol_20": vol_20,
        "vol_60": vol_60,
        "zscore_20": zscore_20,
    }

    # Исходные признаки: лаги 1-шаговых доходностей (ret1_{t-1}, ..., ret1_{t-N_LAGS})
    for i in range(1, N_LAGS + 1):
        feature_dict[f"ret1_lag_{i}"] = ret1.shift(i)

    features = pd.DataFrame(feature_dict)

    data = pd.concat([features, target.rename("target")], axis=1).dropna()
    X = data.drop(columns=["target"]).astype(float)
    y = data["target"].astype(float)
    return X, y


def time_split(X: pd.DataFrame, y: pd.Series, test_frac: float):
    """Split by time: first (1 - test_frac) train, last test_frac test."""
    n = len(X)
    split_idx = int(n * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MSE, MAE, RMSE, R²."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train ML baseline (Random Walk, Linear Regression).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to OHLCV CSV")
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC, help="Fraction of data for test")
    parser.add_argument("--out", type=Path, default=None, help="Metrics output path (default: experiments/ml_baseline_metrics.csv)")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (SCRIPT_DIR / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_and_prepare(csv_path)
    X, y = build_features_and_target(df)
    X_train, X_test, y_train, y_test = time_split(X, y, args.test_frac)

    # Use numpy arrays so sklearn uses row order (no index alignment issues)
    X_train_arr = np.asarray(X_train, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_train_arr = np.asarray(y_train, dtype=np.float64).ravel()
    y_test_arr = np.asarray(y_test, dtype=np.float64).ravel()

    # Scale features so Ridge/LR are not dominated by scale (lagged returns can be tiny)
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train_arr)
    X_test_arr = scaler.transform(X_test_arr)

    n_train, n_test = len(X_train_arr), len(X_test_arr)
    print(f"Train size: {n_train}, test size: {n_test}")

    results = []

    # 1) Random Walk: predict return = 0
    rw_pred = np.zeros_like(y_test_arr)
    rw_metrics = evaluate(y_test_arr, rw_pred)
    rw_metrics["model"] = "Random Walk"
    results.append(rw_metrics)

    # 2) Linear Regression with original + engineered features
    lr = LinearRegression().fit(X_train_arr, y_train_arr)
    lr_pred = lr.predict(X_test_arr)
    lr_metrics = evaluate(y_test_arr, lr_pred)
    lr_metrics["model"] = "LinearRegression"
    results.append(lr_metrics)

    # Diagnostics: показать, что LinearRegression действительно что-то учит
    print("\nPredictions (min, max, mean, std):")
    print(f"  Random Walk:       {rw_pred.min():.6f}, {rw_pred.max():.6f}, {rw_pred.mean():.6f}, {rw_pred.std():.6f}")
    print(f"  LinearRegression:  {lr_pred.min():.6f}, {lr_pred.max():.6f}, {lr_pred.mean():.6f}, {lr_pred.std():.6f}")
    print(f"  LR intercept={lr.intercept_:.6f}, coef norm={np.linalg.norm(lr.coef_):.6f}")

    # Пороги для меток 1/-1/0: 90-й перцентиль положительных и отрицательных доходностей
    pos_returns = y_train_arr[y_train_arr > 0]
    neg_returns = y_train_arr[y_train_arr < 0]
    threshold_buy = float(np.percentile(pos_returns, 10)) if len(pos_returns) > 0 else np.inf
    threshold_sell = float(np.percentile(neg_returns, 10)) if len(neg_returns) > 0 else -np.inf  # 70% отрицательных ниже этого
    percentiles = {"threshold_buy": threshold_buy, "threshold_sell": threshold_sell}

    # Save trained model, scaler and percentiles
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    lr_path = models_dir / "linear_regression.joblib"
    scaler_path = models_dir / "scaler.joblib"
    dump(lr, lr_path)
    dump(scaler, scaler_path)
    with open(models_dir / "percentiles.json", "w", encoding="utf-8") as f:
        json.dump(percentiles, f, indent=2)
    print(f"LinearRegression model saved to {lr_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Percentiles saved to {models_dir / 'percentiles.json'} (threshold_buy={threshold_buy:.6f}, threshold_sell={threshold_sell:.6f})")

    # Console output
    print("ML baseline metrics (test set, target = next-period return)")
    print("-" * 60)
    for m in results:
        print(f"{m['model']:20} MSE={m['mse']:.6f}  MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.6f}")
    print("-" * 60)

    # Save to file
    out_path = args.out
    if out_path is None:
        out_path = SCRIPT_DIR / "ml_baseline_metrics.csv"
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(results)
    cols = ["model", "mse", "mae", "rmse", "r2"]
    out_df[cols].to_csv(out_path, index=False)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()

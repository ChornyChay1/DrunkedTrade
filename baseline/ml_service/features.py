"""
Построение признаков для линейной регрессии (совместимо с experiments/train_ml_baseline.py).
Один ряд признаков для последней свечи в переданном ряде close.
"""
import numpy as np
import pandas as pd

N_LAGS = 10
MIN_CANDLES = 61  # vol_60 требует 60 точек, лаги 10 — итого достаточно 61


def build_feature_row(closes: list[float] | np.ndarray) -> np.ndarray | None:
    """
    Строит один вектор признаков для последней свечи.
    Порядок колонок совпадает с train_ml_baseline.build_features_and_target.

    :param closes: массив цен close (от старых к новым), длина >= MIN_CANDLES
    :return: массив формы (1, n_features) или None, если данных недостаточно
    """
    if len(closes) < MIN_CANDLES:
        return None
    close = pd.Series(closes, dtype=float)
    ret1 = close.pct_change(1)
    r_1 = ret1
    r_5 = close.pct_change(5)
    r_10 = close.pct_change(10)
    r_20 = close.pct_change(20)
    cumret_5 = ret1.rolling(window=5, min_periods=5).sum()
    cumret_20 = ret1.rolling(window=20, min_periods=20).sum()
    mean_5 = ret1.rolling(window=5, min_periods=5).mean()
    mean_20 = ret1.rolling(window=20, min_periods=20).mean()
    vol_5 = ret1.rolling(window=5, min_periods=5).std()
    vol_20 = ret1.rolling(window=20, min_periods=20).std()
    vol_60 = ret1.rolling(window=60, min_periods=60).std()
    zscore_20 = (ret1 - mean_20) / vol_20

    feature_dict = {
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
    for i in range(1, N_LAGS + 1):
        feature_dict[f"ret1_lag_{i}"] = ret1.shift(i)

    features = pd.DataFrame(feature_dict)
    last_row = features.iloc[-1:]
    if last_row.isna().any(axis=1).item():
        return None
    return np.asarray(last_row, dtype=np.float64).reshape(1, -1)

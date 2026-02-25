"""
Скрипт для подготовки примера входных данных для POST /predict с ответом prediction=1.
Использует реальные свечи из CSV и сохранённую модель.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
CSV_PATH = SCRIPT_DIR / "BTCUSDT_3months_20260223_213051.csv"

# Добавляем baseline/ml_service в path для импорта features
ML_SERVICE = SCRIPT_DIR.parent / "baseline" / "ml_service"
sys.path.insert(0, str(ML_SERVICE))
from features import build_feature_row, MIN_CANDLES


def main():
    with open(MODELS_DIR / "percentiles.json", encoding="utf-8") as f:
        percentiles = json.load(f)
    threshold_buy = percentiles["threshold_buy"]

    model = joblib.load(MODELS_DIR / "linear_regression.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    df = pd.read_csv(CSV_PATH)
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Ищем окно из MIN_CANDLES свечей, для которого предсказание >= threshold_buy
    for start in range(0, len(df) - MIN_CANDLES):
        end = start + MIN_CANDLES
        closes = df["close"].iloc[start:end].astype(float).tolist()
        X = build_feature_row(closes)
        if X is None:
            continue
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0])
        if pred >= threshold_buy:
            slice_df = df.iloc[start:end]
            candles = []
            for _, row in slice_df.iterrows():
                ts = int(row["timestamp"])
                candles.append({
                    "start": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "turnover": float(row.get("turnover", row["volume"])),
                })
            out = {"candles": candles}
            out_path = SCRIPT_DIR / "predict_example_response_1.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"Found slice start={start}, pred={pred:.6f}, threshold_buy={threshold_buy:.6f}")
            print(f"Written to {out_path}")
            return
    print("No slice found with prediction >= threshold_buy")


if __name__ == "__main__":
    main()

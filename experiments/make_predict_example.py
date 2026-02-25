import json
import sys
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
CSV_PATH = SCRIPT_DIR / "BTCUSDT_3months_20260223_213051.csv"

ML_SERVICE = SCRIPT_DIR.parent / "baseline" / "ml_service"
sys.path.insert(0, str(ML_SERVICE))
from features import build_feature_row, MIN_CANDLES


def slice_to_candles(df: pd.DataFrame, start: int, end: int) -> list[dict]:
    candles = []
    for _, row in df.iloc[start:end].iterrows():
        candles.append({
            "start": int(row["timestamp"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "turnover": float(row.get("turnover", row["volume"])),
        })
    return candles


def main():
    with open(MODELS_DIR / "percentiles.json", encoding="utf-8") as f:
        percentiles = json.load(f)
    threshold_buy = percentiles["threshold_buy"]
    threshold_sell = percentiles["threshold_sell"]

    model = joblib.load(MODELS_DIR / "linear_regression.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    df = pd.read_csv(CSV_PATH)
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Ищем по одному примеру для каждого исхода: 1, 0, -1
    found_1 = None  # (start, pred)
    found_0 = None
    found_minus1 = None

    for start in range(0, len(df) - MIN_CANDLES):
        end = start + MIN_CANDLES
        closes = df["close"].iloc[start:end].astype(float).tolist()
        X = build_feature_row(closes)
        if X is None:
            continue
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0])

        if pred >= threshold_buy and found_1 is None:
            found_1 = (start, pred)
        elif pred <= threshold_sell and found_minus1 is None:
            found_minus1 = (start, pred)
        elif threshold_sell < pred < threshold_buy and found_0 is None:
            found_0 = (start, pred)

        if found_1 is not None and found_0 is not None and found_minus1 is not None:
            break

    out_files = []
    if found_1 is not None:
        start, pred = found_1
        candles = slice_to_candles(df, start, start + MIN_CANDLES)
        out_path = SCRIPT_DIR / "predict_example_response_1.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"candles": candles}, f, indent=2)
        out_files.append((out_path, 1, pred))
    if found_0 is not None:
        start, pred = found_0
        candles = slice_to_candles(df, start, start + MIN_CANDLES)
        out_path = SCRIPT_DIR / "predict_example_response_0.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"candles": candles}, f, indent=2)
        out_files.append((out_path, 0, pred))
    if found_minus1 is not None:
        start, pred = found_minus1
        candles = slice_to_candles(df, start, start + MIN_CANDLES)
        out_path = SCRIPT_DIR / "predict_example_response_-1.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"candles": candles}, f, indent=2)
        out_files.append((out_path, -1, pred))

    print(f"threshold_buy={threshold_buy:.6f}, threshold_sell={threshold_sell:.6f}")
    for path, label, pred in out_files:
        print(f"  prediction={label}: pred={pred:.6f} -> {path.name}")
    if not out_files or len(out_files) < 3:
        missing = []
        if found_1 is None:
            missing.append("1")
        if found_0 is None:
            missing.append("0")
        if found_minus1 is None:
            missing.append("-1")
        print(f"No slice found for prediction(s): {', '.join(missing)}")


if __name__ == "__main__":
    main()

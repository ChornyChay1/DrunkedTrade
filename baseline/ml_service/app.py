import os
import json
import logging
from threading import local
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from features import build_feature_row, MIN_CANDLES
from schema import GetPredict, PredictRequest

logger = logging.getLogger("ML-Service")

model = None
scaler = None
threshold_buy = None
threshold_sell = None


def get_model_dir(local_path:str = "experiments/models") -> str:
    return os.path.join(Path(__file__).resolve().parent.parent.parent, local_path)


def prediction_to_label(pred: float) -> int:
    """Преобразование предсказания доходности в метку: 1 — покупка, -1 — продажа, 0 — бездействие."""
    if threshold_buy is None or threshold_sell is None:
        return 0
    if pred >= threshold_buy:
        return 1
    if pred <= threshold_sell:
        return -1
    return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для жизненного цикла приложения"""
    global model, scaler, threshold_buy, threshold_sell
    logger.info("Starting ML-Service")
    try:
        model_dir = get_model_dir()
        model = joblib.load(model_dir + "/linear_regression.joblib")
        scaler = joblib.load(model_dir + "/scaler.joblib")
        percentiles_path = Path(model_dir + "/percentiles.json")
        if percentiles_path.exists():
            with open(percentiles_path, encoding="utf-8") as f:
                p = json.load(f)
            threshold_buy = p["threshold_buy"]
            threshold_sell = p["threshold_sell"]
        else:
            threshold_buy = 0.01
            threshold_sell = -0.01
            logger.warning(
                "percentiles.json not found, using defaults: threshold_buy=%.2f, threshold_sell=%.2f. "
                "Run experiments/train_ml_baseline.py to generate percentiles.",
                threshold_buy,
                threshold_sell,
            )
        logger.info(
            "Model loaded: threshold_buy=%.6f, threshold_sell=%.6f",
            threshold_buy,
            threshold_sell,
        )
    except Exception as e:
        logger.error("Failed to load model/scaler/percentiles: %s", e, exc_info=True)
    yield
    logger.info("Shutting down ML-Service")


app = FastAPI(
    title="ML_SERVICE",
    debug=True,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов"""

    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Time: {process_time:.3f}s"
        )

        return response

    except Exception as e:
        logger.error(
            f"Error processing request: {request.method} {request.url.path} | {str(e)}",
            exc_info=True
        )
        raise


@app.get("/health")
async def health_check():
    """Проверка здоровья приложения"""
    logger.debug("Health check requested")
    return {"status": "healthy"}

@app.post("/predict", response_model=GetPredict)
async def predict(req: PredictRequest):
    if model is None or scaler is None:
        return GetPredict(prediction=0)

    if len(req.candles) < MIN_CANDLES:
        logger.warning(
            "predict: received %d candles, need at least %d for feature calculation",
            len(req.candles),
            MIN_CANDLES,
        )
        return GetPredict(prediction=0)

    closes = [float(c.close) for c in req.candles]
    X = build_feature_row(closes)
    if X is None:
        return GetPredict(prediction=0)

    X_scaled = scaler.transform(X)
    pred = float(model.predict(X_scaled)[0])
    label = prediction_to_label(pred)
    return GetPredict(prediction=label)

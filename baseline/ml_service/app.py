import logging
import time
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io

from schema import Candle, GetPredict, PredictRequest, TrainResponse
from model import model

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML-Service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для жизненного цикла приложения"""
    logger.info(f"Starting ML-Service")

    # Пытаемся загрузить сохраненную модель при старте
    try:
        model.load_model()
        logger.info("Model loaded from disk")
    except:
        logger.info("No saved model found, will train on first CSV upload")

    yield
    logger.info(f"Shutting down ML-Service")


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
    return {
        "status": "healthy",
        "model_trained": model.is_trained,
        "samples": model.training_samples if hasattr(model, 'training_samples') else 0
    }


@app.post("/predict", response_model=GetPredict)
async def predict(request: PredictRequest):
    """
    Предсказание сигнала на основе исторических свечей

    Возвращает:
    -1 - продажа
    0 - держать
    1 - покупка
    """
    if len(request.candles) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 candles for prediction")

    # Преобразуем в список словарей для модели
    candles_dict = [c.dict() for c in request.candles]

    # Получаем предсказание
    prediction = model.predict(candles_dict)

    logger.info(f"Prediction: {prediction} based on {len(candles_dict)} candles")

    return GetPredict(prediction=prediction)


@app.post("/train/csv", response_model=TrainResponse)
async def train_from_csv(
        file: UploadFile = File(...),
        test_size: float = 0.2
):
    """
    Обучение модели на CSV файле со структурой:
    timestamp,open,high,low,close,volume,datetime
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV")

    try:
        # Читаем CSV файл
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Проверяем наличие необходимых колонок
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {required_columns}"
            )

        # Преобразуем в формат свечей
        candles_list = []
        for _, row in df.iterrows():
            candle = {
                'start': int(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'turnover': float(row['close'] * row['volume'])  # Рассчитываем turnover
            }
            candles_list.append(candle)

        logger.info(f"Loaded {len(candles_list)} candles from CSV")

        # Обучаем модель
        accuracy, samples = model.train(candles_list)

        # Сохраняем модель
        model.save_model()

        return TrainResponse(
            message="Model trained successfully on CSV data",
            accuracy=float(accuracy),
            samples=samples,
            filename=file.filename
        )

    except Exception as e:
        logger.error(f"Error training from CSV: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/train/sample")
async def train_with_sample_data():
    """
    Обучение модели на синтетических данных (для тестирования)
    """
    # Генерируем синтетические данные
    np.random.seed(42)
    n_candles = 1000
    price = 50000.0
    candles_list = []

    for i in range(n_candles):
        # Симулируем движение цены
        change = np.random.randn() * 0.02
        price = price * (1 + change)

        # Создаем свечу
        candle = {
            'start': int(time.time()) + i * 60,
            'open': float(price * (1 - np.random.rand() * 0.001)),
            'high': float(price * (1 + np.random.rand() * 0.002)),
            'low': float(price * (1 - np.random.rand() * 0.002)),
            'close': float(price),
            'volume': float(np.random.rand() * 100),
            'turnover': float(price * np.random.rand() * 100)
        }
        candles_list.append(candle)

    # Обучаем модель
    accuracy, samples = model.train(candles_list)
    model.save_model()

    return {
        "message": "Model trained on sample data",
        "accuracy": float(accuracy),
        "samples": samples
    }


@app.get("/model/info")
async def get_model_info():
    """Информация о текущем состоянии модели"""
    return {
        "is_trained": model.is_trained,
        "training_samples": model.training_samples if hasattr(model, 'training_samples') else 0,
        "feature_names": model.feature_names if hasattr(model, 'feature_names') else [],
        "model_type": type(model.model).__name__ if model.model else None
    }


@app.post("/model/reset")
async def reset_model():
    """Сброс модели до начального состояния"""
    model.reset()
    return {"message": "Model reset successfully"}
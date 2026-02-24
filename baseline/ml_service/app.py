import logging
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from schema import Candle, GetPredict

logger = logging.getLogger("ML-Service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для жизненного цикла приложения"""
    logger.info(f"Starting ML-Service")
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
    logger.debug("Health check requested")
    return {"status": "healthy"}

@app.post("/predict", response_model=GetPredict)
async def predict(candle: Candle):
    return GetPredict(prediction = random.choice((-1,0,1)) )

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import asyncio
from core.db import Base, get_engine
from services.candles import fetch_candles
from core.settings import get_app_name, get_environment, get_debug
from core.logging import setup_logging, get_logger
from services.candles import candle_loop
from api.indicator_routes import router 

setup_logging()

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для жизненного цикла приложения"""
    logger.info(f"Starting {get_app_name()}")
    logger.info(f"Environment: {get_environment()}")
    logger.info(f"Debug mode: {get_debug()}")
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Successfully create database")

    asyncio.create_task(candle_loop())
    logger.info(f"Candle loop started")
    yield 
    logger.info(f"Shutting down {get_app_name()}")


app = FastAPI(
    title=get_app_name(),
    debug=get_debug(),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов"""
    logger = get_logger("api")
    
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

import os
from dotenv import load_dotenv

load_dotenv()

def getDatabaseUrl():
    return os.getenv("DATABASE_URL")

def get_bybit_url():
    return "https://api.bybit.com/v5/market/kline"

def get_app_name():
    return os.getenv("APP_NAME")

def get_environment():
    return os.getenv("ENVIRONMENT")

def get_debug():
    return os.getenv("DEBUG")
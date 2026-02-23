import asyncio
import httpx
import pandas as pd

from core.logging import get_logger
from core.settings import get_bybit_url
from state.memory import candles
from services.indicators import recalc_all_indicators

_logger = get_logger("CandlesService")

async def fetch_candles():
    _logger.debug(f"Starting catch candles")

    params = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "interval": "1",
        "limit": 100
    }

    async with httpx.AsyncClient() as client:
        r = await client.get(get_bybit_url(), params=params)
        data = r.json()

    raw = data["result"]["list"]

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df = df.astype(float).sort_values("timestamp")
    candles.clear()
    candles.extend(df.to_dict(orient="records"))

    _logger.debug(f"Finish catch candles, catched {len(df)} candles")
    await recalc_all_indicators()

async def candle_loop():
    while True:
        try:
            await fetch_candles()
        except Exception as e:
            print("fetch error", e)
        await asyncio.sleep(10)
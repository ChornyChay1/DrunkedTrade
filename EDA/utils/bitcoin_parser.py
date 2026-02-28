import os
import httpx
import asyncio
import pandas as pd
from EDA.utils.settings import get_logger
from enum import Enum
from datetime import datetime

logger = get_logger(__name__)


class BitcoinConstants(Enum):
    BYBIT_API_URL = "https://api.bybit.com/v5/market/kline"
    DATA_FOLDER = "../data/bitcoinData"
    BTC = "BTCUSDT"
    INTERVAL = "1"


class BitCoinParser:
    @classmethod
    def get_and_save_candles(cls, start_time: int, end_time: int, interval=BitcoinConstants.INTERVAL.value, filename=None):
        logger.info(f"Запрос свечей BTCUSDT с {start_time} по {end_time}")
        df = cls.get_candles(start_time, end_time, interval)
        if df.empty:
            return df, None
        path = cls.__save_dataset(df, filename)
        logger.info(f"Свечи сохранены в {path}")
        return df, path

    @classmethod
    def get_and_save_full_history(cls, start_time: int, end_time: int, interval=BitcoinConstants.INTERVAL.value, filename=None):
        logger.info(f"Начало загрузки полной истории BTCUSDT с {start_time} по {end_time}")
        df = asyncio.run(cls.__get_full_history(start_time, end_time, interval))
        if df.empty:
            return df, None
        path = cls.__save_dataset(df, filename)
        logger.info(f"Полная история сохранена в {path}")
        return df, path

    @staticmethod
    def __save_dataset(df: pd.DataFrame, filename=None):
        folder = BitcoinConstants.DATA_FOLDER.value
        if not os.path.exists(folder):
            os.makedirs(folder)

        if filename is None:
            filename = f"bitcoin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(folder, filename)
        df.to_csv(path)
        return path

    @classmethod
    def get_candles(cls, start_time: int, end_time: int, interval=BitcoinConstants.INTERVAL.value):
        return asyncio.run(cls.__get_candles_async(start_time, end_time, interval))

    @staticmethod
    async def __get_candles_async(start_time: int, end_time: int, interval: str):
        params = {
            "category": "linear",
            "symbol": BitcoinConstants.BTC.value,
            "interval": interval,
            "start": start_time,
            "end": end_time,
            "limit": 200
        }

        async with httpx.AsyncClient() as client:
            r = await client.get(BitcoinConstants.BYBIT_API_URL.value, params=params)
            data = r.json()

        if data["retCode"] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        raw = data["result"]["list"]
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.astype(float).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        return df

    @staticmethod
    async def __get_full_history(start_time: int, end_time: int, interval: str):
        current_end = end_time
        all_candles = []

        async with httpx.AsyncClient() as client:
            while current_end > start_time:
                params = {
                    "category": "linear",
                    "symbol": BitcoinConstants.BTC.value,
                    "interval": interval,
                    "end": current_end,
                    "limit": 200
                }

                r = await client.get(BitcoinConstants.BYBIT_API_URL.value, params=params)
                data = r.json()

                if data["retCode"] != 0:
                    raise Exception(f"Bybit API error: {data['retMsg']}")

                candles = data["result"]["list"]
                if not candles:
                    break

                for candle in candles:
                    ts = int(candle[0])
                    if ts >= start_time:
                        all_candles.append({
                            "timestamp": ts,
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5])
                        })

                last_ts = int(candles[-1][0])
                if last_ts <= start_time:
                    break

                current_end = last_ts - 60 * 1000
                await asyncio.sleep(0.1)

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        return df

    @classmethod
    def get_candles_sync(cls, start_time: int, end_time: int, interval=BitcoinConstants.INTERVAL.value):
        """Синхронный аналог __get_candles_async"""
        params = {
            "category": "linear",
            "symbol": BitcoinConstants.BTC.value,
            "interval": interval,
            "start": start_time,
            "end": end_time,
            "limit": 200
        }

        r = httpx.get(BitcoinConstants.BYBIT_API_URL.value, params=params)
        data = r.json()

        if data["retCode"] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        raw = data["result"]["list"]
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.astype(float).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        return df

    @classmethod
    def get_full_history_sync(cls, start_time: int, end_time: int, interval=BitcoinConstants.INTERVAL.value):
        """Синхронный аналог __get_full_history"""
        current_end = end_time
        all_candles = []

        while current_end > start_time:
            params = {
                "category": "linear",
                "symbol": BitcoinConstants.BTC.value,
                "interval": interval,
                "end": current_end,
                "limit": 200
            }

            r = httpx.get(BitcoinConstants.BYBIT_API_URL.value, params=params)
            data = r.json()

            if data["retCode"] != 0:
                raise Exception(f"Bybit API error: {data['retMsg']}")

            candles = data["result"]["list"]
            if not candles:
                break

            for candle in candles:
                ts = int(candle[0])
                if ts >= start_time:
                    all_candles.append({
                        "timestamp": ts,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })

            last_ts = int(candles[-1][0])
            if last_ts <= start_time:
                break

            current_end = last_ts - 60 * 1000

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.set_index("timestamp", inplace=True)
        return df

    @classmethod
    def get_and_save_full_history_sync(cls, start_time: int, end_time: int, interval=BitcoinConstants.INTERVAL.value, filename=None):
        """Синхронный аналог get_and_save_full_history"""
        logger.info(f"Начало загрузки полной истории BTCUSDT с {start_time} по {end_time}")
        df = cls.get_full_history_sync(start_time, end_time, interval)
        if df.empty:
            return df, None
        path = cls.__save_dataset(df, filename)
        logger.info(f"Полная история сохранена в {path}")
        return df, path
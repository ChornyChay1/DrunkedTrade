from fastapi import APIRouter, Query
from typing import List, Optional
from state.memory import candles
from utils.indicator_calculator import IndicatorsCalculator, clean
from schemas.indicator import IndicatorQuery
import pandas as pd
import json
import httpx
import asyncio

router = APIRouter(prefix="/symbol", tags=["symbol"])

PREDICT_URL = "http://localhost:8001/predict"
# Должно совпадать с MIN_CANDLES в ml_service/features.py (признаки требуют 61 свечу)
MIN_CANDLES_FOR_PREDICT = 61


def _candle_to_predict_item(c: dict) -> dict:
    """Приводит свечу из state к формату Candle для PredictRequest (поле start вместо timestamp)."""
    return {
        "start": c["timestamp"],
        "open": c["open"],
        "high": c["high"],
        "low": c["low"],
        "close": c["close"],
        "volume": c["volume"],
        "turnover": c["turnover"],
    }


@router.get("/data")
async def get_data(
    indicators: Optional[str] = Query(
        default=None,
        description='JSON list: [{"id":"ema50","type":"ema","period":50}]'
    )
):
    dynamic_values = {}
    indicators_meta: List[IndicatorQuery] = []

    # --- расчет индикаторов ---
    if indicators:
        indicators_meta = [IndicatorQuery(**i) for i in json.loads(indicators)]
        close_series = pd.Series([c["close"] for c in candles])
        for ind in indicators_meta:
            values = IndicatorsCalculator.calculate(ind.type, close_series, ind.period)
            dynamic_values[ind.id] = clean(values)

    # --- предсказания: PredictRequest ожидает список свечей (от старых к новым), предсказание для последней ---
    async def fetch_action(client, end_index: int):
        if end_index < MIN_CANDLES_FOR_PREDICT - 1:
            return 0
        payload = {
            "candles": [_candle_to_predict_item(c) for c in candles[: end_index + 1]],
        }
        try:
            resp = await client.post(PREDICT_URL, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return data.get("prediction", 0)
        except Exception:
            return None

    async with httpx.AsyncClient() as client:
        actions = await asyncio.gather(
            *[fetch_action(client, i) for i in range(len(candles))]
        )

    # --- объединение и аналитика ---
    merged = []
    total_buy = 0
    total_sell = 0
    sum_buy = 0
    sum_sell = 0

    for i, c in enumerate(candles):
        row = dict(c)
        row["indicators"] = {
            ind_id: dynamic_values[ind_id][i]
            for ind_id in dynamic_values
            if i < len(dynamic_values[ind_id])
        }
        action = actions[i] if actions[i] is not None else 0
        row["action"] = action

        if action == 1:
            total_buy += 1
            sum_buy += c["close"]
        elif action == -1:
            total_sell += 1
            sum_sell += c["close"]

        merged.append(row)

    avg_buy = sum_buy / total_buy if total_buy > 0 else 0
    avg_sell = sum_sell / total_sell if total_sell > 0 else 0
    profit = avg_buy - avg_sell  # простая прибыль = сумма покупок минус продаж

    analytics = {
        "total_buy": total_buy,
        "total_sell": total_sell,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
        "avg_profit": profit
    }

    return {
        "candles": merged,
        "indicators": indicators_meta,
        "analytics": analytics
    }
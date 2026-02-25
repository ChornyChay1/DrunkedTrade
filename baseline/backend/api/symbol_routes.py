from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from state.memory import candles
from utils.indicator_calculator import IndicatorsCalculator, clean
from schemas.indicator import IndicatorQuery
import pandas as pd
import json
import httpx
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/symbol", tags=["symbol"])

PREDICT_URL = "http://localhost:8001/predict"  # URL вашего ML сервиса

# Константы для стратегии
HISTORY_LENGTH = 20  # Количество свечей для анализа (можно настроить)
MIN_CANDLES = 10  # Минимальное количество свечей для предсказания


@router.get("/data")
async def get_data(
        indicators: Optional[str] = Query(
            default=None,
            description='JSON list: [{"id":"ema50","type":"ema","period":50}]'
        )
):
    if len(candles) < MIN_CANDLES:
        return {
            "candles": candles[-50:],  # Возвращаем последние 50 свечей
            "indicators": [],
            "analytics": {
                "total_buy": 0,
                "total_sell": 0,
                "avg_buy": 0,
                "avg_sell": 0,
                "avg_profit": 0,
                "message": f"Need at least {MIN_CANDLES} candles for predictions"
            }
        }

    dynamic_values = {}
    indicators_meta: List[IndicatorQuery] = []

    # --- расчет индикаторов ---
    if indicators:
        indicators_meta = [IndicatorQuery(**i) for i in json.loads(indicators)]
        close_series = pd.Series([c["close"] for c in candles])
        for ind in indicators_meta:
            values = IndicatorsCalculator.calculate(ind.type, close_series, ind.period)
            dynamic_values[ind.id] = clean(values)

    # --- функция запроса predict с историей ---
    async def fetch_action(client, index):
        """
        Отправляем запрос в ML сервис с историей свечей
        """
        # Берем HISTORY_LENGTH свечей до текущей (или меньше, если в начале)
        start_idx = max(0, index - HISTORY_LENGTH + 1)
        history_candles = candles[start_idx:index + 1]

        # Формируем список свечей для предсказания
        candles_for_prediction = []
        for c in history_candles:
            candles_for_prediction.append({
                "start": c["timestamp"],  # Используем timestamp как start
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"]),
                "turnover": float(c["turnover"])
            })

        payload = {
            "candles": candles_for_prediction  # Отправляем список свечей
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(PREDICT_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                prediction = data.get("prediction", 0)

                # Логируем результат для отладки
                if prediction is not None:
                    logger.debug(
                        f"Index {index}: Prediction={prediction}, History={len(candles_for_prediction)} candles")
                else:
                    logger.debug(f"Index {index}: No prediction, using default 0")

                return prediction if prediction is not None else 0
        except httpx.TimeoutException:
            logger.warning(f"Timeout for index {index}, using default 0")
            return 0
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error for index {index}: {e}, using default 0")
            return 0
        except Exception as e:
            logger.error(f"Error for index {index}: {str(e)}, using default 0")
            return 0

    # --- параллельные запросы к ML сервису ---
    tasks = []
    for i in range(len(candles)):
        tasks.append(fetch_action(None, i))

    # Выполняем все запросы параллельно
    actions = await asyncio.gather(*tasks, return_exceptions=True)

    # Обрабатываем результаты - заменяем все None и исключения на 0
    processed_actions = []
    for action in actions:
        if isinstance(action, Exception) or action is None:
            processed_actions.append(0)
        else:
            processed_actions.append(action)

    # --- ФИЛЬТРАЦИЯ СИГНАЛОВ: убираем повторяющиеся sell-sell и buy-buy ---
    filtered_actions = []
    last_signal = 0  # 0 - нет позиции, 1 - последний был buy, -1 - последний был sell

    for action in processed_actions:
        if action == 1:  # BUY сигнал
            if last_signal != 1:  # Если последний был не BUY
                filtered_actions.append(1)
                last_signal = 1
            else:
                filtered_actions.append(0)  # Заменяем повторный BUY на HOLD
                logger.debug(f"Filtered out duplicate BUY signal")
        elif action == -1:  # SELL сигнал
            if last_signal != -1:  # Если последний был не SELL
                filtered_actions.append(-1)
                last_signal = -1
            else:
                filtered_actions.append(0)  # Заменяем повторный SELL на HOLD
                logger.debug(f"Filtered out duplicate SELL signal")
        else:  # 0 сигнал
            filtered_actions.append(0)
            # Не меняем last_signal при HOLD

    # --- объединение и аналитика ---
    merged = []
    total_buy = 0
    total_sell = 0
    sum_buy = 0.0
    sum_sell = 0.0
    successful_predictions = 0

    # Счетчик для отслеживания последовательности buy-sell-buy-sell
    buy_positions = []  # Список цен покупок
    sell_positions = []  # Список цен продаж
    current_position = 0  # 0 - нет позиции, 1 - куплено

    for i, c in enumerate(candles):
        row = dict(c)

        # Добавляем индикаторы - заменяем None на 0
        row["indicators"] = {
            ind_id: dynamic_values[ind_id][i] if i < len(dynamic_values[ind_id]) and dynamic_values[ind_id][
                i] is not None else 0
            for ind_id in dynamic_values
        }

        # Добавляем отфильтрованное действие
        action = filtered_actions[i]
        row["action"] = action
        row["original_action"] = processed_actions[i]  # Для отладки можно добавить оригинальный сигнал

        # Статистика и логика чередования buy-sell
        if action == 1:  # BUY сигнал
            total_buy += 1
            sum_buy += float(c["close"])
            successful_predictions += 1
            buy_positions.append(float(c["close"]))
            current_position = 1
        elif action == -1:  # SELL сигнал
            total_sell += 1
            sum_sell += float(c["close"])
            successful_predictions += 1
            sell_positions.append(float(c["close"]))
            current_position = 0
        elif action == 0:
            successful_predictions += 1  # HOLD тоже считаем успешным предсказанием

        merged.append(row)

    # Расчет средней прибыли по стратегии с учетом чередования
    avg_buy = sum_buy / total_buy if total_buy > 0 else 0
    avg_sell = sum_sell / total_sell if total_sell > 0 else 0

    # Расчет прибыли по парным сделкам (buy-sell пары)
    pair_profit = 0.0
    pair_count = min(len(buy_positions), len(sell_positions))

    if pair_count > 0:
        for j in range(pair_count):
            pair_profit += sell_positions[j] - buy_positions[j]
        avg_pair_profit = pair_profit / pair_count
    else:
        avg_pair_profit = 0.0

    analytics = {
        "total_candles": len(candles),
        "total_buy_signals": total_buy,
        "total_sell_signals": total_sell,
        "total_hold_signals": len(candles) - successful_predictions,
        "avg_buy_price": round(avg_buy, 2),
        "avg_sell_price": round(avg_sell, 2),
        "strategy_profit": round(avg_sell - avg_buy if total_buy > 0 and total_sell > 0 else 0, 2),
        "pair_profit": round(avg_pair_profit, 2),  # Прибыль по парам buy-sell
        "completed_trades": pair_count,  # Количество завершенных сделок
        "successful_predictions": successful_predictions,
        "prediction_coverage": f"{(successful_predictions / len(candles) * 100):.1f}%",
        "ml_service_url": PREDICT_URL,
        "filtered_signals": len([a for a in filtered_actions if a != 0]),
        # Количество ненулевых сигналов после фильтрации
        "original_signals": len([a for a in processed_actions if a != 0])  # Количество ненулевых сигналов до фильтрации
    }

    # Логируем аналитику
    logger.info(f"Analytics: {analytics}")
    logger.info(f"Buy-Sell sequence: Buys={total_buy}, Sells={total_sell}, Pairs={pair_count}")

    return {
        "candles": merged[-100:],
        "indicators": indicators_meta,
        "analytics": analytics
    }


@router.post("/train-model")
async def train_ml_model():
    """
    Эндпоинт для обучения ML модели на всех доступных свечах
    """
    if len(candles) < 50:
        raise HTTPException(status_code=400, detail="Need at least 50 candles for training")

    # Формируем данные для обучения
    training_candles = []
    for c in candles[-500:]:  # Берем последние 500 свечей для обучения
        training_candles.append({
            "start": c["timestamp"],
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["volume"]),
            "turnover": float(c["turnover"])
        })

    # Отправляем запрос на обучение
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "http://localhost:8001/train",
                json=training_candles
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/model-info")
async def get_model_info():
    """
    Получить информацию о текущем состоянии ML модели
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:8001/health")
            if resp.status_code == 200:
                return {
                    "status": "connected",
                    "message": "ML service is running"
                }
            else:
                return {
                    "status": "error",
                    "message": "ML service returned error"
                }
    except Exception as e:
        return {
            "status": "disconnected",
            "message": f"Cannot connect to ML service: {str(e)}"
        }
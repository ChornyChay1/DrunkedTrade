# tests/test_indicators.py
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from models.indicator import IndicatorDB
from schemas.indicator import IndicatorCreate, IndicatorUpdate
from services.candles import fetch_candles
from services.indicators import recalc_indicator, clean
from state.memory import candles, indicator_values
from utils.constants import Indicators


@pytest.mark.asyncio
async def test_recalc_indicator(sample_ohlc, mock_candles):
    """Тест пересчета одного индикатора"""
    indicator = IndicatorDB(
        id=1,
        name="SMA 14",
        type=Indicators.SMA,
        period=14,
        color="#FF0000"
    )

    # Вызываем функцию
    await recalc_indicator(indicator)

    # Проверяем, что результат сохранился
    assert indicator.id in indicator_values


@pytest.mark.asyncio
async def test_recalc_indicator_no_candles(sample_ohlc):
    """Тест пересчета индикатора когда нет свечей"""
    candles.clear()

    indicator = IndicatorDB(
        id=1,
        name="SMA 14",
        type=Indicators.SMA,
        period=14,
        color="#FF0000"
    )

    await recalc_indicator(indicator)

    assert indicator.id in indicator_values
    assert indicator_values[indicator.id] == []


@pytest.mark.asyncio
async def test_recalc_indicator_different_types(sample_ohlc, mock_candles):
    """Тест пересчета разных типов индикаторов"""
    indicator_types = [
        (Indicators.SMA, "SMA 14"),
        (Indicators.EMA, "EMA 14"),
        (Indicators.WMA, "WMA 14")
    ]

    for ind_type, name in indicator_types:
        indicator = IndicatorDB(
            id=hash(ind_type.value),  # уникальный ID для каждого типа
            name=name,
            type=ind_type,
            period=14,
            color="#FF0000"
        )

        await recalc_indicator(indicator)

        # Проверяем что результат сохранился
        assert indicator.id in indicator_values
        assert len(indicator_values[indicator.id]) == len(candles)


def test_clean_function():
    """Тест функции очистки значений"""
    test_values = [
        1.5,                    # нормальное значение
        None,                   # None
        float('nan'),           # NaN
        2.7,                    # нормальное значение
        float('inf'),           # бесконечность
        float('-inf'),          # отрицательная бесконечность
        3.2,                    # нормальное значение
        True,
        False
    ]

    cleaned = clean(test_values)

    assert cleaned[0] == 1.5
    assert cleaned[1] is None
    assert cleaned[2] is None
    assert cleaned[3] == 2.7
    assert cleaned[4] is None
    assert cleaned[5] is None
    assert cleaned[6] == 3.2
    assert cleaned[7] == 1.0  # True преобразуется в 1.0
    assert cleaned[8] == 0.0  # False преобразуется в 0.0
    assert len(cleaned) == len(test_values)


def test_indicator_create_schema():
    """Тест схемы создания индикатора"""
    # Тест с обязательными полями
    indicator = IndicatorCreate(
        name="Test SMA",
        type=Indicators.SMA,
        period=14
    )
    assert indicator.name == "Test SMA"
    assert indicator.type == Indicators.SMA
    assert indicator.period == 14
    assert indicator.color is None

    indicator = IndicatorCreate(
        name="Test EMA",
        type=Indicators.EMA,
        period=20,
        color="#00FF00"
    )
    assert indicator.name == "Test EMA"
    assert indicator.type == Indicators.EMA
    assert indicator.period == 20
    assert indicator.color == "#00FF00"

    indicator = IndicatorCreate(
        name="Test WMA",
        type=Indicators.WMA,
        period=15,
        color="#0000FF"
    )
    assert indicator.name == "Test WMA"
    assert indicator.type == Indicators.WMA
    assert indicator.period == 15
    assert indicator.color == "#0000FF"


def test_indicator_update_schema():
    """Тест схемы обновления индикатора"""
    # Тест с частичным обновлением
    update = IndicatorUpdate(name="New Name")
    assert update.name == "New Name"
    assert update.type is None
    assert update.period is None
    assert update.color is None

    # Тест с обновлением типа
    update = IndicatorUpdate(type=Indicators.EMA)
    assert update.name is None
    assert update.type == Indicators.EMA
    assert update.period is None
    assert update.color is None

    # Тест с обновлением периода
    update = IndicatorUpdate(period=21)
    assert update.name is None
    assert update.type is None
    assert update.period == 21
    assert update.color is None

    # Тест с обновлением цвета
    update = IndicatorUpdate(color="#FF0000")
    assert update.name is None
    assert update.type is None
    assert update.period is None
    assert update.color == "#FF0000"

    # Тест с обновлением всех полей
    update = IndicatorUpdate(
        name="Updated",
        type=Indicators.WMA,
        period=21,
        color="#FF0000"
    )
    assert update.name == "Updated"
    assert update.type == Indicators.WMA
    assert update.period == 21
    assert update.color == "#FF0000"


@pytest.mark.asyncio
async def test_fetch_candles():
    """Тест получения свечей с биржи"""
    # Мокаем httpx клиент
    mock_response = {
        "result": {
            "list": [
                ["1625097600000", "50000", "51000", "49000", "50500", "100", "5000000"],
                ["1625184000000", "50500", "52000", "50000", "51500", "150", "7500000"],
                ["1625270400000", "51500", "53000", "51000", "52500", "200", "10000000"]
            ]
        }
    }

    with patch('httpx.AsyncClient') as mock_client:
        # Настраиваем мок клиента
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Создаем мок ответа
        mock_response_obj = MagicMock()  # Используем MagicMock для синхронного объекта
        mock_response_obj.json.return_value = mock_response

        # get - асинхронный метод, возвращает корутину с response
        mock_client_instance.get = AsyncMock(return_value=mock_response_obj)

        # Очищаем candles перед тестом
        candles.clear()
        indicator_values.clear()

        # Мокаем recalc_all_indicators чтобы избежать реального пересчета
        with patch('services.candles.recalc_all_indicators', new_callable=AsyncMock) as mock_recalc:
            # Вызываем функцию
            await fetch_candles()

            # Проверяем, что свечи добавились
            assert len(candles) == 3
            assert "timestamp" in candles[0]
            assert "open" in candles[0]
            assert "high" in candles[0]
            assert "low" in candles[0]
            assert "close" in candles[0]
            assert "volume" in candles[0]
            assert "turnover" in candles[0]

            # Проверяем что значения преобразованы в float
            assert isinstance(candles[0]["close"], float)
            assert candles[0]["close"] == 50500.0

            # Проверяем сортировку по timestamp
            assert candles[0]["timestamp"] < candles[1]["timestamp"]

            # Проверяем что индикаторы пересчитаны
            mock_recalc.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_candles_empty_response():
    """Тест получения свечей с пустым ответом от биржи"""
    mock_response = {
        "result": {
            "list": []
        }
    }

    with patch('httpx.AsyncClient') as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Исправляем: json должен быть синхронным методом
        mock_response_obj = MagicMock()  # MagicMock для синхронного объекта
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.status_code = 200

        # get - асинхронный метод
        mock_client_instance.get = AsyncMock(return_value=mock_response_obj)

        candles.clear()
        indicator_values.clear()

        with patch('services.candles.recalc_all_indicators', new_callable=AsyncMock) as mock_recalc:
            await fetch_candles()

            # Проверяем что свечи не добавились
            assert len(candles) == 0
            # Индикаторы все равно пересчитываются
            mock_recalc.assert_called_once()


def test_indicator_constants():
    """Тест констант индикаторов"""
    assert Indicators.SMA == "sma"
    assert Indicators.EMA == "ema"
    assert Indicators.WMA == "wma"

    # Проверяем что все значения уникальны
    values = [v.value for v in Indicators]
    assert len(values) == len(set(values))
    assert len(values) == 3  # только SMA, EMA, WMA


@pytest.mark.asyncio
async def test_recalc_indicator_with_invalid_type(sample_ohlc, mock_candles):
    """Тест пересчета индикатора с неверным типом"""
    # Создаем индикатор с неверным типом
    indicator = IndicatorDB(
        id=1,
        name="Invalid",
        type="invalid_type",  # неверный тип
        period=14,
        color="#FF0000"
    )

    # Вызываем функцию (должна обработать ошибку)
    await recalc_indicator(indicator)

    # Проверяем что результат все равно сохранился (возможно с ошибкой)
    assert indicator.id in indicator_values
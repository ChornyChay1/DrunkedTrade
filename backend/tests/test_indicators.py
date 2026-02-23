import pytest
import pandas as pd
import numpy as np
from services.indicators import IndicatorsCalculator
from utils.constants import Indicators

# --------------------------
# Фикстуры
# --------------------------
@pytest.fixture
def sample_series():
    """Тестовая серия для индикаторов"""
    return pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])

@pytest.fixture
def mock_candles(monkeypatch):
    """Мокаем candles для calculate()"""
    from state import memory
    memory.candles = [{"close": v} for v in range(10)]
    yield
    memory.candles = []


# =========================
# SMA
# =========================
def test_sma_basic(sample_series):
    period = 5
    sma = IndicatorsCalculator.calc_sma(sample_series, period)

    assert len(sma) == len(sample_series)
    assert isinstance(sma, list)

    # Первые period-1 значений должны быть NaN
    for i in range(period-1):
        assert pd.isna(sma[i])

    # Проверка первого вычисленного значения
    expected = sample_series.iloc[:period].mean()
    assert sma[period-1] == expected


def test_sma_empty():
    sma = IndicatorsCalculator.calc_sma(pd.Series(dtype=float), 5)
    assert sma == []


# =========================
# EMA
# =========================
def test_ema_formula(sample_series):
    period = 5
    ema = IndicatorsCalculator.calc_ema(sample_series, period)

    # Первое значение = первый элемент серии
    assert ema[0] == sample_series.iloc[0]

    # Проверка формулы EMA для второго значения
    k = 2 / (period + 1)
    expected = (sample_series.iloc[1] - ema[0]) * k + ema[0]
    assert abs(ema[1] - expected) < 1e-6


# =========================
# WMA
# =========================
def test_wma_manual(sample_series):
    period = 5
    wma = IndicatorsCalculator.calc_wma(sample_series, period)

    weights = np.arange(1, period+1)
    expected = np.dot(sample_series.iloc[:period], weights) / weights.sum()

    # Проверка пятого значения
    assert abs(wma.iloc[period-1] - expected) < 1e-6


# =========================
# calculate() для close-only
# =========================
def test_calculate_close_only(sample_series, mock_candles):
    period = 5

    sma = IndicatorsCalculator.calculate(Indicators.SMA, sample_series, period)
    ema = IndicatorsCalculator.calculate(Indicators.EMA, sample_series, period)
    wma = IndicatorsCalculator.calculate(Indicators.WMA, sample_series, period)

    assert len(sma) == len(sample_series)
    assert len(ema) == len(sample_series)
    assert len(wma) == len(sample_series)


# =========================
# Проверка NaN в начале
# =========================
@pytest.mark.parametrize(
    "indicator,expected_nans",
    [
        (Indicators.SMA, 4),  # period-1
        (Indicators.EMA, 0),
        (Indicators.WMA, 4),  # period-1
    ],
)
def test_nan_patterns(sample_series, indicator, expected_nans, mock_candles):
    result = IndicatorsCalculator.calculate(indicator, sample_series, 5)

    if isinstance(result, pd.Series):
        nans = result.isna().sum()
    else:
        nans = sum(pd.isna(x) for x in result)

    assert nans == expected_nans

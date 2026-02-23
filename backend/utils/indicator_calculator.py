import pandas as pd
import numpy as np
from state.memory import candles
from utils.constants import Indicators
from core.logging import get_logger
class IndicatorsCalculator: 
    _logger = get_logger("IndicatorCalculator")


    @staticmethod
    def calc_sma(series, period):
        return series.rolling(period).mean().tolist()

    @staticmethod
    def calc_ema(series, period):
        return series.ewm(span=period, adjust=False).mean().tolist()

    @staticmethod
    def calc_wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    
    @classmethod
    def calculate(cls, ind_type, close, period):
        values = []
        try:
            if ind_type == Indicators.SMA:
                values = cls.calc_sma(close, period)
            
            elif ind_type == Indicators.EMA:
                values = cls.calc_ema(close, period)

            elif ind_type == Indicators.WMA:
                values = cls.calc_wma(close, period)

            cls._logger.debug(f"The indicator {ind_type} value has been calculated successfully: {values}")
        
        except Exception as exception:
            cls._logger.exception(f"Exception {exception} while calculating indicator with type {ind_type}")
            
        return values
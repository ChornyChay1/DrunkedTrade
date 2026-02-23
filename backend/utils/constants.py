from enum import Enum


class Indicators(str, Enum):
    """Поддерживаемые типы индикаторов"""
    
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
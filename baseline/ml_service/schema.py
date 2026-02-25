from pydantic import BaseModel, Field


class Candle(BaseModel):
    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class PredictRequest(BaseModel):
    """Список свечей (от старых к новым) для расчёта признаков с лагами и скользящими окнами."""

    candles: list[Candle] = Field(..., min_length=1, description="Свечи в хронологическом порядке")


class GetPredict(BaseModel):
    prediction: int
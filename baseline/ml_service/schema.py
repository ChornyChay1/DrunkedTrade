from pydantic import BaseModel, Field


class Candle(BaseModel):
    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

class GetPredict(BaseModel):
    prediction: int
from pydantic import BaseModel, Field
from typing import List, Optional

class Candle(BaseModel):
    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

class PredictRequest(BaseModel):
    candles: List[Candle]

class GetPredict(BaseModel):
    prediction: int  # -1, 0, или 1

class TrainResponse(BaseModel):
    message: str
    accuracy: float
    samples: int
    filename: Optional[str] = None
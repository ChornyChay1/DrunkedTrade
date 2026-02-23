import math
import pandas as pd

from sqlalchemy import select
from core.db import get_session_local
from models.indicator import IndicatorDB
from state.memory import candles, indicator_values
from utils.indicator_calculator import IndicatorsCalculator

def clean(values):
    return [
        None if (v is None or not math.isfinite(v)) else float(v)
        for v in values
    ]


async def recalc_indicator(ind: IndicatorDB):
    if not candles:
        indicator_values[ind.id] = []
        return

    df = pd.DataFrame(candles)
    period = ind.period or 14
    values = IndicatorsCalculator.calculate(ind.type, df["close"], period)
    indicator_values[ind.id] = values


async def recalc_all_indicators():
    SessionLocal = get_session_local()
    async with SessionLocal() as session:
        result = await session.execute(select(IndicatorDB))
        inds = result.scalars().all()

    for ind in inds:
        await recalc_indicator(ind)

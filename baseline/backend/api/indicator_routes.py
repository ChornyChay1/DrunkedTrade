# from core.db import get_session_local
# from fastapi import APIRouter, HTTPException
# from models.indicator import IndicatorDB
# from schemas.indicator import IndicatorCreate, IndicatorUpdate
# from services.indicators import recalc_indicator, clean
# from sqlalchemy import select, delete
# from state.memory import candles, indicator_values
#
# router = APIRouter(prefix="/indicators", tags=["indicators"])
#

# @router.post("/indicator")
# async def create_indicator(ind: IndicatorCreate):
#     db_ind = IndicatorDB(name=ind.name, type=ind.type, period=ind.period, color=ind.color)
#     SessionLocal = get_session_local()
#     async with SessionLocal() as session:
#         session.add(db_ind)
#         await session.commit()
#         await session.refresh(db_ind)
#
#     await recalc_indicator(db_ind)
#
#     return {"id": db_ind.id}
#
# @router.put("/indicator/{ind_id}")
# async def update_indicator(ind_id: int, upd: IndicatorUpdate):
#     SessionLocal = get_session_local()
#     async with SessionLocal() as session:
#         result = await session.execute(
#             select(IndicatorDB).where(IndicatorDB.id == ind_id)
#         )
#         ind = result.scalar_one_or_none()
#
#         if not ind:
#             raise HTTPException(404)
#
#         if upd.name:
#             ind.name = upd.name
#         if upd.period:
#             ind.period = upd.period
#         if upd.type:
#             ind.type = upd.type
#         if upd.color:
#             ind.color = upd.color
#
#         await session.commit()
#
#     await recalc_indicator(ind)
#     return {"status": "updated"}
#
#
# @router.delete("/indicator/{ind_id}")
# async def delete_indicator(ind_id: int):
#     SessionLocal = get_session_local()
#     async with SessionLocal() as session:
#         await session.execute(delete(IndicatorDB).where(IndicatorDB.id == ind_id))
#         await session.commit()
#
#     indicator_values.pop(ind_id, None)
#     return {"status": "deleted"}

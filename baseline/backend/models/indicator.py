from sqlalchemy import Column, String, Integer
from core.db import Base

class IndicatorDB(Base):
    __tablename__ = "indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    period = Column(Integer, nullable=False)
    color = Column(String, nullable=False, default="#000")


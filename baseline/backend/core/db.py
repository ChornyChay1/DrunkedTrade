# core/db.py
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

from core.settings import getDatabaseUrl


def get_engine():
    database_url = getDatabaseUrl()
    if not database_url:
        raise RuntimeError("DATABASE_URL not set")
    return create_async_engine(database_url, echo=False)

def get_session_local():
    engine = get_engine()
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

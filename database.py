"""
Database connection and management utilities using SQLAlchemy.

Adapted from Text to SQL/database.py.
"""

import os

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "app.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"


def quote_identifier(identifier: str) -> str:
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'


def get_engine() -> Engine:
    return create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )


def get_sql_database() -> SQLDatabase:
    return SQLDatabase(engine=get_engine())


def list_tables() -> list[str]:
    engine = get_engine()
    inspector = inspect(engine)
    return inspector.get_table_names()


def drop_all_tables():
    engine = get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    with engine.begin() as conn:
        for table in tables:
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {quote_identifier(table)}")
    return len(tables)


def run_query(sql: str) -> list:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return result.fetchall()

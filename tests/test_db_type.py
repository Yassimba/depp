"""Verify db_type() returns the underlying adapter type, not 'depp'."""

from unittest.mock import MagicMock

from dbt.adapters.depp.adapter import DeppAdapter


def test_db_type_returns_snowflake() -> None:
    adapter = object.__new__(DeppAdapter)
    adapter._db_adapter = MagicMock()
    adapter._db_adapter.type.return_value = "snowflake"

    assert adapter.db_type() == "snowflake"


def test_db_type_returns_postgres() -> None:
    adapter = object.__new__(DeppAdapter)
    adapter._db_adapter = MagicMock()
    adapter._db_adapter.type.return_value = "postgres"

    assert adapter.db_type() == "postgres"

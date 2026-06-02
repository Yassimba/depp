"""Array-column type classification for Python-model writes.

Regression coverage for the ARRAY-of-BIGINT bug: before ArrayKind, a list
column was a bare ``is_int`` bool, so int64 values above the int32 ceiling were
still flagged "int" and the post-write cast emitted ``INTEGER[]``. Postgres then
raised ``value "..." is out of range for type integer``. These tests pin the
classifier to ``BIGINT`` for those values, across the polars and pandas paths,
and assert the DDL that the writer ultimately runs.
"""

import polars as pl

from dbt.adapters.depp.db.base import ArrayKind
from dbt.adapters.depp.db.postgres import PostgresOps
from dbt.adapters.depp.executors.converters import (
    PolarsConverter,
    detect_pandas_array_columns,
)

INT32_MAX = 2**31 - 1
ABOVE_INT32 = INT32_MAX + 1  # the value that broke INTEGER[]

OPS = PostgresOps()


def _polars_kind(values: list, dtype: pl.DataType) -> ArrayKind:
    df = pl.DataFrame({"a": pl.Series([values], dtype=pl.List(dtype))})
    _, cols = PolarsConverter().prepare_array_columns(df, OPS)
    return cols["a"]


def _pandas_kind(values: list) -> ArrayKind:
    import pandas as pd

    df = pd.DataFrame({"a": pd.Series([values], dtype="object")})
    _, cols = detect_pandas_array_columns(df, OPS)
    return cols["a"]


# --- polars path -----------------------------------------------------------


def test_polars_int32_range_is_int() -> None:
    assert _polars_kind([1, 2, 3], pl.Int32) == ArrayKind.INT


def test_polars_int64_above_int32_is_bigint() -> None:
    # The regression: this exact value used to be cast to INTEGER[] and crash.
    assert _polars_kind([ABOVE_INT32], pl.Int64) == ArrayKind.BIGINT


def test_polars_uint32_is_bigint() -> None:
    # UInt32 max (~4.29e9) exceeds signed int32, so it must widen to BIGINT.
    assert _polars_kind([4_000_000_000], pl.UInt32) == ArrayKind.BIGINT


def test_polars_uint64_overflowing_bigint_falls_back_to_text() -> None:
    assert _polars_kind([2**63], pl.UInt64) == ArrayKind.TEXT


def test_polars_string_is_text() -> None:
    assert _polars_kind(["a", "b"], pl.String) == ArrayKind.TEXT


# --- pandas path -----------------------------------------------------------


def test_pandas_int32_range_is_int() -> None:
    assert _pandas_kind([1, 2, 3]) == ArrayKind.INT


def test_pandas_value_above_int32_is_bigint() -> None:
    assert _pandas_kind([ABOVE_INT32]) == ArrayKind.BIGINT


def test_pandas_float_is_text() -> None:
    assert _pandas_kind([1.5, 2.5]) == ArrayKind.TEXT


def test_pandas_bool_is_text() -> None:
    # bool is an int subclass in Python; SQL semantics say it is not an array of int.
    assert _pandas_kind([True, False]) == ArrayKind.TEXT


# --- resulting DDL ---------------------------------------------------------


def test_bigint_kind_emits_bigint_array_cast() -> None:
    dtype = OPS.array_type(ArrayKind.BIGINT)
    sql = OPS.post_write_sql("s", "t", "c", dtype)
    assert dtype == "BIGINT[]"
    assert "TYPE BIGINT[] USING c::BIGINT[]" in sql


def test_int_kind_emits_integer_array_cast() -> None:
    assert OPS.array_type(ArrayKind.INT) == "INTEGER[]"

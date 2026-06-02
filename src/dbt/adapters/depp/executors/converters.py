from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import shapely

from dbt.adapters.depp.db import DatabaseOps
from dbt.adapters.depp.db.base import ArrayKind

from .protocols import GeoArrowResult

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)
INT64_MAX = 2**63 - 1
_POLARS_INT32_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16}
_POLARS_BIGINT_TYPES = {pl.Int64, pl.UInt32}


def is_array_like(x: object) -> bool:
    """Check if a value is a list or numpy array."""
    return isinstance(x, (list, np.ndarray))


def _pandas_array_kind(col: pd.Series) -> ArrayKind:
    """Classify a pandas list-column's element type into an ArrayKind."""
    flat = (
        col.dropna()
        .map(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        .explode()
        .dropna()
    )
    if flat.empty:
        return ArrayKind.TEXT
    # Bools are ints in Python; reject them explicitly to match SQL semantics.
    if flat.map(lambda x: isinstance(x, bool)).any():
        return ArrayKind.TEXT
    numeric = pd.to_numeric(flat, errors="coerce")
    if numeric.isna().any() or (numeric % 1 != 0).any():
        return ArrayKind.TEXT
    if (numeric > INT32_MAX).any() or (numeric < INT32_MIN).any():
        return ArrayKind.BIGINT
    return ArrayKind.INT


def _polars_array_kind(series: pl.Series) -> ArrayKind:
    """Classify a Polars list-column's inner dtype into an ArrayKind.

    UInt64 is checked at runtime because its max exceeds signed BIGINT.
    """
    dtype = series.dtype
    if not isinstance(dtype, pl.List):
        return ArrayKind.TEXT
    inner = dtype.inner
    if inner in _POLARS_INT32_TYPES:
        return ArrayKind.INT
    if inner in _POLARS_BIGINT_TYPES:
        return ArrayKind.BIGINT
    if inner == pl.UInt64:
        max_val = cast(int | None, series.explode().max())
        if max_val is None or max_val <= INT64_MAX:
            return ArrayKind.BIGINT
        return ArrayKind.TEXT
    return ArrayKind.TEXT


def detect_pandas_array_columns(
    df: pd.DataFrame, db_ops: DatabaseOps
) -> tuple[pd.DataFrame, dict[str, ArrayKind]]:
    """Detect object-dtype columns containing lists and format them."""
    array_cols: dict[str, ArrayKind] = {}
    for col in df.select_dtypes("object").columns:
        if not df[col].apply(is_array_like).any():
            continue
        kind = _pandas_array_kind(df[col])
        df[col] = df[col].apply(
            lambda x, ops=db_ops: (
                ops.format_array(list(x)) if is_array_like(x) else None
            )
        )
        array_cols[str(col)] = kind
    return df, array_cols


class PolarsConverter:
    """Convert between Arrow and Polars DataFrames."""

    __slots__ = ()

    def from_arrow(self, arrow: Any) -> pl.DataFrame:
        """Convert an Arrow table to a Polars DataFrame."""
        return cast(pl.DataFrame, pl.from_arrow(arrow))

    def prepare_array_columns(
        self, df: pl.DataFrame, db_ops: DatabaseOps
    ) -> tuple[pl.DataFrame, dict[str, ArrayKind]]:
        """Detect and format list columns for database insertion."""
        array_cols: dict[str, ArrayKind] = {}
        for col in df.columns:
            series = df[col]
            if not isinstance(series.dtype, pl.List):
                continue
            kind = _polars_array_kind(series)
            values = series.to_list()
            df = df.with_columns(
                pl.Series(
                    col,
                    [db_ops.format_array(v) if v is not None else None for v in values],
                )
            )
            array_cols[col] = kind
        return df, array_cols

    def to_arrow(self, df: pl.DataFrame) -> pa.Table:
        """Convert a Polars DataFrame to Arrow (near zero-copy)."""
        return df.to_arrow()

    def to_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return the DataFrame as-is (already Polars)."""
        return df

    def to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        """Convert a Polars DataFrame to Pandas."""
        return df.to_pandas(use_pyarrow_extension_array=True)


class PandasConverter:
    """Convert between Arrow and Pandas DataFrames."""

    __slots__ = ()

    def from_arrow(self, arrow: Any) -> pd.DataFrame:
        """Convert an Arrow table to a Pandas DataFrame."""
        return cast(pl.DataFrame, pl.from_arrow(arrow)).to_pandas()

    def prepare_array_columns(
        self, df: pd.DataFrame, db_ops: DatabaseOps
    ) -> tuple[pd.DataFrame, dict[str, ArrayKind]]:
        """Detect and format list columns for database insertion."""
        return detect_pandas_array_columns(df, db_ops)

    def to_arrow(self, df: pd.DataFrame) -> pa.Table:
        """Convert a Pandas DataFrame to Arrow."""
        return pa.Table.from_pandas(df)

    def to_polars(self, df: pd.DataFrame) -> pl.DataFrame:
        """Convert a Pandas DataFrame to Polars."""
        return pl.from_pandas(df)

    def to_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the DataFrame as-is (already Pandas)."""
        return df


class GeoPandasConverter:
    """Convert between Arrow/GeoArrowResult and GeoDataFrames."""

    __slots__ = ()

    def from_arrow(self, arrow: Any) -> Any:
        """Convert a GeoArrowResult to a GeoDataFrame."""
        result = cast(GeoArrowResult, arrow)
        df: pd.DataFrame = result.table.to_pandas()
        if not result.geometry_columns:
            return gpd.GeoDataFrame(df)
        if result.format == "wkb":
            for col in result.geometry_columns:
                df[col] = gpd.GeoSeries.from_wkb(df[f"{col}_wkb"], crs=result.srid)
            df = df.drop(columns=[f"{col}_wkb" for col in result.geometry_columns])
        elif result.format == "geojson":
            for col in result.geometry_columns:
                df[col] = df[col].apply(
                    lambda x: (shapely.from_geojson(x) if x else None)
                )
                df[col] = gpd.GeoSeries(df[col], crs=result.srid)
        return gpd.GeoDataFrame(df, geometry=result.geometry_columns[0])

    def prepare_array_columns(
        self, df: Any, db_ops: DatabaseOps
    ) -> tuple[Any, dict[str, ArrayKind]]:
        """Detect and format list columns for database insertion."""
        return detect_pandas_array_columns(df, db_ops)

    def to_arrow(self, df: Any) -> pa.Table:
        """Convert a GeoDataFrame to Arrow via Pandas."""
        return pa.Table.from_pandas(pd.DataFrame(df))

    def to_polars(self, df: Any) -> pl.DataFrame:
        """Convert a GeoDataFrame to Polars via Pandas."""
        return pl.from_pandas(pd.DataFrame(df))

    def to_pandas(self, df: Any) -> pd.DataFrame:
        """Convert a GeoDataFrame to a plain Pandas DataFrame."""
        return pd.DataFrame(df)

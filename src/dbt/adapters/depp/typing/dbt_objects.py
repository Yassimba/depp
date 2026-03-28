"""Type hints for dbt Python models to improve developer experience."""

from typing import Any, Literal, Protocol, overload

import geopandas as gpd
import pandas as pd
import polars as pl

from .base import DbtConfig, DbtThis, IndexConfig, PrimaryKeyConstraint

PolarsDataFrame = pl.DataFrame
DataFrame = pd.DataFrame | PolarsDataFrame | gpd.GeoDataFrame


class DbtObject[DataFrameT_co: DataFrame](Protocol):
    """Provide methods to reference other models and sources."""

    @overload
    def ref(
        self,
        model_name: str,
        *additional_names: str,
        version: str | int | None = None,
        v: str | int | None = None,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: Literal[False] = ...,
    ) -> DataFrameT_co: ...

    @overload
    def ref(
        self,
        model_name: str,
        *additional_names: str,
        version: str | int | None = None,
        v: str | int | None = None,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: Literal[True] = ...,
    ) -> pl.LazyFrame: ...

    def ref(
        self,
        model_name: str,
        *additional_names: str,
        version: str | int | None = None,
        v: str | int | None = None,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: bool = False,
    ) -> DataFrameT_co | pl.LazyFrame:
        """Return the referenced model as a DataFrame.

        Args:
            model_name: Name of the model to reference
            *additional_names: Additional parts of the model name (for two-part names)
            version: Model version (alternative to 'v')
            v: Model version (short form)
            partition_on: Column name (numeric/date) to partition parallel reads on
            partition_num: Number of parallel partitions (defaults to cpu count)
            lowercase: Lowercase all column names (default True)
            lazy: Return a Polars LazyFrame instead of DataFrame (default False)

        Returns:
            A DataFrame, or LazyFrame when lazy=True.
        """
        ...

    @overload
    def source(
        self,
        source_name: str,
        table_name: str,
        *,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: Literal[False] = ...,
    ) -> DataFrameT_co: ...

    @overload
    def source(
        self,
        source_name: str,
        table_name: str,
        *,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: Literal[True] = ...,
    ) -> pl.LazyFrame: ...

    def source(
        self,
        source_name: str,
        table_name: str,
        *,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: bool = False,
    ) -> DataFrameT_co | pl.LazyFrame:
        """Return the source table as a DataFrame.

        Args:
            source_name: Name of the source
            table_name: Name of the table within the source
            partition_on: Column name (numeric/date) to partition parallel reads on
            partition_num: Number of parallel partitions (defaults to cpu count)
            lowercase: Lowercase all column names (default True)
            lazy: Return a Polars LazyFrame instead of DataFrame (default False)

        Returns:
            A DataFrame, or LazyFrame when lazy=True.
        """
        ...

    def config(
        self,
        library: Literal["polars", "pandas", "geopandas"] | None = None,
        *,
        indexes: list[IndexConfig] | None = None,
        constraints: list[PrimaryKeyConstraint] | None = None,
        **kwargs: Any,
    ) -> DbtConfig:
        """Configure the model.

        Args:
            library: DataFrame library to use
            indexes: List of index configurations
            constraints: List of constraint configurations (e.g., primary keys)
            **kwargs: Additional dbt configuration options

        Examples:
            >>> # With indexes
            >>> dbt.config(
            ...     indexes=[
            ...         {"columns": ["id"], "unique": True},
            ...         {"columns": ["geom"], "type": "gist"},
            ...     ]
            ... )
            >>>
            >>> # With primary key
            >>> dbt.config(
            ...     constraints=[
            ...         {"type": "primary_key", "columns": ["id"]}
            ...     ]
            ... )
        """
        ...

    @property
    def this(self) -> DbtThis:
        """Reference to the current model."""
        ...

    def is_incremental(self) -> bool:
        """Check if this is an incremental model run."""
        ...

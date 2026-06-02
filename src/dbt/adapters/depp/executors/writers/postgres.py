from typing import Any, cast

import geopandas as gpd
from dbt.adapters.postgres.connections import PostgresCredentials
from geoalchemy2 import Geometry
from sqlalchemy import create_engine

from dbt.adapters.depp.db.base import DEFAULT_SRID
from dbt.adapters.depp.db.postgres import PostgresOps
from dbt.adapters.depp.executors.protocols import (
    Converter,
    DbContext,
    SourceInfo,
    geometry_columns,
)
from dbt.adapters.depp.executors.result import ExecutionResult


class PostgresWriter:
    """Write DataFrames to PostgreSQL via ADBC."""

    __slots__ = ("converter",)

    def __init__(self, converter: Converter[Any]) -> None:
        self.converter = converter

    def write(
        self, ctx: DbContext, df: Any, source: SourceInfo
    ) -> ExecutionResult:
        df, array_cols = self.converter.prepare_array_columns(df, ctx.ops)
        polars_df = self.converter.to_polars(df)
        conn = ctx.ops.connection_string(ctx.creds)
        ctx.ops.drop_table(conn, source.schema, source.table)
        polars_df.write_database(
            table_name=f"{source.schema}.{source.table}",
            connection=conn,
            if_table_exists="replace",
            engine="adbc",
        )
        if array_cols:
            ops = cast(PostgresOps, ctx.ops)
            creds = cast(PostgresCredentials, ctx.creds)
            engine = create_engine(ops.sqlalchemy_url(creds))
            with engine.begin() as db_conn:
                for col, kind in array_cols.items():
                    sql = ops.post_write_sql(
                        source.schema, source.table, col, ops.array_type(kind)
                    )
                    db_conn.exec_driver_sql(sql)
        return ExecutionResult(
            rows_affected=len(polars_df),
            schema=source.schema,
            table=source.table,
        )


class PostgresGeoWriter:
    """Write GeoDataFrames to PostgreSQL via PostGIS."""

    __slots__ = ()

    def write(
        self, ctx: DbContext, df: Any, source: SourceInfo
    ) -> ExecutionResult:
        gdf: gpd.GeoDataFrame = df
        srid = DEFAULT_SRID
        if hasattr(gdf, "geometry") and gdf.geometry.crs:
            srid = gdf.geometry.crs.to_epsg() or DEFAULT_SRID
        geom_dtype: dict[str, Any] = {
            col: Geometry("GEOMETRY", srid=srid)
            for col in geometry_columns(gdf)
        }
        engine = create_engine(ctx.ops.sqlalchemy_url(ctx.creds))
        gdf.to_postgis(
            name=source.table,
            con=engine,
            schema=source.schema,
            if_exists="replace",
            index=False,
            dtype=geom_dtype,
        )
        return ExecutionResult(
            rows_affected=len(gdf),
            schema=source.schema,
            table=source.table,
        )

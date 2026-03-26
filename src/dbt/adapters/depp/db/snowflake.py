import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import snowflake.connector
from dbt.adapters.snowflake.connections import SnowflakeCredentials

from .base import DEFAULT_SRID


class SnowflakeOps:
    """Snowflake database operations."""

    def connection_string(self, creds: SnowflakeCredentials) -> str:
        """Build a Snowflake connection string."""
        return (
            f"snowflake://{creds.user}:{creds.password}@{creds.account}/"
            f"{creds.database}/{creds.schema}?warehouse={creds.warehouse}"
        )

    def sqlalchemy_url(self, creds: SnowflakeCredentials) -> str:
        """Build a SQLAlchemy URL with optional role."""
        base = self.connection_string(creds)
        return f"{base}&role={creds.role}" if creds.role else base

    @contextmanager
    def get_connection(
        self, creds: SnowflakeCredentials, schema: str | None = None
    ) -> Iterator["snowflake.connector.SnowflakeConnection"]:
        """Get Snowflake connection as context manager."""
        conn = snowflake.connector.connect(
            account=creds.account,
            database=creds.database,
            schema=schema or creds.schema,
            warehouse=creds.warehouse,
            role=creds.role,
            **creds.auth_args(),
        )
        try:
            yield conn
        finally:
            conn.close()

    def drop_table(self, conn: Any, schema: str, table: str) -> None:
        """Drop a table with CASCADE (Snowflake handles via overwrite=True)."""

    def format_identifier(self, name: str) -> str:
        """Format identifier as uppercase for Snowflake."""
        return name.upper()

    def build_select_query(self, schema: str, table: str) -> str:
        """Build SELECT * query with uppercased identifiers."""
        return f"SELECT * FROM {schema.upper()}.{table.upper()}"

    def _query_column_names(
        self, conn: "snowflake.connector.SnowflakeConnection", query: str
    ) -> list[str]:
        """Execute query and return lowercase column names."""
        cursor = conn.cursor().execute(query)
        if cursor is None:
            return []
        result = cursor.fetch_pandas_all()
        return result["COLUMN_NAME"].str.lower().tolist() if len(result) > 0 else []

    def get_geometry_columns(
        self, conn: "snowflake.connector.SnowflakeConnection", schema: str, table: str
    ) -> dict[str, int]:
        """Get GEOGRAPHY columns from Snowflake information_schema."""
        names = self._query_column_names(conn, f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema.upper()}'
              AND table_name = '{table.upper()}'
              AND data_type = 'GEOGRAPHY'
        """)
        return dict.fromkeys(names, DEFAULT_SRID)

    def get_all_columns(
        self, conn: "snowflake.connector.SnowflakeConnection", schema: str, table: str
    ) -> list[str]:
        """Get all column names from Snowflake."""
        return self._query_column_names(conn, f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema.upper()}'
              AND table_name = '{table.upper()}'
            ORDER BY ordinal_position
        """)

    def format_array(self, values: list[Any]) -> str:
        """Format array values as JSON for Snowflake."""
        return json.dumps(values)

    def geometry_to_db(self, geom: Any) -> str:
        """Convert geometry to GeoJSON string for Snowflake."""
        return json.dumps(geom.__geo_interface__)

    def write_from_arrow(
        self, creds: SnowflakeCredentials, arrow: pa.Table, table: str, schema: str
    ) -> int:
        """Write Arrow table via Parquet PUT + COPY INTO."""
        arrow = arrow.rename_columns([c.upper() for c in arrow.column_names])
        s = self.format_identifier(schema)
        t = self.format_identifier(table)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            pq.write_table(arrow, path, use_dictionary=False)
            with self.get_connection(creds, schema) as conn:
                cur = conn.cursor()
                stage = f"{s}.depp_tmp_stage"
                fmt = f"{s}.depp_tmp_parquet_format"
                cur.execute(
                    f"CREATE OR REPLACE TEMPORARY FILE FORMAT {fmt} TYPE=PARQUET"
                )
                cur.execute(
                    f"CREATE OR REPLACE TEMPORARY STAGE {stage}"
                    f" FILE_FORMAT={fmt}"
                )
                cur.execute(f"PUT 'file://{path}' @{stage} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
                cur.execute(
                    f"CREATE OR REPLACE TABLE {s}.{t}"
                    f" USING TEMPLATE (SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))"
                    f" FROM TABLE(INFER_SCHEMA(LOCATION => '@{stage}',"
                    f" FILE_FORMAT => '{fmt}')))"
                )
                cur.execute(
                    f"COPY INTO {s}.{t} FROM @{stage}"
                    " MATCH_BY_COLUMN_NAME=CASE_INSENSITIVE PURGE=TRUE"
                )
        return arrow.num_rows

    def read_arrow(self, creds: SnowflakeCredentials, schema: str, query: str) -> Any:
        """Execute query and return Arrow table."""
        with self.get_connection(creds, schema) as conn:
            cursor = conn.cursor().execute(query)
            return (
                cursor.fetch_arrow_all(force_return_table=True)
                if cursor
                else pa.table({})
            )

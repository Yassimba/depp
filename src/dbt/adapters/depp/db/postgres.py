import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import adbc_driver_postgresql.dbapi as pg_dbapi
import connectorx as cx
import pandas as pd
from dbt.adapters.postgres.connections import PostgresCredentials

from .base import DEFAULT_SRID, ArrayKind


class PostgresOps:
    """PostgreSQL database operations."""

    def connection_string(self, creds: PostgresCredentials) -> str:
        """Build a PostgreSQL connection string."""
        return (
            f"postgresql://{creds.user}:{creds.password}"
            f"@{creds.host}:{creds.port}/{creds.database}"
        )

    def sqlalchemy_url(self, creds: PostgresCredentials) -> str:
        """Build a SQLAlchemy URL for PostgreSQL."""
        return self.connection_string(creds)

    @contextmanager
    def get_connection(self, creds: PostgresCredentials) -> Iterator[str]:
        """Yield connection string (connectorx handles connection pooling)."""
        yield self.connection_string(creds)

    def drop_table(self, conn: str, schema: str, table: str) -> None:
        """Drop a table with CASCADE to remove dependent objects."""
        with pg_dbapi.connect(conn) as db_conn, db_conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE')
            db_conn.commit()

    def format_identifier(self, name: str) -> str:
        """Format identifier with double quotes for PostgreSQL."""
        return f'"{name}"'

    def build_select_query(self, schema: str, table: str) -> str:
        """Build SELECT * query with quoted identifiers."""
        return f'SELECT * FROM "{schema}"."{table}"'

    def get_geometry_columns(
        self, conn: str, schema: str, table: str
    ) -> dict[str, int]:
        """Get geometry columns with SRID from PostGIS catalog."""
        query = f"""
            SELECT f_geometry_column AS col_name, srid
            FROM geometry_columns
            WHERE f_table_schema = '{schema}' AND f_table_name = '{table}'
        """
        df: pd.DataFrame = cx.read_sql(conn, query)
        result: dict[str, int] = {}
        for _, row in df.iterrows():
            srid = DEFAULT_SRID
            with contextlib.suppress(ValueError, TypeError, KeyError):
                srid = int(row["srid"])
            result[row["col_name"]] = srid if srid != 0 else DEFAULT_SRID
        return result

    def get_all_columns(self, conn: str, schema: str, table: str) -> list[str]:
        """Get all column names from PostgreSQL."""
        query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table}'
            ORDER BY ordinal_position
        """
        result = cx.read_sql(conn, query)
        return result["column_name"].tolist()

    def format_array(self, values: list[Any]) -> str:
        """Format array values for PostgreSQL."""
        return "{" + ",".join(str(x) for x in values) + "}"

    def array_type(self, kind: ArrayKind) -> str:
        """Return PostgreSQL array type for an ArrayKind."""
        return {
            ArrayKind.INT: "INTEGER[]",
            ArrayKind.BIGINT: "BIGINT[]",
            ArrayKind.TEXT: "TEXT[]",
        }[kind]

    def post_write_sql(self, schema: str, table: str, col: str, dtype: str) -> str:
        """Return ALTER TABLE SQL to cast a column type."""
        return (
            f"ALTER TABLE {schema}.{table} "
            f"ALTER COLUMN {col} TYPE {dtype} USING {col}::{dtype}"
        )

    def geometry_to_db(self, geom: Any) -> str:
        """Convert geometry to WKT string for PostgreSQL."""
        return str(geom.wkt)

from enum import StrEnum
from typing import Any, Protocol

from dbt.adapters.contracts.connection import Credentials

DEFAULT_SRID = 28992  # Dutch RD (Rijksdriehoeksmeting) coordinate system


class ArrayKind(StrEnum):
    """SQL array element kind inferred from a DataFrame list column."""

    INT = "int"
    BIGINT = "bigint"
    TEXT = "text"


class DatabaseOps(Protocol):
    """Protocol defining database-specific operations."""

    def connection_string(self, creds: Credentials) -> str: ...
    def sqlalchemy_url(self, creds: Credentials) -> str: ...
    def drop_table(self, conn: Any, schema: str, table: str) -> None: ...
    def format_identifier(self, name: str) -> str: ...
    def build_select_query(self, schema: str, table: str) -> str: ...
    def get_geometry_columns(
        self, conn: Any, schema: str, table: str
    ) -> dict[str, int]: ...
    def get_all_columns(self, conn: Any, schema: str, table: str) -> list[str]: ...
    def format_array(self, values: list[Any]) -> str: ...

    def geometry_to_db(self, geom: Any) -> str: ...

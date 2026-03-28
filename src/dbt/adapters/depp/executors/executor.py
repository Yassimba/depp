import time
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd
import polars as pl
from dbt.adapters.contracts.connection import Credentials

from dbt.adapters.depp.db import get_db_ops

from .protocols import Converter, DbContext, Reader, ReadOptions, SourceInfo, Writer
from .result import ExecutionResult


@dataclass(frozen=True)
class ExecutorConfig:
    """Bundle a reader, writer, and converter for a library+db combo."""

    reader: Reader
    writer: Writer
    converter: Converter[Any]


class Executor:
    """Compose reader, writer, and converter to execute Python models."""

    registry: ClassVar[dict[tuple[str, str], ExecutorConfig]] = {}
    type_mapping: ClassVar[dict[str, str]] = {}

    def __init__(self, config: ExecutorConfig, ctx: DbContext) -> None:
        self.reader = config.reader
        self.writer = config.writer
        self.converter = config.converter
        self.ctx = ctx
        self.read_time = 0.0
        self.write_time = 0.0

    @classmethod
    def register(
        cls,
        library: str,
        db_type: str,
        handled_types: list[str],
        config: ExecutorConfig,
    ) -> None:
        """Register an executor config for a library+db pair."""
        cls.registry[(library, db_type)] = config
        for t in handled_types:
            cls.type_mapping[t] = library

    @classmethod
    def create(cls, library: str, db_type: str, creds: Credentials) -> "Executor":
        """Create an executor from the registry."""
        key = (library, db_type)
        if key not in cls.registry:
            raise ValueError(
                f"No executor for {library}+{db_type}. Available: {list(cls.registry)}"
            )
        db_ops = get_db_ops(creds)
        ctx = DbContext(db_ops, creds)
        return cls(cls.registry[key], ctx)

    def read_df(
        self,
        table_name: str,
        *,
        partition_on: str | None = None,
        partition_num: int | None = None,
        lowercase: bool = True,
        lazy: bool = False,
    ) -> Any:
        """Read a table into a DataFrame via reader and converter."""
        source = SourceInfo.parse(table_name)
        options = ReadOptions(partition_on, partition_num) if partition_on else None
        start = time.perf_counter()
        arrow = self.reader.read_arrow(self.ctx, source, options)
        self.read_time += time.perf_counter() - start
        df = self.converter.from_arrow(arrow)
        if lowercase and isinstance(df, pl.DataFrame):
            df = df.rename(str.lower)
        elif lowercase and isinstance(df, pd.DataFrame):
            df = df.rename(columns=str.lower)
        return df.lazy() if lazy and isinstance(df, pl.DataFrame) else df

    def write_df(self, table_name: str, df: Any) -> ExecutionResult:
        """Write a DataFrame to a table via writer."""
        source = SourceInfo.parse(table_name)
        start = time.perf_counter()
        result = self.writer.write(self.ctx, df, source)
        self.write_time += time.perf_counter() - start
        return result

    def submit(self, compiled_code: str) -> ExecutionResult:
        """Execute compiled dbt Python model code."""
        self.read_time = 0.0
        self.write_time = 0.0
        local_vars: dict[str, Any] = {}
        exec(compiled_code, local_vars)
        if "main" not in local_vars:
            raise RuntimeError("No main function found in compiled code")
        start = time.perf_counter()
        result = local_vars["main"](self.read_df, self.write_df)
        total_time = time.perf_counter() - start
        exec_result = ExecutionResult(**result) if isinstance(result, dict) else result
        exec_result.read_time = self.read_time
        exec_result.write_time = self.write_time
        exec_result.transform_time = total_time - self.read_time - self.write_time
        return exec_result

    @classmethod
    def get_library_for_type(cls, type_hint: str) -> str | None:
        """Get the library name for a given type hint."""
        return cls.type_mapping.get(type_hint)

import builtins
import contextlib
import inspect
import time
from collections.abc import Callable, Iterable
from functools import cached_property, partial, wraps
from multiprocessing.context import SpawnContext
from pathlib import Path
from typing import Any, Concatenate, cast

from dbt.adapters.base.impl import BaseAdapter
from dbt.adapters.base.meta import AdapterMeta, available
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.contracts.connection import AdapterResponse, Credentials
from dbt.adapters.contracts.relation import RelationConfig
from dbt.adapters.events.types import CodeExecution, CodeExecutionStatus
from dbt.adapters.factory import FACTORY, get_adapter_by_type
from dbt.adapters.protocol import AdapterConfig, RelationProtocol
from dbt.artifacts.resources.types import ModelLanguage
from dbt.clients.jinja import MacroGenerator
from dbt.compilation import Compiler
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.parser.manifest import ManifestLoader
from dbt_common.events.functions import fire_event

from .config import (
    ADAPTER_NAME,
    DbInfo,
    DeppCredentials,
    DeppCredentialsWrapper,
    ModelConfig,
)
from .executors import Executor
from .utils.ast_utils import extract_python_docstring
from .utils.general import find_funcs_in_stack, release_plugin_lock

type FuncT[T, **P] = Callable[Concatenate[T, P], AdapterResponse]


def logs[T: "DeppAdapter", **P](func: FuncT[T, P]) -> FuncT[T, P]:
    """Log execution timing for python executor methods."""

    @wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> AdapterResponse:
        connection_name = self.connections.get_thread_connection().name
        compiled_code = args[1]
        fire_event(CodeExecution(conn_name=connection_name, code_content=compiled_code))

        start_time = time.time()
        response = func(self, *args, **kwargs)
        elapsed = round((time.time() - start_time), 2)

        fire_event(CodeExecutionStatus(status=response.__str__(), elapsed=elapsed))
        return response

    return wrapper


class AdapterTypeDescriptor:
    """Descriptor that returns the adapter type string."""

    type_str: str = ADAPTER_NAME

    def __get__(
        self, obj: "DeppAdapter | None", objtype: type["DeppAdapter"] | None = None
    ) -> Callable[[], str] | partial[str]:
        def _type(instance: "DeppAdapter | None" = None) -> str:
            if instance is None:
                return ADAPTER_NAME
            if find_funcs_in_stack({"render", "db_materialization"}):
                return instance.db_adapter.type()
            return ADAPTER_NAME

        return partial(_type, obj) if obj else _type


class RelationDescriptor:
    """Descriptor that lazily loads and caches the Relation class."""

    def __init__(self) -> None:
        self._relation: RelationProtocol | None = None

    def __get__(
        self, instance: "DeppAdapter | None", owner: type["DeppAdapter"] | None
    ) -> RelationProtocol | None:
        if self._relation is None:
            self._relation = DbInfo.get_cached_with_relation().relation
        return self._relation


class DeppAdapter(metaclass=AdapterMeta):
    """DBT adapter for executing Python models with database backends."""

    Relation = RelationDescriptor()
    AdapterSpecificConfigs = AdapterConfig
    type = AdapterTypeDescriptor()
    _db_adapter_class: builtins.type[BaseAdapter]
    _db_adapter: BaseAdapter
    db_creds: Credentials

    def __new__(cls, config: RuntimeConfig, mp_context: SpawnContext) -> "DeppAdapter":
        """Create adapter instance and configure underlying database adapter."""
        instance = super().__new__(cls)
        db_creds = cls.get_db_credentials(config)

        db_info = DbInfo.get_cached_with_relation()
        db_info.apply_overrides(config)

        with release_plugin_lock():
            db_adapter: type[BaseAdapter] = FACTORY.get_adapter_class_by_name(
                db_creds.type
            )
            original_plugin = FACTORY.get_plugin_by_name(config.credentials.type)
            original_plugin.dependencies = [db_creds.type]
            config.credentials = db_creds
            FACTORY.register_adapter(config, mp_context)
            config.credentials = cast(Credentials, DeppCredentialsWrapper(db_creds))

        instance._db_adapter_class = db_adapter
        instance.db_creds = db_creds
        return instance

    def __init__(self, config: RuntimeConfig, mp_context: SpawnContext) -> None:
        """Initialize adapter with database connection and configuration."""
        self.config = config
        self.mp_context = mp_context

        self._db_adapter = get_adapter_by_type(self._db_adapter_class.type())
        self.connections = self._db_adapter.connections
        self._available_ = self._db_adapter._available_.union(self._available_)
        self._parse_replacements_.update(self._db_adapter._parse_replacements_)

    @logs
    def submit_python_job(
        self, parsed_model: dict[str, Any], compiled_code: str
    ) -> AdapterResponse:
        # TODO: Add remote executors
        """Execute Python model code selecting the requested executor."""
        config = ModelConfig.from_model(parsed_model, compiled_code)
        executor = self.get_executor(config.library)
        result = executor.submit(compiled_code)
        return AdapterResponse(_message=f"PYTHON | {result}")

    def get_executor(self, library: str) -> Executor:
        """Get Python executor based on model's configured library and database type."""
        return Executor.create(library, self.db_creds.type, self.db_creds)

    @available
    def db_type(self) -> str:
        """Return the underlying database adapter type."""
        return self._db_adapter.type()

    @available
    def db_materialization(self, context: dict[str, Any], materialization: str) -> Any:
        """Execute database materialization macro."""
        macro = self.manifest.find_materialization_macro_by_name(
            self.config.project_name, materialization, self._db_adapter.type()
        )
        if macro is None:
            raise ValueError("Invalid Macro")
        return MacroGenerator(macro, context, stack=context["context_macro_stack"])()

    @classmethod
    def get_db_credentials(cls, config: RuntimeConfig) -> Credentials:
        """Extract database credentials from adapter configuration."""
        dep_credentials = cast(
            DeppCredentials | DeppCredentialsWrapper, config.credentials
        )
        if isinstance(dep_credentials, DeppCredentials):
            db_info = DbInfo.get_cached_with_relation()
            return db_info.profile.credentials
        with release_plugin_lock():
            FACTORY.load_plugin(dep_credentials.db_creds.type)
        return dep_credentials.db_creds

    def get_compiler(self) -> Compiler:
        """Get DBT compiler instance for this adapter."""
        return Compiler(self.config)

    def __getattr__(self, name: str) -> Any:
        """Directly proxy to the DB adapter."""
        if hasattr(self._db_adapter, name):
            return getattr(self._db_adapter, name)
        return getattr(super(), name)

    @classmethod
    def is_cancelable(cls) -> bool:
        """Python jobs cannot be cancelled once started."""
        return False

    @property
    def db_adapter(self) -> BaseAdapter:
        """Access underlying database adapter."""
        return self._db_adapter

    @cached_property
    def manifest(self) -> Manifest:
        """Get cached DBT manifest for the project."""
        return ManifestLoader.get_full_manifest(self.config)

    def get_filtered_catalog(
        self,
        relation_configs: Iterable[RelationConfig],
        used_schemas: frozenset[tuple[str, str]],
        relations: set[BaseRelation] | None = None,
    ) -> tuple[Any, list[Exception]]:
        """Override to enrich Python models with docstrings."""
        for manifest in [self.manifest, self._find_parent_manifest()]:
            if manifest:
                self.inject_docstring(manifest)

        return self._db_adapter.get_filtered_catalog(
            relation_configs, used_schemas, relations
        )

    def _find_parent_manifest(self) -> Manifest | None:
        """Find manifest from parent GenerateTask if it exists."""
        frame = inspect.currentframe()
        if frame is None:
            return None
        with contextlib.suppress(AttributeError):
            while frame := frame.f_back:
                if (
                    (obj := frame.f_locals.get("self"))
                    and type(obj).__name__ == "GenerateTask"
                    and getattr(obj, "manifest", None)
                ):
                    return obj.manifest
        return None

    def inject_docstring(self, manifest: Manifest) -> None:
        """Extract Python model docstrings as descriptions (YAML takes precedence)."""
        project_root = Path(self.config.project_root)
        for node in manifest.nodes.values():
            if (
                getattr(node, "language", None) == ModelLanguage.python
                and node.resource_type.value == "model"
                and not (node.description or "").strip()
                and (
                    docstring := extract_python_docstring(
                        str(project_root / node.original_file_path)
                    )
                )
            ):
                node.description = docstring

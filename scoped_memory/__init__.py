"""Tree-RAG (`scoped_memory`): Hierarchical Scoped Memory Engine for Autonomous Agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scoped_memory.atomic import (
    _atomic_write_faiss,
    _atomic_write_json,
    atomic_write_faiss,
    atomic_write_json,
)
from scoped_memory.embed import (
    CpuHttpEmbedder,
    Embedder,
    MockEmbedder,
    chunk_text,
)
from scoped_memory.guard import (
    ConcurrencyGuard,
    PeerNode,
    SyncPolicy,
    heal_workspace_conflicts,
    merge_doc_collections,
    parse_pending_markdown,
    pending_count,
    pending_health,
    write_pending_markdown,
)
from scoped_memory.hierarchy import (
    EmptyContext,
    HierarchyEngine,
    SearchHit,
    SearchResult,
    requires_confirmation,
)
from scoped_memory.index import (
    ALLOWED_CATEGORIES,
    CATEGORY_TO_LAYER,
    CATEGORY_TO_PILLAR,
    WorkspaceIndex,
)
from scoped_memory.locks import (
    release_lock,
    try_global_lock,
    try_workspace_lock,
    workspace_lock,
)
from scoped_memory.sanitize import (
    _scrub_secrets,
    normalize_text,
    sanitize,
    scrub_secrets,
    text_hash,
)
from scoped_memory.store import ScopedMemoryStore
from scoped_memory.swap import (
    EmbeddingModelConfig,
    SwapPlan,
    SwapReport,
    execute_swap,
    plan_swap,
)
from scoped_memory.sync import (
    ConflictPolicy,
    DeltaTransport,
    DimensionMismatchError,
    Doc,
    FirstWriteWins,
    HttpTransport,
    LastWriteWins,
    LocalFsTransport,
    PushResult,
    RAGSyncEngine,
    SyncReport,
    UnionMerge,
)

if TYPE_CHECKING:
    from scoped_memory.mcp_server import (
        MemoryMcpServer,
        build_fastmcp_app,
        enrich_document,
        generate_mermaid_canvas,
    )

__version__ = "0.3.0"


def __getattr__(name: str) -> Any:
    if name in {
        "MemoryMcpServer",
        "build_fastmcp_app",
        "enrich_document",
        "generate_mermaid_canvas",
    }:
        from scoped_memory import mcp_server

        return getattr(mcp_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ALLOWED_CATEGORIES",
    "CATEGORY_TO_LAYER",
    "CATEGORY_TO_PILLAR",
    "ConcurrencyGuard",
    "ConflictPolicy",
    "CpuHttpEmbedder",
    "DeltaTransport",
    "DimensionMismatchError",
    "Doc",
    "Embedder",
    "EmbeddingModelConfig",
    "EmptyContext",
    "FirstWriteWins",
    "HierarchyEngine",
    "HttpTransport",
    "LastWriteWins",
    "LocalFsTransport",
    "MemoryMcpServer",
    "MockEmbedder",
    "PeerNode",
    "PushResult",
    "RAGSyncEngine",
    "ScopedMemoryStore",
    "SearchHit",
    "SearchResult",
    "SwapPlan",
    "SwapReport",
    "SyncPolicy",
    "SyncReport",
    "UnionMerge",
    "WorkspaceIndex",
    "__version__",
    "_atomic_write_faiss",
    "_atomic_write_json",
    "_scrub_secrets",
    "atomic_write_faiss",
    "atomic_write_json",
    "build_fastmcp_app",
    "chunk_text",
    "enrich_document",
    "execute_swap",
    "generate_mermaid_canvas",
    "heal_workspace_conflicts",
    "merge_doc_collections",
    "normalize_text",
    "parse_pending_markdown",
    "pending_count",
    "pending_health",
    "plan_swap",
    "release_lock",
    "requires_confirmation",
    "sanitize",
    "scrub_secrets",
    "text_hash",
    "try_global_lock",
    "try_workspace_lock",
    "workspace_lock",
    "write_pending_markdown",
]

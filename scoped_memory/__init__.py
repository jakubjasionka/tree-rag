"""Tree-RAG (`scoped_memory`): Hierarchical Scoped Memory Engine for Autonomous Agents."""

from __future__ import annotations

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
from scoped_memory.index import ALLOWED_CATEGORIES, WorkspaceIndex
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

__version__ = "0.2.0"

__all__ = [
    "ALLOWED_CATEGORIES",
    "ConcurrencyGuard",
    "CpuHttpEmbedder",
    "Embedder",
    "EmptyContext",
    "HierarchyEngine",
    "MockEmbedder",
    "PeerNode",
    "ScopedMemoryStore",
    "SearchHit",
    "SearchResult",
    "SyncPolicy",
    "WorkspaceIndex",
    "__version__",
    "_atomic_write_faiss",
    "_atomic_write_json",
    "_scrub_secrets",
    "atomic_write_faiss",
    "atomic_write_json",
    "chunk_text",
    "heal_workspace_conflicts",
    "merge_doc_collections",
    "normalize_text",
    "parse_pending_markdown",
    "pending_count",
    "pending_health",
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

#!/usr/bin/env python3
"""Optional FastMCP server wrapper exposing `ScopedMemoryStore` over stdio/MCP.

Requires optional MCP dependency:
    pip install -e ".[mcp]"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scoped_memory import CpuHttpEmbedder, MockEmbedder, ScopedMemoryStore


def create_default_store() -> ScopedMemoryStore:
    base_dir = Path(os.environ.get("TREE_RAG_BASE_DIR", "./.treerag_data"))
    use_http = os.environ.get("TREE_RAG_USE_HTTP_EMBEDDER", "0") == "1"
    embedder = CpuHttpEmbedder() if use_http else MockEmbedder()
    require_hitl = os.environ.get("TREE_RAG_REQUIRE_HITL", "1") == "1"
    return ScopedMemoryStore(
        base_dir=base_dir,
        embedder=embedder,
        require_global_confirmation=require_hitl,
    )


store = create_default_store()


def build_mcp_app():
    """Construct the FastMCP server instance if `mcp` is installed."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("tree-rag-memory")

    @mcp.tool()
    def memory_save(
        category: str,
        text: str,
        source: str = "",
        workspace: str = "",
        tags: str = "",
    ) -> str:
        """Save an observation into the active or specified workspace branch."""
        res = store.save(
            category=category,
            text=text,
            source=source,
            workspace=workspace or None,
            tags=tags or None,
        )
        return str(res.get("message", json.dumps(res)))

    @mcp.tool()
    def memory_search(
        query: str,
        top_k: int = 5,
        category: str = "",
        scope: str = "workspace",
        workspace: str = "",
        confirmed: bool = False,
    ) -> str:
        """Search workspace memory (`scope="workspace"`) or escalate globally (`scope="global"`)."""
        return store.search_formatted(
            query=query,
            top_k=top_k,
            category=category,
            scope=scope,
            workspace=workspace or None,
            confirmed=confirmed if scope == "global" else None,
        )

    @mcp.tool()
    def memory_list(
        category: str = "",
        limit: int = 50,
        scope: str = "workspace",
        workspace: str = "",
    ) -> str:
        """List recent entries in the current workspace or across all workspaces."""
        return store.list_formatted(
            category=category,
            limit=limit,
            scope=scope,
            workspace=workspace or None,
        )

    @mcp.tool()
    def memory_health() -> str:
        """Inspect workspace indices, locks, and pending queue status."""
        return json.dumps(store.health(), indent=2)

    return mcp


if __name__ == "__main__":
    app = build_mcp_app()
    app.run()

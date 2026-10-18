"""Production FastMCP stdio server built on the `ScopedMemoryStore` facade.

Exposes hierarchical Tree-RAG tools over Model Context Protocol (MCP):
- `memory_save`
- `memory_search`
- `memory_list`
- `memory_health`
- `memory_merge_pending`

All configuration is driven strictly via environment variables (no hostname sniffing):
- `TREE_RAG_BASE_DIR` (default: `./.treerag_data`)
- `TREE_RAG_EMBED_URL` / `MEMORY_EMBED_URL` (when set, uses `CpuHttpEmbedder`; else `MockEmbedder`)
- `TREE_RAG_EMBED_URL_PLUS` / `MEMORY_EMBED_PLUS_URL` (optional high-dim embedding endpoint)
- `MEMORY_WORKSPACE` / `TREE_RAG_DEFAULT_WORKSPACE` (default: `default`)
- `MEMORY_EMBED_TIMEOUT` (default: `15.0` seconds)
- `TREE_RAG_CONCURRENCY_PATTERNS` (comma-separated process patterns; default: disabled)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scoped_memory.embed import CpuHttpEmbedder, Embedder, MockEmbedder
from scoped_memory.hierarchy import EmptyContext, SearchResult
from scoped_memory.index import (
    ALLOWED_CATEGORIES,
    CATEGORY_TO_LAYER,
    CATEGORY_TO_PILLAR,
    WorkspaceIndex,
)
from scoped_memory.store import ScopedMemoryStore

LAYER_DESCRIPTIONS: dict[str, str] = {
    "L3": "L3 Persona / Decisions & Skills",
    "L2": "L2 Scenario / Code & Wiki",
    "L1": "L1 Atomic Facts / Bugs & Assumptions",
    "L0": "L0 Raw Context / Notes & Logs",
}

PILLAR_DESCRIPTIONS: dict[str, str] = {
    "chat": "Chat & Context",
    "skill": "Skills & Decisions",
    "wiki": "Knowledge & Wiki",
    "code": "Code & Bug Graph",
}

TOOL_SPECIFICATIONS: list[dict[str, Any]] = [
    {
        "name": "memory_save",
        "description": (
            "Save an observation into a workspace branch with secret scrubbing, "
            "SHA-256 deduplication, L0-L3 layer tagging, and atomic FAISS persistence."
        ),
        "parameters": [
            "category",
            "text",
            "source",
            "workspace",
            "tags",
        ],
    },
    {
        "name": "memory_search",
        "description": (
            "Semantic vector search across the active workspace ('workspace') or "
            "all workspaces ('global'), returning scored hits with L0-L3 layer and pillar metadata."
        ),
        "parameters": [
            "query",
            "top_k",
            "category",
            "scope",
            "workspace",
        ],
    },
    {
        "name": "memory_list",
        "description": (
            "List recent memory records from a workspace or globally with "
            "category, layer (L0-L3), and pillar annotations."
        ),
        "parameters": [
            "category",
            "limit",
            "scope",
            "workspace",
        ],
    },
    {
        "name": "memory_health",
        "description": (
            "Inspect per-workspace FAISS index sizes, pending queue counts, "
            "concurrency guard status, and embedding endpoint reachability."
        ),
        "parameters": [],
    },
    {
        "name": "memory_merge_pending",
        "description": (
            "Compact queued `_pending/<workspace>/*.md` markdown entries into "
            "their respective FAISS workspace indices."
        ),
        "parameters": [
            "workspace",
            "force",
        ],
    },
]


def enrich_document(doc: dict[str, Any], workspace: str) -> dict[str, Any]:
    """Attach canonical `workspace`, `layer` (L0-L3), `pillar`, and `tags` to a raw record."""
    cat = str(doc.get("category") or "note").strip().lower()
    if cat not in ALLOWED_CATEGORIES:
        cat = "note"
    layer = str(doc.get("layer") or CATEGORY_TO_LAYER.get(cat, "L0"))
    pillar = str(doc.get("pillar") or CATEGORY_TO_PILLAR.get(cat, "chat"))
    raw_tags = doc.get("tags")
    if isinstance(raw_tags, list):
        tags = [str(t) for t in raw_tags]
    else:
        tags = WorkspaceIndex.extract_tags(
            str(doc.get("text") or ""),
            str(doc.get("source") or ""),
            raw_tags if isinstance(raw_tags, str) else None,
        )
    return {
        "id": str(doc.get("id") or ""),
        "workspace": workspace,
        "category": cat,
        "layer": layer,
        "pillar": pillar,
        "source": str(doc.get("source") or "manual"),
        "text": str(doc.get("text") or ""),
        "time": str(doc.get("time") or ""),
        "updated_at": str(doc.get("updated_at") or ""),
        "tags": tags,
    }


def _clean_mermaid_label(text: str, max_len: int = 52) -> str:
    """Sanitize arbitrary record text into a safe single-line Mermaid node label."""
    cleaned = re.sub(r'["\[\]{}()<>|`#&;\\]', " ", (text or "").strip())
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "…"
    return cleaned or "entry"


def generate_mermaid_canvas(
    store: ScopedMemoryStore,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Generate a Mermaid flowchart representing the Symbolic Memory Canvas.

    - When `workspace` is empty or `"global"`, renders the top workspaces and category counts.
    - When `workspace` specifies a branch, renders the L3 -> L0 layer hierarchy and recent nodes.
    """
    ws_clean = (workspace or "").strip()
    if ws_clean and ws_clean.lower() != "global":
        ws_index = store.get_workspace_index(ws_clean)
        ws_name = ws_index.name
        docs = [enrich_document(d, ws_name) for d in ws_index.docs]
        lines = [
            "flowchart LR",
            f'    WS["Workspace: {ws_name} ({len(docs)} docs)"]',
        ]
        by_layer: dict[str, list[dict[str, Any]]] = {"L3": [], "L2": [], "L1": [], "L0": []}
        for d in docs:
            by_layer.setdefault(d["layer"], []).append(d)

        for lyr in ("L3", "L2", "L1", "L0"):
            items = by_layer.get(lyr, [])
            if not items:
                continue
            title = LAYER_DESCRIPTIONS.get(lyr, lyr)
            lid = f"LYR_{lyr}"
            lines.append(f'    {lid}["{title} ({len(items)})"]')
            lines.append(f"    WS --> {lid}")
            for idx, item in enumerate(items[-5:]):
                nid = f"N_{lyr}_{idx}"
                short_id = item["id"][-6:] if item["id"] else f"idx{idx}"
                lbl = _clean_mermaid_label(item["text"], 46)
                src = _clean_mermaid_label(item["source"], 22)
                lines.append(f'    {nid}["[{short_id}] {lbl}<br/>src: {src}"]')
                lines.append(f"    {lid} --> {nid}")
        return {
            "ok": True,
            "workspace": ws_name,
            "mermaid": "\n".join(lines),
            "docs_count": len(docs),
        }

    stores = store.all_workspace_indices()
    total_docs = sum(len(s.docs) for s in stores.values())
    lines = [
        "flowchart LR",
        f'    ROOT["Global Memory Hub ({len(stores)} workspaces / {total_docs} docs)"]',
    ]
    sorted_ws = sorted(stores.items(), key=lambda kv: len(kv[1].docs), reverse=True)[:12]
    for idx, (wn, st) in enumerate(sorted_ws):
        wid = f"WS_{idx}"
        cats: dict[str, int] = {}
        for d in st.docs:
            c = str(d.get("category") or "note").lower()
            cats[c] = cats.get(c, 0) + 1
        cat_summary = ", ".join(f"{k}:{v}" for k, v in sorted(cats.items())) or "empty"
        lines.append(f'    {wid}["{wn} ({len(st.docs)})<br/>{cat_summary}"]')
        lines.append(f"    ROOT --> {wid}")
    return {
        "ok": True,
        "workspace": "global",
        "mermaid": "\n".join(lines),
        "docs_count": total_docs,
    }


def _is_process_pattern_running(pattern: str) -> bool:
    """Check via `pgrep -a -f` whether a matching process is running (excluding self)."""
    cleaned = pattern.strip()
    if not cleaned:
        return False
    try:
        out = subprocess.check_output(
            ["pgrep", "-a", "-f", cleaned],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except Exception:
        return False

    pid_self = os.getpid()
    try:
        compiled = re.compile(cleaned)
    except re.error:
        compiled = None

    for line in out.strip().splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == pid_self:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        if "scoped_memory.mcp_server" in cmd or cmd.lstrip().startswith("pgrep "):
            continue
        if "bash -c" in cmd and "pgrep" in cmd:
            continue
        if compiled is not None and not compiled.search(cmd):
            continue
        return True
    return False


def build_env_concurrency_detector(
    patterns: Sequence[str] | None = None,
) -> Callable[[], tuple[bool, str]] | None:
    """Build an optional process-pattern concurrency detector from environment configuration.

    Disabled by default unless `patterns` are explicitly provided or
    `TREE_RAG_CONCURRENCY_PATTERNS` / `MEMORY_CONCURRENCY_PATTERNS` is set.
    """
    raw_patterns: list[str] = []
    if patterns is not None:
        raw_patterns = [p.strip() for p in patterns if p and p.strip()]
    else:
        env_val = (
            os.environ.get("TREE_RAG_CONCURRENCY_PATTERNS")
            or os.environ.get("MEMORY_CONCURRENCY_PATTERNS")
            or ""
        ).strip()
        if env_val:
            raw_patterns = [p.strip() for p in env_val.split(",") if p.strip()]

    if not raw_patterns:
        return None

    def _detect() -> tuple[bool, str]:
        matched: list[str] = []
        for pat in raw_patterns:
            if _is_process_pattern_running(pat):
                matched.append(pat)
        if matched:
            return True, f"concurrent writer active ({', '.join(matched)})"
        return False, ""

    return _detect


def probe_http_endpoint(url: str | None, timeout_sec: float = 0.8) -> dict[str, Any]:
    """Check TCP/HTTP reachability of an OpenAI-compatible `/v1` embedding endpoint."""
    raw = (url or "").strip()
    if not raw:
        return {"configured": False, "reachable": False, "endpoint": ""}

    import socket

    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    reachable = False
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            reachable = True
    except OSError:
        reachable = False

    return {
        "configured": True,
        "reachable": reachable,
        "endpoint": raw,
    }


def create_embedder_from_env() -> Embedder:
    """Instantiate `CpuHttpEmbedder` when `TREE_RAG_EMBED_URL` is set, else `MockEmbedder`."""
    embed_url = os.environ.get("TREE_RAG_EMBED_URL", "").strip()
    timeout_raw = (
        os.environ.get("MEMORY_EMBED_TIMEOUT")
        or os.environ.get("TREE_RAG_EMBED_TIMEOUT")
        or "15.0"
    )
    try:
        timeout_sec = float(timeout_raw)
    except ValueError:
        timeout_sec = 15.0

    if embed_url:
        model_name = (
            os.environ.get("TREE_RAG_EMBED_MODEL")
            or "intfloat/multilingual-e5-small"
        ).strip()
        return CpuHttpEmbedder(
            endpoint_url=embed_url,
            model_name=model_name,
            timeout_sec=timeout_sec,
        )
    return MockEmbedder()


class MemoryMcpServer:
    """High-level MCP server controller wrapping `ScopedMemoryStore` with layer metadata."""

    def __init__(
        self,
        store: ScopedMemoryStore | None = None,
        base_dir: str | Path | None = None,
        embedder: Embedder | None = None,
        default_workspace: str | None = None,
        concurrency_patterns: Sequence[str] | None = None,
    ) -> None:
        if store is not None:
            self.store = store
        else:
            resolved_embedder = embedder if embedder is not None else create_embedder_from_env()
            detector = build_env_concurrency_detector(concurrency_patterns)
            require_hitl = os.environ.get("TREE_RAG_REQUIRE_HITL", "0") == "1"
            self.store = ScopedMemoryStore(
                base_dir=base_dir,
                default_workspace=default_workspace,
                embedder=resolved_embedder,
                require_global_confirmation=require_hitl,
                concurrency_detector=detector,
            )

    def _log_telemetry(
        self,
        op: str,
        workspace: str,
        category: str = "",
        query: str = "",
        hits: int = 0,
        top_score: float = 0.0,
        latency_ms: float = 0.0,
        status: str = "ok",
    ) -> None:
        """Append a structured JSONL telemetry line to `<base_dir>/_telemetry.jsonl`."""
        try:
            telem_path = self.store.base_dir / "_telemetry.jsonl"
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "op": op,
                "workspace": workspace,
                "category": category,
                "query": (query or "")[:160],
                "hits": hits,
                "top_score": round(float(top_score), 4),
                "latency_ms": round(float(latency_ms), 2),
                "status": status,
                "client": os.environ.get("MEMORY_CLIENT_ID", "mcp-stdio"),
            }
            with telem_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def memory_save(
        self,
        category: str,
        text: str,
        source: str = "",
        workspace: str = "",
        tags: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        """Save an observation and expose `category`, `layer`, and `pillar` in the response."""
        t0 = time.perf_counter()
        res = self.store.save(
            category=category,
            text=text,
            source=source,
            workspace=workspace or None,
            tags=tags,
        )
        cat = str(res.get("category") or category or "note").lower()
        if cat not in ALLOWED_CATEGORIES:
            cat = "note"
        layer = CATEGORY_TO_LAYER.get(cat, "L0")
        pillar = CATEGORY_TO_PILLAR.get(cat, "chat")
        res["category"] = cat
        res["layer"] = layer
        res["pillar"] = pillar
        if isinstance(res.get("doc"), dict):
            res["doc"] = enrich_document(res["doc"], str(res.get("workspace") or ""))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._log_telemetry(
            op="memory_save",
            workspace=str(res.get("workspace") or self.store.default_workspace),
            category=cat,
            query=(text or "")[:120],
            hits=1 if res.get("ok") else 0,
            latency_ms=latency_ms,
            status=str(res.get("status") or "ok"),
        )
        return res

    def memory_search(
        self,
        query: str,
        top_k: int = 5,
        category: str = "",
        scope: str = "workspace",
        workspace: str = "",
    ) -> dict[str, Any]:
        """Search workspace or global memory and return hits enriched with `layer` and `pillar`."""
        t0 = time.perf_counter()
        resolved_scope = "global" if (scope or "").strip().lower() == "global" else "workspace"
        result = self.store.search(
            query=query,
            top_k=max(1, min(50, int(top_k))),
            category=category,
            scope=resolved_scope,
            workspace=workspace or None,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        hits_out: list[dict[str, Any]] = []
        hl_workspaces: list[str] = []
        hl_categories: list[str] = []
        hl_sources: list[str] = []
        hl_ids: list[str] = []

        if isinstance(result, SearchResult):
            for h in result.hits:
                ed = enrich_document(h.doc, h.workspace)
                ed["score"] = round(float(h.score), 4)
                ed["label"] = h.label
                hits_out.append(ed)
                if ed["workspace"] not in hl_workspaces:
                    hl_workspaces.append(ed["workspace"])
                cat_key = f"{ed['workspace']}::{ed['category']}"
                if cat_key not in hl_categories:
                    hl_categories.append(cat_key)
                src_key = f"{ed['workspace']}::{ed['category']}::{ed['source']}"
                if src_key not in hl_sources:
                    hl_sources.append(src_key)
                if ed["id"] and ed["id"] not in hl_ids:
                    hl_ids.append(ed["id"])

        top_score = hits_out[0]["score"] if hits_out else 0.0
        target_ws = (
            "global"
            if resolved_scope == "global"
            else self.store._resolve_workspace(workspace or None)
        )
        self._log_telemetry(
            op="memory_search",
            workspace=target_ws,
            category=category,
            query=query,
            hits=len(hits_out),
            top_score=top_score,
            latency_ms=latency_ms,
            status="empty" if isinstance(result, EmptyContext) else "ok",
        )

        return {
            "ok": True,
            "query": query,
            "scope": resolved_scope,
            "workspace": target_ws,
            "count": len(hits_out),
            "is_empty": bool(result.is_empty),
            "requires_confirmation": bool(result.requires_confirmation()),
            "reason": getattr(result, "reason", "ok"),
            "latency_ms": latency_ms,
            "results": hits_out,
            "highlights": {
                "workspaces": hl_workspaces,
                "categories": hl_categories,
                "sources": hl_sources,
                "ids": hl_ids,
            },
            "formatted": self._format_search_hits(
                query=query,
                scope=resolved_scope,
                workspace=target_ws,
                hits=hits_out,
                raw_result=result,
            ),
        }

    @staticmethod
    def _format_search_hits(
        query: str,
        scope: str,
        workspace: str,
        hits: list[dict[str, Any]],
        raw_result: SearchResult | EmptyContext,
    ) -> str:
        if isinstance(raw_result, EmptyContext) or not hits:
            return raw_result.format()
        header = (
            f'Found {len(hits)} hits (global, all workspaces) for: "{query}"'
            if scope == "global"
            else f'Found {len(hits)} hits in [{workspace}] for: "{query}"'
        )
        blocks = [header]
        for idx, item in enumerate(hits, start=1):
            ws_tag = f" [workspace={item['workspace']}]" if scope == "global" else ""
            blocks.append(
                f"### #{idx} (score={item['score']:.3f}){ws_tag} "
                f"[{item['category']} | layer={item['layer']} | pillar={item['pillar']}] "
                f"src={item['source']}\n{item['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def memory_list(
        self,
        category: str = "",
        limit: int = 50,
        scope: str = "workspace",
        workspace: str = "",
    ) -> dict[str, Any]:
        """List recent memory records enriched with `layer` (L0-L3) and `pillar`."""
        t0 = time.perf_counter()
        resolved_scope = "global" if (scope or "").strip().lower() == "global" else "workspace"
        raw_items = self.store.list(
            category=category,
            limit=max(1, min(250, int(limit))),
            scope=resolved_scope,
            workspace=workspace or None,
        )
        enriched = [
            enrich_document(d, str(d.get("workspace") or self.store.default_workspace))
            for d in raw_items
        ]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        target_ws = (
            "global"
            if resolved_scope == "global"
            else self.store._resolve_workspace(workspace or None)
        )
        self._log_telemetry(
            op="memory_list",
            workspace=target_ws,
            category=category,
            hits=len(enriched),
            latency_ms=latency_ms,
        )

        if not enriched:
            formatted = "memory_list: 0 entries."
        else:
            lines = [f"Last {len(enriched)} entries ({resolved_scope}):"]
            for item in enriched:
                preview = item["text"]
                if len(preview) > 160:
                    preview = preview[:160] + " [...]"
                ws_prefix = (
                    f"[workspace={item['workspace']}] " if resolved_scope == "global" else ""
                )
                lines.append(
                    f"- {ws_prefix}[{item['category']} | {item['layer']} | {item['pillar']}] "
                    f"{item['source']} ({item['time']}): {preview}"
                )
            formatted = "\n".join(lines)

        return {
            "ok": True,
            "scope": resolved_scope,
            "workspace": target_ws,
            "count": len(enriched),
            "records": enriched,
            "formatted": formatted,
        }

    def memory_health(self) -> dict[str, Any]:
        """Return workspace index byte sizes, pending queue, guard status, and embedder status."""
        base_health = self.store.health()
        workspaces_detail: dict[str, Any] = {}
        total_index_bytes = 0
        total_docs = 0

        for ws_name, ws_info in base_health.get("workspaces", {}).items():
            ws_dir = self.store.base_dir / ws_name
            faiss_path = ws_dir / "index.faiss"
            docs_path = ws_dir / "docs.json"
            idx_bytes = faiss_path.stat().st_size if faiss_path.exists() else 0
            docs_bytes = docs_path.stat().st_size if docs_path.exists() else 0
            total_index_bytes += idx_bytes
            total_docs += int(ws_info.get("docs", 0))
            workspaces_detail[ws_name] = {
                **ws_info,
                "index_bytes": idx_bytes,
                "docs_bytes": docs_bytes,
            }

        embed_url = (
            os.environ.get("TREE_RAG_EMBED_URL")
            or os.environ.get("MEMORY_EMBED_URL")
            or ""
        ).strip()
        plus_url = (
            os.environ.get("TREE_RAG_EMBED_URL_PLUS")
            or os.environ.get("MEMORY_EMBED_PLUS_URL")
            or ""
        ).strip()

        base_probe = probe_http_endpoint(embed_url)
        plus_probe = probe_http_endpoint(plus_url)
        embedder_type = type(self.store.embedder).__name__

        return {
            "ok": True,
            **base_health,
            "workspaces": workspaces_detail,
            "total_docs": total_docs,
            "total_index_bytes": total_index_bytes,
            "embedder_type": embedder_type,
            "embedder_dim": self.store.embedder.dim,
            "concurrency_detection_enabled": self.store.guard.detector is not None,
            "embed_endpoints": {
                "base": base_probe,
                "plus": plus_probe,
            },
            "layer_mapping": dict(CATEGORY_TO_LAYER),
            "pillar_mapping": dict(CATEGORY_TO_PILLAR),
        }

    def memory_merge_pending(
        self,
        workspace: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        """Merge queued `_pending/<workspace>/*.md` files into FAISS workspace indices."""
        t0 = time.perf_counter()
        res = self.store.merge_pending(
            workspace=workspace or None,
            force=force,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self._log_telemetry(
            op="memory_merge_pending",
            workspace=workspace or "all",
            hits=int(res.get("merged", 0)),
            latency_ms=latency_ms,
        )
        return {"ok": True, "latency_ms": latency_ms, **res}

    def generate_mermaid(self, workspace: str = "global") -> dict[str, Any]:
        """Generate a Mermaid symbolic canvas for a workspace or the global hub."""
        return generate_mermaid_canvas(self.store, workspace=workspace)


def build_fastmcp_app(server: MemoryMcpServer | None = None) -> Any:
    """Construct a `FastMCP` application wrapping `MemoryMcpServer`.

    The `mcp` package is imported lazily so `scoped_memory.mcp_server` remains importable
    when installed without the `.[mcp]` optional extra.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "FastMCP is not installed. Install optional MCP dependencies via: "
            "pip install -e '.[mcp]'"
        ) from exc

    controller = server if server is not None else MemoryMcpServer()
    mcp = FastMCP("tree-rag-memory")

    @mcp.tool()
    def memory_save(
        category: str,
        text: str,
        source: str = "",
        workspace: str = "",
        tags: str = "",
    ) -> str:
        """Save an observation into a workspace branch (note, bug, code, wiki, decision, skill)."""
        res = controller.memory_save(
            category=category,
            text=text,
            source=source,
            workspace=workspace,
            tags=tags or None,
        )
        return json.dumps(res, ensure_ascii=False)

    @mcp.tool()
    def memory_search(
        query: str,
        top_k: int = 5,
        category: str = "",
        scope: str = "workspace",
        workspace: str = "",
    ) -> str:
        """Semantic search in workspace ('workspace') or across all workspaces ('global')."""
        res = controller.memory_search(
            query=query,
            top_k=top_k,
            category=category,
            scope=scope,
            workspace=workspace,
        )
        return str(res.get("formatted") or json.dumps(res, ensure_ascii=False))

    @mcp.tool()
    def memory_list(
        category: str = "",
        limit: int = 50,
        scope: str = "workspace",
        workspace: str = "",
    ) -> str:
        """List recent memory entries with category, L0-L3 layer, and pillar metadata."""
        res = controller.memory_list(
            category=category,
            limit=limit,
            scope=scope,
            workspace=workspace,
        )
        return str(res.get("formatted") or json.dumps(res, ensure_ascii=False))

    @mcp.tool()
    def memory_health() -> str:
        """Inspect workspace index sizes, pending queue, concurrency guard, and embedders."""
        return json.dumps(controller.memory_health(), indent=2, ensure_ascii=False)

    @mcp.tool()
    def memory_merge_pending(
        workspace: str = "",
        force: bool = False,
    ) -> str:
        """Drain queued `_pending/<workspace>/*.md` markdown files into FAISS indices."""
        return json.dumps(
            controller.memory_merge_pending(workspace=workspace, force=force),
            indent=2,
            ensure_ascii=False,
        )

    return mcp


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `python -m scoped_memory.mcp_server`."""
    parser = argparse.ArgumentParser(
        prog="scoped_memory.mcp_server",
        description="Tree-RAG FastMCP stdio server built on ScopedMemoryStore.",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        default=True,
        help="Run the MCP server over stdio transport (default).",
    )
    parser.add_argument(
        "--print-tools",
        action="store_true",
        help="Print registered MCP tool specifications as JSON and exit.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.print_tools:
        print(json.dumps({"tools": TOOL_SPECIFICATIONS}, indent=2))
        return 0

    app = build_fastmcp_app()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

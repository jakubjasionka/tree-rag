#!/usr/bin/env python3
"""Python FAISS sidecar for the Tree-RAG Memory MCP Hub dashboard (`dashboard/server.js`).

Communicates over stdio using JSON Lines (or single-shot `--op <name> --payload <json>`):
- Uses `ScopedMemoryStore` and `MemoryMcpServer` when `scoped_memory` is importable.
- Falls back to `MockEmbedder` automatically when `TREE_RAG_EMBED_URL` is not configured
  or unreachable so offline dev, CI, and smoke tests work with zero external services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path when invoked directly from `dashboard/`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scoped_memory.embed import CpuHttpEmbedder, Embedder, MockEmbedder  # noqa: E402
from scoped_memory.mcp_server import (  # noqa: E402
    MemoryMcpServer,
    enrich_document,
    generate_mermaid_canvas,
    probe_http_endpoint,
)
from scoped_memory.store import ScopedMemoryStore  # noqa: E402


def _resolve_embedder(mode: str = "base") -> Embedder:
    """Return an HTTP embedder when configured and reachable for `mode`, else `MockEmbedder`."""
    if mode == "plus":
        plus_url = os.environ.get("TREE_RAG_EMBED_URL_PLUS", "").strip()
        if plus_url and probe_http_endpoint(plus_url).get("reachable"):
            plus_model = (
                os.environ.get("TREE_RAG_EMBED_MODEL_PLUS")
                or "custom-4096d-embedder"
            ).strip()
            return CpuHttpEmbedder(
                endpoint_url=plus_url,
                model_name=plus_model,
                dim=4096,
                passage_prefix="",
                query_prefix=(
                    "Instruct: Retrieve relevant technical project notes and code decisions.\n"
                    "Query: "
                ),
            )
    base_url = os.environ.get("TREE_RAG_EMBED_URL", "").strip()
    if base_url and probe_http_endpoint(base_url).get("reachable"):
        base_model = (
            os.environ.get("TREE_RAG_EMBED_MODEL")
            or "intfloat/multilingual-e5-small"
        ).strip()
        return CpuHttpEmbedder(
            endpoint_url=base_url,
            model_name=base_model,
        )
    return MockEmbedder()


def _build_server(base_dir: str | Path, mode: str = "base") -> MemoryMcpServer:
    store = ScopedMemoryStore(
        base_dir=Path(base_dir),
        embedder=_resolve_embedder(mode),
        min_score=0.0,
        require_global_confirmation=False,
    )
    return MemoryMcpServer(store=store)


def handle_request(req: dict[str, Any]) -> dict[str, Any]:
    """Execute a single JSON sidecar command and return a JSON-serializable response."""
    op = str(req.get("op") or "").strip().lower()
    base_dir = str(
        req.get("base_dir")
        or os.environ.get("TREE_RAG_BASE_DIR")
        or "./.treerag_data"
    )
    mode = str(req.get("mode") or "base").strip().lower()
    srv = _build_server(base_dir=base_dir, mode=mode)
    store = srv.store

    if op == "search":
        query = str(req.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "Search query cannot be empty."}
        scope = "global" if str(req.get("scope") or "global").lower() == "global" else "workspace"
        workspace = str(req.get("workspace") or "").strip()
        category = str(req.get("category") or "").strip().lower()
        top_k = max(1, min(30, int(req.get("top_k") or 8)))
        res = srv.memory_search(
            query=query,
            top_k=top_k,
            category=category,
            scope=scope,
            workspace=workspace,
        )
        res["mode"] = mode
        res["backend"] = "faiss"
        res["warnings"] = []
        return res

    if op == "neighbors":
        t0 = time.perf_counter()
        workspace = str(req.get("workspace") or "").strip()
        doc_id = str(req.get("id") or req.get("doc_id") or "").strip()
        top_k = max(1, min(15, int(req.get("top_k") or 6)))
        if not workspace or not doc_id:
            return {"ok": False, "error": "Fields 'workspace' and 'id' are required."}
        ws_index = store.get_workspace_index(workspace)
        target_doc: dict[str, Any] | None = None
        for d in ws_index.docs:
            if d.get("id") == doc_id:
                target_doc = d
                break
        if target_doc is None:
            return {
                "ok": False,
                "error": f"Document {doc_id} not found in workspace [{workspace}].",
            }
        target_src = str(target_doc.get("source") or "manual")
        same_source = [
            enrich_document(d, ws_index.name)
            for d in ws_index.docs
            if d.get("id") != doc_id and str(d.get("source") or "manual") == target_src
        ]
        raw_neighbors = store.nearest_neighbors(
            doc_id=doc_id,
            top_k=top_k,
            workspace=ws_index.name,
        )
        sem_neighbors: list[dict[str, Any]] = []
        for score, ndoc in raw_neighbors:
            ed = enrich_document(ndoc, ws_index.name)
            ed["score"] = round(float(score), 4)
            sem_neighbors.append(ed)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": True,
            "backend": "faiss",
            "target": enrich_document(target_doc, ws_index.name),
            "same_source": same_source[:15],
            "semantic_neighbors": sem_neighbors,
            "latency_ms": latency_ms,
        }

    if op == "save":
        workspace = str(req.get("workspace") or "").strip()
        category = str(req.get("category") or "note").strip().lower()
        text = str(req.get("text") or "").strip()
        source = str(req.get("source") or "hub-ui").strip()
        tags = req.get("tags")
        if not text:
            return {"ok": False, "error": "Field 'text' cannot be empty."}
        res = srv.memory_save(
            category=category,
            text=text,
            source=source,
            workspace=workspace,
            tags=tags,
        )
        return res

    if op == "update":
        workspace = str(req.get("workspace") or "").strip()
        doc_id = str(req.get("id") or "").strip()
        if not workspace or not doc_id:
            return {"ok": False, "error": "Fields 'workspace' and 'id' are required."}
        res = store.update_doc(
            doc_id=doc_id,
            text=req.get("text"),
            category=req.get("category"),
            source=req.get("source"),
            tags=req.get("tags"),
            workspace=workspace,
        )
        if res.get("ok") and isinstance(res.get("doc"), dict):
            res["doc"] = enrich_document(res["doc"], store._resolve_workspace(workspace))
            res["message"] = f"Updated record {doc_id} in [{workspace}]."
        return res

    if op == "delete":
        workspace = str(req.get("workspace") or "").strip()
        doc_id = str(req.get("id") or "").strip()
        if not workspace or not doc_id:
            return {"ok": False, "error": "Fields 'workspace' and 'id' are required."}
        res = store.delete_doc(doc_id=doc_id, workspace=workspace)
        return res

    if op == "merge":
        workspace = str(req.get("workspace") or "").strip()
        force = bool(req.get("force", True))
        res = srv.memory_merge_pending(workspace=workspace, force=force)
        res["output"] = (
            f"Merged {res.get('merged', 0)} pending files "
            f"(deduped={res.get('deduped', 0)}, remaining={res.get('remaining', 0)})."
        )
        return res

    if op == "mermaid":
        workspace = str(req.get("workspace") or "global").strip() or "global"
        return generate_mermaid_canvas(store, workspace=workspace)

    if op == "workspace-sync":
        stores = store.all_workspace_indices()
        reports = []
        for ws_name in sorted(stores.keys()):
            reports.append(store.heal_workspace(ws_name))
        return {
            "ok": True,
            "workspaces_synced": len(reports),
            "reports": reports,
            "output": f"Synchronized and verified {len(reports)} workspace FAISS indices.",
        }

    if op == "health":
        return srv.memory_health()

    return {"ok": False, "error": f"Unknown sidecar operation: {op}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Tree-RAG Dashboard FAISS Sidecar")
    parser.add_argument("--op", help="Single operation name")
    parser.add_argument("--payload", help="JSON payload string for single operation")
    args = parser.parse_args()

    if args.op or args.payload:
        try:
            payload = json.loads(args.payload or "{}")
            if args.op and "op" not in payload:
                payload["op"] = args.op
            out = handle_request(payload)
        except Exception as exc:
            out = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            resp = handle_request(req)
        except Exception as exc:
            resp = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

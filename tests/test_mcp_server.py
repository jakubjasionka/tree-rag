"""Unit tests for `scoped_memory.mcp_server`, Mermaid generator, and `search_sidecar`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dashboard import search_sidecar
from scoped_memory.embed import CpuHttpEmbedder, MockEmbedder
from scoped_memory.mcp_server import (
    MemoryMcpServer,
    build_env_concurrency_detector,
    build_fastmcp_app,
    create_embedder_from_env,
    enrich_document,
    generate_mermaid_canvas,
    main,
)
from scoped_memory.store import ScopedMemoryStore


def _make_mcp_server(
    tmp_path: Path,
    default_workspace: str = "sample_project",
    concurrency_detector: Any = None,
) -> MemoryMcpServer:
    store = ScopedMemoryStore(
        base_dir=tmp_path,
        default_workspace=default_workspace,
        embedder=MockEmbedder(),
        min_score=0.10,
        require_global_confirmation=False,
        concurrency_detector=concurrency_detector,
    )
    return MemoryMcpServer(store=store)


def test_mcp_save_exposes_layer_and_pillar_mappings(tmp_path: Path) -> None:
    """Every saved category maps deterministically to L0-L3 layer and 4-pillar taxonomy."""
    srv = _make_mcp_server(tmp_path)

    cases = [
        ("note", "L0", "chat", "Session scratchpad note about local dev setup."),
        ("assumption", "L1", "chat", "Assume upstream weather radar updates every 300 seconds."),
        ("bug", "L1", "code", "Fix race condition when refreshing expired JWT tokens."),
        ("code", "L2", "code", "Implement sliding-window rate limiter middleware in FastAPI."),
        ("wiki", "L2", "wiki", "Runbook for rotating API gateway TLS certificates."),
        ("decision", "L3", "skill", "ADR-001: Use token-bucket rate limiting at API gateway."),
        ("skill", "L3", "skill", "Reusable SOP for zero-downtime FAISS index atomic swap."),
    ]

    for category, expected_layer, expected_pillar, text in cases:
        res = srv.memory_save(
            category=category,
            text=text,
            source=f"src/{category}.py",
            workspace="sample_project",
            tags=["acme-weather-api", category],
        )
        assert res["ok"] is True
        assert res["status"] == "saved"
        assert res["category"] == category
        assert res["layer"] == expected_layer
        assert res["pillar"] == expected_pillar
        assert res["doc"]["layer"] == expected_layer
        assert res["doc"]["pillar"] == expected_pillar
        assert category in res["doc"]["tags"]


def test_mcp_search_and_list_workspace_and_global(tmp_path: Path) -> None:
    """memory_search and memory_list return enriched records, highlights, and formatted output."""
    srv = _make_mcp_server(tmp_path)

    srv.memory_save(
        category="decision",
        text="Use token-bucket rate limiting (120 req/min) at the gateway with Redis.",
        source="docs/adr/001.md",
        workspace="sample_project",
        tags="gateway,redis",
    )
    srv.memory_save(
        category="bug",
        text="Fix timezone DST offset bug in hourly precipitation normalizer.",
        source="src/normalizer.py",
        workspace="demo_workspace",
        tags="timezone,dst",
    )

    ws_search = srv.memory_search(
        query="token-bucket rate limiting Redis gateway",
        top_k=5,
        scope="workspace",
        workspace="sample_project",
    )
    assert ws_search["ok"] is True
    assert ws_search["count"] == 1
    assert ws_search["results"][0]["layer"] == "L3"
    assert ws_search["results"][0]["pillar"] == "skill"
    assert "sample_project" in ws_search["highlights"]["workspaces"]
    assert "sample_project::decision" in ws_search["highlights"]["categories"]
    assert "layer=L3" in ws_search["formatted"]

    global_search = srv.memory_search(
        query="timezone DST offset precipitation normalizer",
        top_k=5,
        scope="global",
    )
    assert global_search["ok"] is True
    assert global_search["count"] >= 1
    assert any(r["workspace"] == "demo_workspace" for r in global_search["results"])

    listed = srv.memory_list(scope="global", limit=10)
    assert listed["ok"] is True
    assert listed["count"] == 2
    assert all("layer" in r and "pillar" in r for r in listed["records"])
    assert "L3" in listed["formatted"]
    assert "L1" in listed["formatted"]


def test_mcp_health_and_pending_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """memory_health reports index sizes and pending queue; memory_merge_pending compacts queue."""
    monkeypatch.delenv("TREE_RAG_EMBED_URL", raising=False)
    monkeypatch.delenv("TREE_RAG_EMBED_URL_PLUS", raising=False)

    blocked_flag = {"active": True}

    def detector() -> tuple[bool, str]:
        if blocked_flag["active"]:
            return True, "simulated concurrent indexer"
        return False, ""

    srv = _make_mcp_server(tmp_path, concurrency_detector=detector)
    queued = srv.memory_save(
        category="bug",
        text="Queued observation while concurrent writer is active.",
        source="src/worker.py",
        workspace="sample_project",
    )
    assert queued["ok"] is True
    assert queued["status"] == "pending"
    assert queued["layer"] == "L1"

    health_before = srv.memory_health()
    assert health_before["ok"] is True
    assert health_before["pending_total"] == 1
    assert health_before["embedder_type"] == "MockEmbedder"
    assert health_before["concurrency_detection_enabled"] is True
    assert health_before["layer_mapping"]["decision"] == "L3"
    assert health_before["pillar_mapping"]["bug"] == "code"

    blocked_flag["active"] = False
    srv.store.guard.invalidate_cache()

    merged = srv.memory_merge_pending(workspace="sample_project")
    assert merged["ok"] is True
    assert merged["merged"] == 1
    assert merged["remaining"] == 0

    health_after = srv.memory_health()
    assert health_after["pending_total"] == 0
    assert health_after["total_docs"] == 1
    assert health_after["total_index_bytes"] > 0


def test_env_embedder_and_concurrency_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedder and concurrency detector are controlled strictly via environment variables."""
    monkeypatch.delenv("TREE_RAG_EMBED_URL", raising=False)
    monkeypatch.delenv("TREE_RAG_CONCURRENCY_PATTERNS", raising=False)
    monkeypatch.delenv("MEMORY_CONCURRENCY_PATTERNS", raising=False)

    emb_default = create_embedder_from_env()
    assert isinstance(emb_default, MockEmbedder)
    assert build_env_concurrency_detector() is None

    monkeypatch.setenv("TREE_RAG_EMBED_URL", "http://127.0.0.1:8089/v1")
    monkeypatch.setenv("MEMORY_EMBED_TIMEOUT", "9.5")
    emb_http = create_embedder_from_env()
    assert isinstance(emb_http, CpuHttpEmbedder)
    assert emb_http.endpoint_url == "http://127.0.0.1:8089/v1"
    assert emb_http._timeout_sec == 9.5

    monkeypatch.setenv("TREE_RAG_CONCURRENCY_PATTERNS", "nonexistent_proc_xyz_987654")
    det = build_env_concurrency_detector()
    assert det is not None
    blocked, reason = det()
    assert blocked is False
    assert reason == ""


def test_mermaid_generator_global_and_workspace(tmp_path: Path) -> None:
    """generate_mermaid_canvas builds valid Mermaid flowcharts and scrubs special characters."""
    srv = _make_mcp_server(tmp_path)
    srv.memory_save(
        category="decision",
        text='Use "strict" [JSON] schema {validation} <v2> & `sanitize` | labels; ok',
        source="src/schema[v2].py",
        workspace="sample_project",
    )
    srv.memory_save(
        category="code",
        text=" Secondary workspace observation for global flowchart.",
        source="src/app.py",
        workspace="demo_workspace",
    )

    global_mmd = generate_mermaid_canvas(srv.store, workspace="global")
    assert global_mmd["ok"] is True
    assert global_mmd["workspace"] == "global"
    assert global_mmd["docs_count"] == 2
    assert global_mmd["mermaid"].startswith("flowchart LR" + chr(10))
    assert "sample_project" in global_mmd["mermaid"]
    assert "demo_workspace" in global_mmd["mermaid"]

    ws_mmd = srv.generate_mermaid(workspace="sample_project")
    assert ws_mmd["ok"] is True
    assert ws_mmd["workspace"] == "sample_project"
    assert ws_mmd["docs_count"] == 1
    assert "LYR_L3" in ws_mmd["mermaid"]
    assert "WS --> LYR_L3" in ws_mmd["mermaid"]


def test_cli_print_tools_and_enrich_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m scoped_memory.mcp_server --print-tools` outputs JSON tool specifications."""
    rc = main(["--print-tools"])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    tool_names = [t["name"] for t in captured["tools"]]
    assert tool_names == [
        "memory_save",
        "memory_search",
        "memory_list",
        "memory_health",
        "memory_merge_pending",
    ]

    enriched = enrich_document(
        {"id": "mem_1", "text": "Fix #auth token bug", "category": "bug", "source": "api.py"},
        workspace="sample_project",
    )
    assert enriched["layer"] == "L1"
    assert enriched["pillar"] == "code"
    assert "auth" in enriched["tags"]

    # Verify build_fastmcp_app either returns a FastMCP instance or raises a clean RuntimeError
    try:
        app = build_fastmcp_app()
        assert app is not None
    except RuntimeError as exc:
        assert "FastMCP is not installed" in str(exc)


def test_dashboard_search_sidecar_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify dashboard/search_sidecar.py save, search, neighbors, update, mermaid, and delete."""
    monkeypatch.delenv("TREE_RAG_EMBED_URL", raising=False)
    monkeypatch.delenv("TREE_RAG_EMBED_URL_PLUS", raising=False)

    r1 = search_sidecar.handle_request(
        {
            "op": "save",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
            "category": "decision",
            "source": "docs/adr/001.md",
            "tags": ["acme-weather-api", "redis"],
            "text": "Use Redis token bucket rate limiting for weather forecast endpoints.",
        }
    )
    assert r1["ok"] is True
    doc_id = r1["id"]

    r2 = search_sidecar.handle_request(
        {
            "op": "save",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
            "category": "code",
            "source": "docs/adr/001.md",
            "tags": ["acme-weather-api", "middleware"],
            "text": "Token bucket middleware implementation using Redis Lua script.",
        }
    )
    assert r2["ok"] is True

    s_res = search_sidecar.handle_request(
        {
            "op": "search",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
            "scope": "workspace",
            "query": "Redis token bucket rate limiting",
            "top_k": 5,
        }
    )
    assert s_res["ok"] is True
    assert s_res["backend"] == "faiss"
    assert s_res["count"] >= 1

    n_res = search_sidecar.handle_request(
        {
            "op": "neighbors",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
            "id": doc_id,
            "top_k": 3,
        }
    )
    assert n_res["ok"] is True
    assert len(n_res["same_source"]) == 1
    assert len(n_res["semantic_neighbors"]) == 1

    u_res = search_sidecar.handle_request(
        {
            "op": "update",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
            "id": doc_id,
            "category": "decision",
            "source": "docs/adr/001.md",
            "text": "Use Redis sliding-window rate limiting (200 req/min) for weather endpoints.",
        }
    )
    assert u_res["ok"] is True
    assert "200 req/min" in u_res["doc"]["text"]

    m_res = search_sidecar.handle_request(
        {
            "op": "mermaid",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
        }
    )
    assert m_res["ok"] is True
    assert "flowchart LR" in m_res["mermaid"]

    d_res = search_sidecar.handle_request(
        {
            "op": "delete",
            "base_dir": str(tmp_path),
            "workspace": "sample_project",
            "id": doc_id,
        }
    )
    assert d_res["ok"] is True
    assert d_res["remaining"] == 1

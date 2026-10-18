"""Unit tests for Tree-RAG (`scoped_memory`) on CPU using `MockEmbedder`."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from scoped_memory import (
    EmptyContext,
    MockEmbedder,
    PeerNode,
    ScopedMemoryStore,
    SearchResult,
    SyncPolicy,
    _atomic_write_faiss,
    _atomic_write_json,
    _scrub_secrets,
    heal_workspace_conflicts,
    merge_doc_collections,
    parse_pending_markdown,
    pending_count,
    release_lock,
    sanitize,
    text_hash,
    try_workspace_lock,
)


@pytest.fixture()
def embedder() -> MockEmbedder:
    return MockEmbedder(dim=384)


@pytest.fixture()
def store(tmp_path: Path, embedder: MockEmbedder) -> ScopedMemoryStore:
    return ScopedMemoryStore(
        base_dir=tmp_path / "memory",
        default_workspace="workspace_alpha",
        embedder=embedder,
        min_score=0.25,
        require_global_confirmation=True,
    )


def test_a_zero_pollution_empty_workspace(store: ScopedMemoryStore) -> None:
    """(a) Zero-pollution: empty workspace returns EmptyContext, never random vectors."""
    res = store.search(
        query="How is the database connection pool configured?",
        scope="workspace",
        workspace="empty_branch",
    )
    assert isinstance(res, EmptyContext)
    assert res.is_empty is True
    assert bool(res) is False
    assert len(res) == 0
    assert list(res) == []
    assert res.reason == "empty_workspace"
    assert res.requires_confirmation() is True

    # Also verify sub-threshold query in a populated workspace returns EmptyContext
    store.save(
        category="decision",
        text="Kubernetes ingress controller uses Envoy proxy with mTLS termination.",
        workspace="populated_branch",
    )
    unrelated = store.search(
        query="PostgreSQL vacuum autovacuum naptime parameter tuning",
        scope="workspace",
        workspace="populated_branch",
        min_score=0.85,
    )
    assert isinstance(unrelated, EmptyContext)
    assert unrelated.reason == "below_similarity_threshold"
    assert unrelated.requires_confirmation() is True


def test_b_workspace_scope_isolation(store: ScopedMemoryStore) -> None:
    """(b) Scope isolation: workspace A documents are invisible when querying workspace B."""
    store.save(
        category="code",
        text="FastAPI authentication middleware validates Ed25519 JWT signatures.",
        source="auth.py",
        workspace="workspace_a",
    )
    store.save(
        category="bug",
        text="Redis sentinel failover timeout reduced to 1500 milliseconds.",
        source="redis_client.py",
        workspace="workspace_b",
    )

    # Query workspace_b for workspace_a's JWT knowledge -> must not leak
    res_b = store.search(
        query="FastAPI authentication middleware Ed25519 JWT signatures",
        scope="workspace",
        workspace="workspace_b",
        min_score=0.35,
    )
    assert isinstance(res_b, EmptyContext)
    assert len(res_b) == 0

    # Query workspace_a for its own knowledge -> hits cleanly
    res_a = store.search(
        query="FastAPI authentication middleware Ed25519 JWT signatures",
        scope="workspace",
        workspace="workspace_a",
        min_score=0.35,
    )
    assert isinstance(res_a, SearchResult)
    assert len(res_a) == 1
    assert res_a[0].workspace == "workspace_a"
    assert "Ed25519 JWT" in res_a[0].doc["text"]

    # Category filter within workspace_a
    res_cat_miss = store.search(
        query="FastAPI authentication middleware Ed25519 JWT signatures",
        category="bug",
        scope="workspace",
        workspace="workspace_a",
    )
    assert isinstance(res_cat_miss, EmptyContext)


def test_c_hitl_global_escalation_requires_confirmation(store: ScopedMemoryStore) -> None:
    """(c) HITL: global escalation requires explicit confirmation when local scope misses."""
    store.save(
        category="decision",
        text="Zero-copy shared memory ringbuffer uses 64-byte cache-line alignment.",
        source="ringbuffer.hpp",
        workspace="systems_engine",
    )

    query = "Zero-copy shared memory ringbuffer 64-byte cache-line alignment"

    # 1. Local search in active default workspace (`workspace_alpha`) misses
    local_res = store.search(query=query, scope="workspace", workspace="workspace_alpha")
    assert isinstance(local_res, EmptyContext)
    assert local_res.requires_confirmation() is True
    assert store.requires_confirmation(local_res) is True
    assert store.requires_confirmation(query, workspace="workspace_alpha") is True

    # 2. Unconfirmed global search is blocked by the HITL gate
    unconfirmed_global = store.search(
        query=query,
        scope="global",
        workspace="workspace_alpha",
        confirmed=False,
    )
    assert isinstance(unconfirmed_global, EmptyContext)
    assert unconfirmed_global.reason == "hitl_confirmation_required"
    assert unconfirmed_global.requires_confirmation() is True

    # 3. Default global search when store has require_global_confirmation=True also blocks
    default_global = store.search(
        query=query,
        scope="global",
        workspace="workspace_alpha",
    )
    assert isinstance(default_global, EmptyContext)
    assert default_global.reason == "hitl_confirmation_required"

    # 4. Confirmed global search succeeds and tags hits with [workspace=systems_engine]
    confirmed_global = store.search(
        query=query,
        scope="global",
        workspace="workspace_alpha",
        confirmed=True,
    )
    assert isinstance(confirmed_global, SearchResult)
    assert len(confirmed_global) == 1
    hit = confirmed_global[0]
    assert hit.workspace == "systems_engine"
    assert hit.label == "[workspace=systems_engine]"
    assert "[workspace=systems_engine]" in confirmed_global.format()
    assert confirmed_global.requires_confirmation() is False


def test_d_atomic_write_json_and_faiss(tmp_path: Path) -> None:
    """(d) Atomic write: tmp + fsync + rename leaves no .tmp artifacts and preserves valid state."""
    json_path = tmp_path / "ws_atomic" / "docs.json"
    faiss_path = tmp_path / "ws_atomic" / "index.faiss"

    payload = [{"id": "mem_001", "text": "Atomic persistence test", "category": "note"}]
    _atomic_write_json(json_path, payload)
    assert json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload

    idx = faiss.IndexFlatIP(384)
    vec = np.ones((1, 384), dtype="float32") / np.sqrt(384)
    idx.add(vec)
    _atomic_write_faiss(idx, faiss_path)
    assert faiss_path.exists()

    loaded_idx = faiss.read_index(str(faiss_path))
    assert int(loaded_idx.ntotal) == 1
    assert int(loaded_idx.d) == 384

    # Verify zero leftover temporary files in directory
    leftover_tmps = list(json_path.parent.glob("*.tmp*")) + list(json_path.parent.glob(".*.tmp*"))
    assert leftover_tmps == []


def test_e_dedup_by_sha256_hash(store: ScopedMemoryStore) -> None:
    """(e) Deduplication by content hash prevents duplicate indexing across whitespace."""
    first = store.save(
        category="decision",
        text="Use POSIX flock per workspace directory to avoid global write serialization.",
        source="locks.py",
        workspace="dedup_ws",
    )
    assert first["status"] == "saved"

    # Same text with extra whitespace and newlines must match by normalized SHA-256 hash
    nl = chr(10)
    second = store.save(
        category="decision",
        text=(
            f"  Use POSIX   flock per workspace directory{nl}"
            f"to avoid global write serialization.  "
        ),
        source="locks.py",
        workspace="dedup_ws",
    )
    assert second["status"] == "duplicate"
    assert second["id"] == first["id"]

    ws_idx = store.get_workspace_index("dedup_ws")
    assert len(ws_idx.docs) == 1
    assert ws_idx.index is not None
    assert int(ws_idx.index.ntotal) == 1


def test_f_pending_queue_read_only_mode_and_merge(
    tmp_path: Path, embedder: MockEmbedder
) -> None:
    """(f) Read-only mode & lock contention route writes to _pending/<ws>/*.md and merge cleanly."""
    base_dir = tmp_path / "pending_store"
    store = ScopedMemoryStore(
        base_dir=base_dir,
        default_workspace="concurrent_ws",
        embedder=embedder,
        read_only=True,
    )

    # 1. Explicit read_only=True writes to _pending/concurrent_ws/*.md
    res1 = store.save(
        category="bug",
        text="Fixed race condition during non-blocking flock acquisition.",
        source="worker-1",
    )
    assert res1["status"] == "pending"
    pending_file = Path(res1["pending_path"])
    assert pending_file.exists()
    assert pending_count(base_dir, "concurrent_ws") == 1

    cat, body, src, ws = parse_pending_markdown(pending_file)
    assert cat == "bug"
    assert "Fixed race condition" in body
    assert src == "worker-1"
    assert ws == "concurrent_ws"

    # Main FAISS index is still empty while in read-only mode
    assert len(store.get_workspace_index("concurrent_ws").docs) == 0

    # 2. Switch read_only off, but hold the workspace lock externally -> also routes to _pending
    store.read_only = False
    acquired, lock_fh = try_workspace_lock(base_dir, "concurrent_ws")
    assert acquired is True
    try:
        res2 = store.save(
            category="decision",
            text="Queue entries in markdown with YAML frontmatter when lock is busy.",
            source="worker-2",
        )
        assert res2["status"] == "pending"
        assert pending_count(base_dir, "concurrent_ws") == 2
    finally:
        release_lock(lock_fh)

    # 3. Now that lock is free and read_only is False, merge pending queue into FAISS
    merge_res = store.merge_pending(workspace="concurrent_ws")
    assert merge_res["ok"] is True
    assert merge_res["merged"] == 2
    assert merge_res["remaining"] == 0
    assert pending_count(base_dir, "concurrent_ws") == 0

    ws_idx = store.get_workspace_index("concurrent_ws")
    assert len(ws_idx.docs) == 2
    assert ws_idx.index is not None
    assert int(ws_idx.index.ntotal) == 2


def test_g_sanitize_and_scrub_secrets(store: ScopedMemoryStore) -> None:
    """(g) Secret scrubbing redacts tokens before persistence and sanitize() canonicalizes names."""
    assert sanitize("  My Project / Feature-Branch #1! ") == "my_project_feature_branch_1"
    assert sanitize("") == "default"
    assert sanitize("___") == "default"

    # Construct synthetic token patterns dynamically so static scanners see zero literal secrets
    fake_gh = "ghp_" + ("A1b2C3d4E5" * 3)
    fake_pat = "github_pat_" + ("Z9y8X7w6V5_" * 2)
    fake_oa = "sk-" + ("MnOpQrStUvWxYz0123456789")
    fake_slack = "xoxb-" + "1234567890-abcdefABCDEF12345"

    raw_note = (
        f"Exported credentials {fake_gh} and {fake_pat} plus "
        f"{fake_oa} and slack {fake_slack} in debug log."
    )
    scrubbed = _scrub_secrets(raw_note)
    assert fake_gh not in scrubbed
    assert fake_pat not in scrubbed
    assert fake_oa not in scrubbed
    assert fake_slack not in scrubbed
    assert "ghp_***REDACTED***" in scrubbed
    assert "github_pat_***REDACTED***" in scrubbed
    assert "sk-***REDACTED***" in scrubbed
    assert "xox_***REDACTED***" in scrubbed

    # Ensure ScopedMemoryStore.save scrubs secrets prior to writing docs.json
    saved = store.save(category="note", text=raw_note, workspace="sec_ws")
    assert saved["status"] == "saved"
    persisted_text = saved["doc"]["text"]
    assert fake_gh not in persisted_text
    assert "ghp_***REDACTED***" in persisted_text


def test_conflict_free_multi_node_merge_and_sync_abstraction(
    tmp_path: Path, embedder: MockEmbedder
) -> None:
    """Verify lossless 2-way union merge and PeerNode / SyncPolicy abstraction."""
    peer = PeerNode(node_id="edge-node-01", endpoint_url="http://peer-node.example:9100/v1")
    policy = SyncPolicy(near_duplicate_threshold=0.975)
    store_a = ScopedMemoryStore(
        base_dir=tmp_path / "node_a",
        embedder=embedder,
        sync_policy=policy,
        peers=[peer],
    )
    store_b = ScopedMemoryStore(
        base_dir=tmp_path / "node_b",
        embedder=embedder,
        sync_policy=policy,
    )

    store_a.save("decision", "Node A observation on vector index sharding.", workspace="shared")
    store_b.save("code", "Node B implementation of async batch queue.", workspace="shared")

    idx_a = store_a.get_workspace_index("shared")
    idx_b = store_b.get_workspace_index("shared")
    assert idx_a.index is not None
    assert idx_b.index is not None

    vecs_a = {text_hash(d["text"]): idx_a.index.reconstruct(i) for i, d in enumerate(idx_a.docs)}
    vecs_b = {text_hash(d["text"]): idx_b.index.reconstruct(i) for i, d in enumerate(idx_b.docs)}

    merged_docs, merged_index = merge_doc_collections(
        [(idx_a.docs, vecs_a), (idx_b.docs, vecs_b)],
        embedder=embedder,
    )
    assert len(merged_docs) == 2
    assert int(merged_index.ntotal) == 2

    # Simulate a *.sync-conflict-* file in node_a and heal it losslessly
    ws_dir_a = store_a.base_dir / "shared"
    conflict_file = ws_dir_a / "docs.json.sync-conflict-20261018-120000"
    conflict_file.write_text(json.dumps(idx_b.docs), encoding="utf-8")

    heal_res = heal_workspace_conflicts(ws_dir_a, embedder=embedder, policy=policy)
    assert heal_res["docs"] == 2
    assert heal_res["removed_conflicts"] == 1
    assert not conflict_file.exists()

    health = store_a.health()
    assert health["peers"][0]["endpoint_url"] == "http://peer-node.example:9100/v1"

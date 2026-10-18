"""Unit tests for model-agnostic embedding swap (`swap.py`) and universal RAG sync (`sync.py`)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import pytest

from scoped_memory import (
    ConflictPolicy,
    DimensionMismatchError,
    Doc,
    Embedder,
    EmbeddingModelConfig,
    FirstWriteWins,
    HttpTransport,
    LastWriteWins,
    LocalFsTransport,
    MockEmbedder,
    RAGSyncEngine,
    ScopedMemoryStore,
    UnionMerge,
    execute_swap,
    plan_swap,
)


def _seed_store_8d(tmp_path: Path, count: int = 5) -> ScopedMemoryStore:
    store = ScopedMemoryStore(
        base_dir=tmp_path / "swap_store",
        default_workspace="notes_ws",
        embedder=MockEmbedder(dim=8),
    )
    topics = [
        ("decision", "Zero-pollution hierarchical RAG isolates workspace branches."),
        ("code", "FAISS IndexFlatIP inner product over L2-normalized vectors."),
        ("bug", "Resolved POSIX flock contention by queuing markdown files."),
        ("wiki", "Notes in docs.json are decoupled from derived vector indices."),
        ("skill", "Atomic file replacement uses mkstemp, fsync, and rename."),
        ("assumption", "CPU embedding offload preserves 100% of GPU VRAM for LLMs."),
    ]
    items = []
    for i in range(count):
        cat, base_txt = topics[i % len(topics)]
        items.append((cat, f"{base_txt} (entry #{i + 1})", f"doc_{i + 1}.md"))
    store.add_many(items, workspace="notes_ws")
    return store


# ---------------------------------------------------------------------------
# Module 1: scoped_memory/swap.py tests (MockEmbedder dim=8 -> dim=16)
# ---------------------------------------------------------------------------


def test_swap_plan_no_dim_change(tmp_path: Path) -> None:
    """plan_swap with identical dimension sets requires_full_rebuild=False."""
    store = _seed_store_8d(tmp_path, count=4)
    same_dim_model = EmbeddingModelConfig(
        name="cpu-compact-8d-v2",
        dim=8,
        model_id="compact-v2-8d",
        server_url="http://embed.example:8089/v1",
        version="2.0.0",
    )

    plan = plan_swap(store, same_dim_model)
    assert plan.old_dim == 8
    assert plan.new_dim == 8
    assert plan.dim_changed is False
    assert plan.requires_full_rebuild is False
    assert plan.total_docs == 4
    assert plan.estimated_tokens > 0
    assert plan.risk_level == "low"
    assert "Zero text or metadata is lost" in plan.risk_summary


def test_swap_plan_dim_change(tmp_path: Path) -> None:
    """plan_swap with dimension change (8 -> 16) sets requires_full_rebuild=True."""
    store = _seed_store_8d(tmp_path, count=5)
    new_model_16d = EmbeddingModelConfig(
        name="cpu-upgraded-16d",
        dim=16,
        model_id="upgraded-v2-16d",
        server_url="http://embed.example:8089/v1",
        version="2.1.0",
    )

    plan = plan_swap(store, new_model_16d)
    assert plan.old_dim == 8
    assert plan.new_dim == 16
    assert plan.dim_changed is True
    assert plan.requires_full_rebuild is True
    assert plan.total_docs == 5
    assert plan.workspaces == {"notes_ws": 5}
    assert plan.estimated_tokens > 0
    assert "docs.json" in plan.risk_summary


def test_execute_swap_atomic(tmp_path: Path) -> None:
    """execute_swap atomically swaps index.faiss and keeps docs.json byte-identical."""
    store = _seed_store_8d(tmp_path, count=5)
    ws_dir = store.base_dir / "notes_ws"
    docs_path = ws_dir / "docs.json"
    index_path = ws_dir / "index.faiss"

    old_docs_bytes = docs_path.read_bytes()
    old_docs_list = json.loads(old_docs_bytes.decode("utf-8"))
    old_count = len(old_docs_list)
    assert old_count == 5

    new_model_16d = EmbeddingModelConfig(
        name="mock-16d",
        dim=16,
        model_id="mock-16d-v2",
        server_url="http://embed.example:8089/v1",
        version="2.0.0",
        embedder=MockEmbedder(dim=16),
    )

    report = execute_swap(store, new_model_16d, batch_size=2)
    assert report.swapped_docs == old_count
    assert report.old_dim == 8
    assert report.new_dim == 16
    assert report.requires_full_rebuild is True

    # 1. docs.json MUST be byte-for-byte identical (zero text or metadata mutation)
    assert docs_path.read_bytes() == old_docs_bytes
    ws_idx = store.get_workspace_index("notes_ws")
    assert len(ws_idx.docs) == old_count

    # 2. New FAISS index has ntotal == len(docs) and d == 16
    reloaded_index = faiss.read_index(str(index_path))
    assert int(reloaded_index.ntotal) == len(ws_idx.docs)
    assert int(reloaded_index.d) == 16

    # 3. Old 8-d FAISS index is preserved in _pre_swap_backup_<timestamp>/
    backup_dir = Path(report.backup_dir)
    assert backup_dir.exists()
    assert backup_dir.name.startswith("_pre_swap_backup_")
    backed_up_index_path = backup_dir / "notes_ws" / "index.faiss"
    assert backed_up_index_path.exists()
    old_backed_up_index = faiss.read_index(str(backed_up_index_path))
    assert int(old_backed_up_index.ntotal) == old_count
    assert int(old_backed_up_index.d) == 8

    # 4. embed_model.json written beside docs.json with model_hash, dim, version
    embed_meta_path = ws_dir / "embed_model.json"
    assert embed_meta_path.exists()
    embed_meta = json.loads(embed_meta_path.read_text(encoding="utf-8"))
    assert embed_meta["dim"] == 16
    assert embed_meta["version"] == "2.0.0"
    assert embed_meta["model_hash"] == new_model_16d.model_hash()


class _InterruptingEmbedder(Embedder):
    """Test helper that fails after `fail_after_calls` batch calls to simulate interruption."""

    def __init__(self, inner: Embedder, fail_after_calls: int | None = None) -> None:
        self._inner = inner
        self._fail_after = fail_after_calls
        self.calls: int = 0
        self.embedded_texts: list[str] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        if self._fail_after is not None and self.calls > self._fail_after:
            raise RuntimeError("Simulated interruption after checkpoint")
        self.embedded_texts.extend(texts)
        return self._inner.embed_passages(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._inner.embed_query(query)


def test_swap_resume(tmp_path: Path) -> None:
    """Interrupted swap checkpoints progress to swap_state.json and resumes without re-embedding."""
    store = _seed_store_8d(tmp_path, count=6)
    base_16d = MockEmbedder(dim=16)

    # First attempt: batch_size=2, fail on 2nd batch (after 1st batch of 2 docs is checkpointed)
    flaky_embedder = _InterruptingEmbedder(base_16d, fail_after_calls=1)
    cfg_interrupted = EmbeddingModelConfig(
        name="resumable-16d",
        dim=16,
        model_id="resumable-16d-v1",
        server_url="http://embed.example:8089/v1",
        version="1.0.0",
        embedder=flaky_embedder,
    )

    with pytest.raises(RuntimeError, match="Simulated interruption"):
        execute_swap(store, cfg_interrupted, batch_size=2)

    state_file = store.base_dir / "swap_state.json"
    assert state_file.exists()
    saved_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved_state["status"] == "in_progress"
    assert saved_state["completed_docs"] == 2
    assert len(saved_state["vectors"]) == 2

    # Second attempt: resume from checkpoint (resume_from=2) with healthy embedder
    healthy_spy = _InterruptingEmbedder(base_16d, fail_after_calls=None)
    cfg_resume = EmbeddingModelConfig(
        name="resumable-16d",
        dim=16,
        model_id="resumable-16d-v1",
        server_url="http://embed.example:8089/v1",
        version="1.0.0",
        embedder=healthy_spy,
    )

    report = execute_swap(store, cfg_resume, batch_size=2, resume_from=2)
    assert report.swapped_docs == 6
    assert report.resumed_from == 2
    assert report.batches_processed == 2  # Only the remaining 4 docs (2 batches of 2) were embedded
    assert len(healthy_spy.embedded_texts) == 4

    final_index = faiss.read_index(str(store.base_dir / "notes_ws" / "index.faiss"))
    assert int(final_index.ntotal) == 6
    assert int(final_index.d) == 16


def test_swap_report_counts(tmp_path: Path) -> None:
    """SwapReport accurately tracks swapped_docs, batches_processed, avg_batch_ms, and eta_ms."""
    store = _seed_store_8d(tmp_path, count=5)
    target_model = EmbeddingModelConfig(
        name="metrics-16d",
        dim=16,
        model_id="metrics-16d-v1",
        server_url="http://embed.example:8089/v1",
        version="1.0.0",
        embedder=MockEmbedder(dim=16),
    )

    report = execute_swap(store, target_model, batch_size=2)
    assert report.swapped_docs == 5
    assert report.workspaces_swapped == 1
    assert report.batches_processed == 3  # ceil(5 / 2) == 3 batches
    assert report.avg_batch_ms >= 0.0
    assert report.total_duration_ms >= 0.0
    assert report.eta_ms == 0.0
    assert report.old_dim == 8
    assert report.new_dim == 16


# ---------------------------------------------------------------------------
# Module 2: scoped_memory/sync.py tests
# ---------------------------------------------------------------------------


def test_sync_new_doc_one_way(tmp_path: Path) -> None:
    """One-way push sync transfers new documents from local store to peer and scrubs secrets."""
    emb = MockEmbedder(dim=16)
    local_store = ScopedMemoryStore(base_dir=tmp_path / "node_local", embedder=emb)
    peer_store = ScopedMemoryStore(base_dir=tmp_path / "node_peer", embedder=emb)

    fake_token = "ghp_" + ("X9y8Z7w6V5" * 3)
    local_store.save(
        category="decision",
        text=f"Use delta transport for standalone sync (token={fake_token}).",
        source="sync_design.md",
        workspace="alpha",
    )

    transport = LocalFsTransport(peer=peer_store)
    engine = RAGSyncEngine(
        local_store=local_store,
        peer=peer_store,
        transport=transport,
        conflict_policy=UnionMerge,
    )

    report = engine.sync_workspace("alpha", direction="push")
    assert report.pushed == 1
    assert report.pulled == 0

    peer_docs = peer_store.list(workspace="alpha")
    assert len(peer_docs) == 1
    assert "ghp_***REDACTED***" in peer_docs[0]["text"]
    assert fake_token not in peer_docs[0]["text"]


def test_sync_bidirectional_delta(tmp_path: Path) -> None:
    """Subsequent bidirectional sync transfers ONLY delta documents added since last_sync_hash."""
    emb = MockEmbedder(dim=16)
    store_a = ScopedMemoryStore(base_dir=tmp_path / "delta_a", embedder=emb)
    store_b = ScopedMemoryStore(base_dir=tmp_path / "delta_b", embedder=emb)

    store_a.save("note", "Initial document on node A about distributed locks.", workspace="core")
    store_b.save("code", "Initial document on node B about vector quantization.", workspace="core")

    engine = RAGSyncEngine(
        local_store=store_a,
        peer=store_b,
        transport=LocalFsTransport(peer=store_b),
        conflict_policy=UnionMerge,
    )

    # First sync: 1 pushed from A->B, 1 pulled from B->A
    r1 = engine.sync_workspace("core", direction="bidirectional")
    assert r1.pushed == 1
    assert r1.pulled == 1
    assert len(store_a.list(workspace="core")) == 2
    assert len(store_b.list(workspace="core")) == 2

    # Add 1 new doc on A and 1 new doc on B
    store_a.save("decision", "Delta-only round two update created on node A.", workspace="core")
    store_b.save("bug", "Delta-only round two bugfix created on node B.", workspace="core")

    # Second sync: MUST only push 1 and pull 1 (delta since last_sync_hash, NOT all 3)
    r2 = engine.sync_workspace("core", direction="bidirectional")
    assert r2.pushed == 1
    assert r2.pulled == 1
    assert len(store_a.list(workspace="core")) == 4
    assert len(store_b.list(workspace="core")) == 4

    # Third sync with zero new changes: 0 pushed, 0 pulled
    r3 = engine.sync_workspace("core", direction="bidirectional")
    assert r3.pushed == 0
    assert r3.pulled == 0


def test_sync_union_merge(tmp_path: Path) -> None:
    """UnionMerge deduplicates identical text across both stores into a single copy."""
    emb = MockEmbedder(dim=16)
    store_a = ScopedMemoryStore(base_dir=tmp_path / "union_a", embedder=emb)
    store_b = ScopedMemoryStore(base_dir=tmp_path / "union_b", embedder=emb)

    shared_text = "Deterministic SHA-256 content hashing guarantees idempotent union merges."
    store_a.save("decision", shared_text, source="node_a", workspace="dedup_ws")
    store_b.save("decision", shared_text, source="node_b", workspace="dedup_ws")

    engine = RAGSyncEngine(
        local_store=store_a,
        peer=store_b,
        transport=LocalFsTransport(peer=store_b),
        conflict_policy=UnionMerge,
    )

    report = engine.sync_workspace("dedup_ws", direction="bidirectional")
    assert report.conflicts_resolved >= 1
    assert len(store_a.list(workspace="dedup_ws")) == 1
    assert len(store_b.list(workspace="dedup_ws")) == 1


def test_sync_conflict_first_write(tmp_path: Path) -> None:
    """FirstWriteWins preserves the earlier/local document when conflicting edits share an ID."""
    emb = MockEmbedder(dim=16)
    store_a = ScopedMemoryStore(base_dir=tmp_path / "fww_a", embedder=emb)
    store_b = ScopedMemoryStore(base_dir=tmp_path / "fww_b", embedder=emb)

    res_a = store_a.save(
        "decision",
        "Primary storage backend is SQLite WAL mode (first write).",
        source="node_a",
        workspace="cfg_ws",
    )
    shared_id = res_a["id"]
    ws_a = store_a.get_workspace_index("cfg_ws")
    ws_a.docs[0]["time"] = "2026-10-18 08:00:00"
    ws_a._persist()

    # Create a conflicting version with the same document ID but a later timestamp on store_b
    res_b = store_b.save(
        "decision",
        "Primary storage backend migrated to PostgreSQL cluster (second write).",
        source="node_b",
        workspace="cfg_ws",
    )
    assert res_b["status"] == "saved"
    ws_b = store_b.get_workspace_index("cfg_ws")
    ws_b.docs[0]["id"] = shared_id
    ws_b.docs[0]["time"] = "2026-10-18 12:00:00"
    ws_b._persist()

    engine = RAGSyncEngine(
        local_store=store_a,
        peer=store_b,
        transport=LocalFsTransport(peer=store_b, conflict_policy=FirstWriteWins),
        conflict_policy=FirstWriteWins,
    )

    report = engine.sync_workspace("cfg_ws", direction="bidirectional")
    assert report.conflict_policy_used == ConflictPolicy.FIRST_WRITE_WINS.value
    assert report.conflicts_resolved >= 1

    docs_a = store_a.list(workspace="cfg_ws")
    assert len(docs_a) == 1
    assert "first write" in docs_a[0]["text"]


def test_sync_dim_mismatch_rejected(tmp_path: Path) -> None:
    """Vectors or peer stores with mismatched embedding dimensions are rejected before write."""
    store_8d = ScopedMemoryStore(base_dir=tmp_path / "dim8", embedder=MockEmbedder(dim=8))
    store_16d = ScopedMemoryStore(base_dir=tmp_path / "dim16", embedder=MockEmbedder(dim=16))

    store_8d.save("note", "Eight-dimensional observation on node A.", workspace="mismatch_ws")
    store_16d.save("note", "Sixteen-dimensional observation on node B.", workspace="mismatch_ws")

    # 1. Strict mode (default) raises DimensionMismatchError (a ValueError subclass)
    strict_engine = RAGSyncEngine(
        local_store=store_8d,
        peer=store_16d,
        transport=LocalFsTransport(peer=store_16d),
    )
    with pytest.raises(DimensionMismatchError, match="dim"):
        strict_engine.sync_workspace("mismatch_ws")

    # 2. Non-strict push_delta with wrong vector dim rejects the bad vector without writing
    non_strict_transport = LocalFsTransport(peer=store_8d, strict_dim=False)
    bad_doc = Doc(
        id="mem_bad_dim",
        text="Injected vector with wrong dimensionality.",
        category="note",
        vector=[0.1] * 16,
        dim=16,
    )
    push_res = non_strict_transport.push_delta("mismatch_ws", [bad_doc])
    assert push_res.ok is False
    assert push_res.accepted == 0
    assert push_res.rejected == 1
    assert len(store_8d.list(workspace="mismatch_ws")) == 1


def test_sync_all_multiple_workspaces(tmp_path: Path) -> None:
    """sync_all synchronizes multiple workspaces across local and peer stores."""
    emb = MockEmbedder(dim=16)
    store_a = ScopedMemoryStore(base_dir=tmp_path / "multi_a", embedder=emb)
    store_b = ScopedMemoryStore(base_dir=tmp_path / "multi_b", embedder=emb)

    store_a.save("code", "Auth service Ed25519 token verifier.", workspace="ws_auth")
    store_a.save("decision", "Billing webhook idempotency ledger.", workspace="ws_billing")
    store_b.save("bug", "Gateway rate limiter token bucket fix.", workspace="ws_gateway")

    engine = RAGSyncEngine(
        local_store=store_a,
        peer=store_b,
        transport=LocalFsTransport(peer=store_b),
        conflict_policy=LastWriteWins,
    )

    reports = engine.sync_all(["ws_auth", "ws_billing", "ws_gateway"])
    assert set(reports.keys()) == {"ws_auth", "ws_billing", "ws_gateway"}
    assert reports["ws_auth"].pushed == 1
    assert reports["ws_billing"].pushed == 1
    assert reports["ws_gateway"].pulled == 1

    for ws_name in ("ws_auth", "ws_billing", "ws_gateway"):
        assert len(store_a.list(workspace=ws_name)) == 1
        assert len(store_b.list(workspace=ws_name)) == 1

    # Also verify HttpTransport placeholder works offline with RFC 2606 endpoint
    http_transport = HttpTransport(endpoint_url="http://peer-node.example:9100/v1")
    http_push = http_transport.push_delta("ws_auth", [Doc(text="HTTP placeholder test")])
    assert http_push.ok is True
    assert http_transport.pull_delta("ws_auth", "") == []

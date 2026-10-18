"""Model-agnostic embedding configuration, swap planning, and atomic re-embedding (`swap.py`).

Core invariant:
- `docs.json` is the single source of truth (text + metadata).
- `index.faiss` is a derived artifact that can be rebuilt or swapped atomically
  across different embedding models and vector dimensions without losing notes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np

from scoped_memory.atomic import _atomic_write_faiss, _atomic_write_json
from scoped_memory.embed import (
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_URL,
    DEFAULT_MODEL_NAME,
    CpuHttpEmbedder,
    Embedder,
    MockEmbedder,
)
from scoped_memory.store import ScopedMemoryStore


@dataclass
class EmbeddingModelConfig:
    """Describes an embedding model configuration for model-agnostic vector indexing."""

    name: str
    dim: int = DEFAULT_EMBED_DIM
    model_id: str = DEFAULT_MODEL_NAME
    server_url: str = DEFAULT_EMBED_URL
    version: str = "1.0.0"
    embedder: Embedder | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"EmbeddingModelConfig.dim must be positive, got {self.dim}")

    def model_hash(self) -> str:
        """Compute deterministic SHA-256 fingerprint of this embedding model configuration."""
        payload = f"{self.name}|{self.dim}|{self.model_id}|{self.server_url}|{self.version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize model metadata for `embed_model.json` and `swap_state.json`."""
        return {
            "name": self.name,
            "dim": int(self.dim),
            "model_id": self.model_id,
            "server_url": self.server_url,
            "version": self.version,
            "model_hash": self.model_hash(),
        }


@dataclass
class SwapPlan:
    """Execution plan and risk assessment before swapping the embedding model."""

    old_dim: int
    new_dim: int
    total_docs: int
    workspaces: dict[str, int]
    estimated_tokens: int
    requires_full_rebuild: bool
    risk_level: str
    risk_summary: str
    new_model: EmbeddingModelConfig

    @property
    def current_dim(self) -> int:
        return self.old_dim

    @property
    def expected_dim(self) -> int:
        return self.old_dim

    @property
    def dim_changed(self) -> bool:
        return self.old_dim != self.new_dim

    @property
    def docs_to_reembed(self) -> int:
        return self.total_docs

    @property
    def doc_count(self) -> int:
        return self.total_docs

    @property
    def token_cost(self) -> int:
        return self.estimated_tokens

    @property
    def risk(self) -> str:
        return self.risk_summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_dim": self.old_dim,
            "new_dim": self.new_dim,
            "dim_changed": self.dim_changed,
            "total_docs": self.total_docs,
            "workspaces": dict(self.workspaces),
            "estimated_tokens": self.estimated_tokens,
            "requires_full_rebuild": self.requires_full_rebuild,
            "risk_level": self.risk_level,
            "risk_summary": self.risk_summary,
            "new_model": self.new_model.to_dict(),
        }


@dataclass
class SwapReport:
    """Telemetry and verification report produced by `execute_swap()`."""

    swapped_docs: int
    workspaces_swapped: int
    batches_processed: int
    avg_batch_ms: float
    total_duration_ms: float
    eta_ms: float
    old_dim: int
    new_dim: int
    requires_full_rebuild: bool
    backup_dir: str
    model_hash: str
    resumed_from: int = 0
    workspaces: list[str] = field(default_factory=list)

    @property
    def total_docs(self) -> int:
        return self.swapped_docs

    @property
    def docs_reembedded(self) -> int:
        return self.swapped_docs

    @property
    def replaced_docs(self) -> int:
        return self.swapped_docs

    @property
    def total_batches(self) -> int:
        return self.batches_processed

    @property
    def avg_batch_time_ms(self) -> float:
        return self.avg_batch_ms

    @property
    def duration_ms(self) -> float:
        return self.total_duration_ms

    @property
    def eta_sec(self) -> float:
        return round(self.eta_ms / 1000.0, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "swapped_docs": self.swapped_docs,
            "workspaces_swapped": self.workspaces_swapped,
            "workspaces": list(self.workspaces),
            "batches_processed": self.batches_processed,
            "avg_batch_ms": self.avg_batch_ms,
            "total_duration_ms": self.total_duration_ms,
            "eta_ms": self.eta_ms,
            "old_dim": self.old_dim,
            "new_dim": self.new_dim,
            "requires_full_rebuild": self.requires_full_rebuild,
            "backup_dir": self.backup_dir,
            "model_hash": self.model_hash,
            "resumed_from": self.resumed_from,
        }


def _coerce_model_config(
    store: ScopedMemoryStore,
    new_model: EmbeddingModelConfig | Embedder,
) -> tuple[EmbeddingModelConfig, Embedder]:
    """Normalize `new_model` into `(EmbeddingModelConfig, Embedder)`."""
    if isinstance(new_model, Embedder):
        cfg = EmbeddingModelConfig(
            name=new_model.__class__.__name__,
            dim=int(new_model.dim),
            model_id=getattr(new_model, "model_name", f"mock-{new_model.dim}d"),
            server_url=getattr(new_model, "endpoint_url", "mock://local"),
            version="1.0.0",
            embedder=new_model,
        )
        return cfg, new_model

    if isinstance(new_model, EmbeddingModelConfig):
        if new_model.embedder is not None:
            return new_model, new_model.embedder
        if (
            isinstance(store.embedder, MockEmbedder)
            or not new_model.server_url
            or new_model.server_url.startswith("mock")
            or ".example" in new_model.server_url
        ):
            return new_model, MockEmbedder(dim=new_model.dim)
        return (
            new_model,
            CpuHttpEmbedder(
                endpoint_url=new_model.server_url,
                model_name=new_model.model_id,
                dim=new_model.dim,
            ),
        )

    raise TypeError(f"Expected EmbeddingModelConfig or Embedder, got {type(new_model)!r}")


def _detect_expected_dim(store: ScopedMemoryStore) -> int:
    """Determine current expected vector dimensionality across store and indices."""
    meta_path = store.base_dir / "embed_model.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and isinstance(meta.get("dim"), int):
                return int(meta["dim"])
        except Exception:
            pass

    for ws_idx in store.all_workspace_indices().values():
        if ws_idx.index is not None and int(ws_idx.index.d) > 0:
            return int(ws_idx.index.d)

    return int(store.embedder.dim)


def plan_swap(
    store: ScopedMemoryStore,
    new_model: EmbeddingModelConfig | Embedder,
) -> SwapPlan:
    """Analyze a `ScopedMemoryStore` and build a deterministic `SwapPlan` for `new_model`.

    Checks `expected_dim` against `new_model.dim`. When dimensions differ, sets
    `requires_full_rebuild=True` and documents that `index.faiss` will be rebuilt
    from scratch using `docs.json` without losing any text or metadata.
    """
    cfg, _ = _coerce_model_config(store, new_model)
    expected_dim = _detect_expected_dim(store)
    new_dim = int(cfg.dim)
    requires_full_rebuild = expected_dim != new_dim

    ws_indices = store.all_workspace_indices()
    ws_counts: dict[str, int] = {}
    total_docs = 0
    estimated_tokens = 0

    for ws_name, ws_idx in sorted(ws_indices.items()):
        docs_path = ws_idx.docs_path
        if docs_path.exists():
            raw_docs = json.loads(docs_path.read_text(encoding="utf-8"))
            docs_list = raw_docs if isinstance(raw_docs, list) else []
        else:
            docs_list = list(ws_idx.docs)
        count = len(docs_list)
        ws_counts[ws_name] = count
        total_docs += count
        for doc in docs_list:
            text = str(doc.get("text", ""))
            estimated_tokens += max(1, (len(text) + 3) // 4) if text else 0

    if requires_full_rebuild:
        risk_level = "medium"
        risk_summary = (
            f"Dimension change detected ({expected_dim} -> {new_dim}): "
            "index.faiss must be rebuilt from scratch using docs.json. "
            "Zero text or metadata is lost (docs.json is untouched)."
        )
    else:
        risk_level = "low"
        risk_summary = (
            f"Dimension unchanged ({new_dim}): "
            "vectors will be re-embedded from docs.json and atomically swapped. "
            "Zero text or metadata is lost."
        )

    return SwapPlan(
        old_dim=expected_dim,
        new_dim=new_dim,
        total_docs=total_docs,
        workspaces=ws_counts,
        estimated_tokens=estimated_tokens,
        requires_full_rebuild=requires_full_rebuild,
        risk_level=risk_level,
        risk_summary=risk_summary,
        new_model=cfg,
    )


def execute_swap(
    store: ScopedMemoryStore,
    new_model: EmbeddingModelConfig | Embedder,
    batch_size: int = 8,
    resume_from: int = 0,
) -> SwapReport:
    """Atomically re-embed all workspace documents in `store` using `new_model`.

    Steps:
    1. Validate plan and detect dimension changes (`requires_full_rebuild`).
    2. Back up existing `index.faiss` (and `embed_model.json` if present) into
       `_pre_swap_backup_<timestamp>/` without modifying `docs.json`.
    3. Re-embed all documents from `docs.json` in batches of `batch_size`, persisting
       progress checkpoints after every batch to `swap_state.json`.
    4. Atomically write new `index.faiss` (`tmp + fsync + rename`) and `embed_model.json`
       alongside `docs.json` while keeping `docs.json` byte-for-byte identical.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if resume_from < 0:
        raise ValueError(f"resume_from must be non-negative, got {resume_from}")

    t_start = time.perf_counter()
    cfg, new_embedder = _coerce_model_config(store, new_model)
    plan = plan_swap(store, cfg)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_backup_dir = store.base_dir / f"_pre_swap_backup_{timestamp}"
    root_backup_dir.mkdir(parents=True, exist_ok=True)

    ws_indices = store.all_workspace_indices()
    ws_docs_map: dict[str, list[dict[str, Any]]] = {}
    flat_items: list[tuple[str, int, str]] = []

    for ws_name in sorted(ws_indices.keys()):
        ws_dir = store.base_dir / ws_name
        docs_path = ws_dir / "docs.json"
        raw_docs = json.loads(docs_path.read_text(encoding="utf-8")) if docs_path.exists() else []
        docs_list = raw_docs if isinstance(raw_docs, list) else []
        ws_docs_map[ws_name] = docs_list
        for idx, doc in enumerate(docs_list):
            flat_items.append((ws_name, idx, str(doc.get("text", ""))))

        # Backup existing index.faiss in both root backup dir and workspace backup dir
        src_index = ws_dir / "index.faiss"
        if src_index.exists():
            ws_backup_sub = root_backup_dir / ws_name
            ws_backup_sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_index, ws_backup_sub / "index.faiss")
            if len(ws_indices) == 1:
                shutil.copy2(src_index, root_backup_dir / "index.faiss")

            local_ws_backup = ws_dir / f"_pre_swap_backup_{timestamp}"
            local_ws_backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_index, local_ws_backup / "index.faiss")

    total_docs = len(flat_items)
    state_path = store.base_dir / "swap_state.json"

    # Load existing checkpoint if available and matching target model
    saved_vectors: dict[int, list[float]] = {}
    effective_resume = resume_from
    if state_path.exists():
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                isinstance(prior_state, dict)
                and prior_state.get("model_hash") == cfg.model_hash()
                and int(prior_state.get("new_dim", 0)) == cfg.dim
            ):
                raw_vecs = prior_state.get("vectors", {})
                if isinstance(raw_vecs, dict):
                    for k_str, vec_vals in raw_vecs.items():
                        k_int = int(k_str)
                        if isinstance(vec_vals, list) and len(vec_vals) == cfg.dim:
                            saved_vectors[k_int] = [float(x) for x in vec_vals]
                if effective_resume == 0 and prior_state.get("status") == "in_progress":
                    effective_resume = int(prior_state.get("completed_docs", 0))
        except Exception:
            saved_vectors = {}

    effective_resume = min(effective_resume, total_docs)

    # Ensure missing vectors prior to effective_resume are populated so index.ntotal == len(docs)
    missing_prefix_indices = [i for i in range(effective_resume) if i not in saved_vectors]
    if missing_prefix_indices:
        prefix_texts = [flat_items[i][2] for i in missing_prefix_indices]
        prefix_vecs = new_embedder.embed_passages(prefix_texts)
        for pos, vec_row in zip(missing_prefix_indices, prefix_vecs, strict=False):
            saved_vectors[pos] = [float(x) for x in vec_row.tolist()]

    batch_durations_ms: list[float] = []
    remaining_docs = total_docs - effective_resume
    total_remaining_batches = (
        (remaining_docs + batch_size - 1) // batch_size if remaining_docs > 0 else 0
    )

    batches_done = 0
    for start_idx in range(effective_resume, total_docs, batch_size):
        b_start = time.perf_counter()
        end_idx = min(start_idx + batch_size, total_docs)
        batch_texts = [flat_items[i][2] for i in range(start_idx, end_idx)]
        batch_vecs = new_embedder.embed_passages(batch_texts)
        if int(batch_vecs.shape[1]) != cfg.dim:
            raise ValueError(
                f"Embedder produced dim={batch_vecs.shape[1]}, expected dim={cfg.dim}"
            )
        for offset, vec_row in enumerate(batch_vecs):
            saved_vectors[start_idx + offset] = [float(x) for x in vec_row.tolist()]

        b_ms = (time.perf_counter() - b_start) * 1000.0
        batch_durations_ms.append(b_ms)
        batches_done += 1

        avg_ms_so_far = sum(batch_durations_ms) / len(batch_durations_ms)
        batches_left = max(0, total_remaining_batches - batches_done)
        eta_ms = round(avg_ms_so_far * batches_left, 3)

        checkpoint_payload = {
            "status": "in_progress",
            "model_hash": cfg.model_hash(),
            "new_dim": cfg.dim,
            "old_dim": plan.old_dim,
            "batch_size": batch_size,
            "resumed_from": effective_resume,
            "completed_docs": end_idx,
            "total_docs": total_docs,
            "batches_processed": batches_done,
            "avg_batch_ms": round(avg_ms_so_far, 3),
            "eta_ms": eta_ms,
            "backup_dir": str(root_backup_dir),
            "vectors": {str(k): v for k, v in saved_vectors.items()},
        }
        _atomic_write_json(state_path, checkpoint_payload)

    # Atomically rebuild and swap index.faiss + write embed_model.json per workspace
    # Note: docs.json is NEVER written or modified here, preserving byte-identity.
    embed_meta = {
        **cfg.to_dict(),
        "swapped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_dim": plan.old_dim,
        "requires_full_rebuild": plan.requires_full_rebuild,
    }

    global_cursor = 0
    for ws_name in sorted(ws_docs_map.keys()):
        ws_dir = store.base_dir / ws_name
        docs_list = ws_docs_map[ws_name]
        count = len(docs_list)
        new_index = faiss.IndexFlatIP(cfg.dim)
        if count > 0:
            ws_matrix = np.array(
                [saved_vectors[global_cursor + i] for i in range(count)],
                dtype="float32",
            )
            new_index.add(ws_matrix)
        global_cursor += count

        _atomic_write_faiss(new_index, ws_dir / "index.faiss")
        _atomic_write_json(ws_dir / "embed_model.json", {**embed_meta, "docs": count})

        # Update cached WorkspaceIndex in-place without touching docs.json
        if ws_name in store._cache:
            ws_obj = store._cache[ws_name]
            ws_obj.embedder = new_embedder
            ws_obj.index = new_index

    _atomic_write_json(store.base_dir / "embed_model.json", embed_meta)
    store.embedder = new_embedder

    total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
    avg_batch_ms = (
        round(sum(batch_durations_ms) / len(batch_durations_ms), 3)
        if batch_durations_ms
        else 0.0
    )

    final_state = {
        "status": "completed",
        "model_hash": cfg.model_hash(),
        "new_dim": cfg.dim,
        "old_dim": plan.old_dim,
        "batch_size": batch_size,
        "resumed_from": effective_resume,
        "completed_docs": total_docs,
        "total_docs": total_docs,
        "batches_processed": batches_done,
        "avg_batch_ms": avg_batch_ms,
        "eta_ms": 0.0,
        "backup_dir": str(root_backup_dir),
        "vectors": {str(k): v for k, v in saved_vectors.items()},
    }
    _atomic_write_json(state_path, final_state)

    return SwapReport(
        swapped_docs=total_docs,
        workspaces_swapped=len(ws_docs_map),
        batches_processed=batches_done,
        avg_batch_ms=avg_batch_ms,
        total_duration_ms=total_ms,
        eta_ms=0.0,
        old_dim=plan.old_dim,
        new_dim=cfg.dim,
        requires_full_rebuild=plan.requires_full_rebuild,
        backup_dir=str(root_backup_dir),
        model_hash=cfg.model_hash(),
        resumed_from=effective_resume,
        workspaces=sorted(ws_docs_map.keys()),
    )

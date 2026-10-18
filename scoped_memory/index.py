"""Per-workspace FAISS `IndexFlatIP` and `docs.json` branch index management."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from scoped_memory.atomic import _atomic_write_faiss, _atomic_write_json
from scoped_memory.embed import Embedder
from scoped_memory.sanitize import normalize_text, text_hash

ALLOWED_CATEGORIES: set[str] = {
    "code",
    "bug",
    "decision",
    "assumption",
    "note",
    "skill",
    "wiki",
}

CATEGORY_TO_LAYER: dict[str, str] = {
    "note": "L0",
    "assumption": "L1",
    "bug": "L1",
    "code": "L2",
    "wiki": "L2",
    "decision": "L3",
    "skill": "L3",
}

CATEGORY_TO_PILLAR: dict[str, str] = {
    "note": "chat",
    "assumption": "chat",
    "decision": "skill",
    "skill": "skill",
    "wiki": "wiki",
    "code": "code",
    "bug": "code",
}


class WorkspaceIndex:
    """Manages an isolated namespace branch (`<base_dir>/<workspace>/index.faiss` + `docs.json`).

    Invariants:
    - `docs.json` is the single source of truth for text and metadata.
    - `index.faiss` is a positionally aligned `IndexFlatIP` (`index.ntotal == len(docs)`).
    - Optimistic staleness detection reloads from disk whenever `(mtime, size)` changes.
    """

    def __init__(self, store_dir: Path | str, embedder: Embedder) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.docs_path = self.store_dir / "docs.json"
        self.embedder = embedder
        self.docs: list[dict[str, Any]] = []
        self.index: faiss.Index | None = None
        self._mtime: float = 0.0
        self._sig: tuple[float, int] | None = None
        self._load()

    @property
    def name(self) -> str:
        return self.store_dir.name

    @property
    def is_empty(self) -> bool:
        return len(self.docs) == 0 or self.index is None or int(self.index.ntotal) == 0

    def _load(self) -> None:
        if self.docs_path.exists():
            try:
                raw = json.loads(self.docs_path.read_text(encoding="utf-8"))
                self.docs = raw if isinstance(raw, list) else []
                st = self.docs_path.stat()
                self._mtime = st.st_mtime
                self._sig = (st.st_mtime, st.st_size)
            except Exception:
                self.docs = []
                self._mtime = 0.0
                self._sig = None
        else:
            self.docs = []
            self._mtime = 0.0
            self._sig = None

        if self.index_path.exists():
            try:
                loaded = faiss.read_index(str(self.index_path))
                self.index = loaded
            except Exception:
                self.index = None
        else:
            self.index = None

        if self.docs and (self.index is None or int(self.index.ntotal) != len(self.docs)):
            try:
                vecs = self.embedder.embed_passages([d.get("text", "") for d in self.docs])
                idx = faiss.IndexFlatIP(int(vecs.shape[1]))
                idx.add(vecs.astype("float32"))
                self.index = idx
            except Exception:
                pass

    def _check_stale_and_reload(self) -> bool:
        """Reload in-memory state if `docs.json` changed on disk (`mtime` or byte size)."""
        if not self.docs_path.exists():
            return False
        try:
            st = self.docs_path.stat()
            cur_sig = (st.st_mtime, st.st_size)
            if self._sig is None:
                self._sig = cur_sig
            if st.st_mtime != self._mtime or cur_sig != self._sig:
                self._load()
                return True
        except OSError:
            pass
        return False

    def _persist(self) -> None:
        """Atomically persist `docs.json` and `index.faiss`."""
        _atomic_write_json(self.docs_path, self.docs)
        try:
            st = self.docs_path.stat()
            self._mtime = st.st_mtime
            self._sig = (st.st_mtime, st.st_size)
        except OSError:
            pass
        if self.index is not None:
            _atomic_write_faiss(self.index, self.index_path)

    @staticmethod
    def extract_tags(
        text: str,
        source: str = "",
        explicit_tags: Sequence[str] | str | None = None,
    ) -> list[str]:
        tags: list[str] = []
        if explicit_tags:
            if isinstance(explicit_tags, str):
                for t in re.split(r"[,;\s]+", explicit_tags):
                    cleaned = t.strip().lstrip("#").lower()
                    if cleaned and cleaned not in tags:
                        tags.append(cleaned)
            else:
                for t in explicit_tags:
                    cleaned = str(t).strip().lstrip("#").lower()
                    if cleaned and cleaned not in tags:
                        tags.append(cleaned)
        for m in re.findall(r"(?:^|\s)#([a-zA-Z0-9_\-]{2,28})", text or ""):
            t = m.lower()
            if t not in tags:
                tags.append(t)
        for m in re.findall(r"\[([a-zA-Z0-9_\-]{2,24})\]", source or ""):
            t = m.lower()
            if t not in tags:
                tags.append(t)
        return tags[:10]

    def find_duplicate_by_hash(self, normalized_text: str) -> dict[str, Any] | None:
        """Return existing document if its SHA-256 content hash matches `normalized_text`."""
        target_hash = text_hash(normalized_text)
        for doc in self.docs:
            if text_hash(doc.get("text", "")) == target_hash:
                return doc
        return None

    def append_document(
        self,
        category: str,
        normalized_text: str,
        vec: np.ndarray,
        source: str = "manual",
        tags: Sequence[str] | str | None = None,
        near_dup_threshold: float = 0.955,
    ) -> tuple[str, dict[str, Any]]:
        """Append a single embedded document or return existing duplicate/near-duplicate."""
        cat = (category or "note").lower()
        if cat not in ALLOWED_CATEGORIES:
            cat = "note"

        vec_2d = np.asarray(vec, dtype="float32").reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatIP(int(vec_2d.shape[1]))
        elif int(self.index.ntotal) > 0:
            scores, positions = self.index.search(vec_2d, 1)
            top_sim = float(scores[0][0])
            top_idx = int(positions[0][0])
            if top_sim >= near_dup_threshold and 0 <= top_idx < len(self.docs):
                return "near_duplicate", {
                    **self.docs[top_idx],
                    "similarity": round(top_sim, 4),
                }

        self.index.add(vec_2d)
        doc: dict[str, Any] = {
            "id": "mem_" + uuid.uuid4().hex[:12],
            "text": normalized_text,
            "source": source or "manual",
            "category": cat,
            "layer": CATEGORY_TO_LAYER.get(cat, "L0"),
            "pillar": CATEGORY_TO_PILLAR.get(cat, "chat"),
            "tags": self.extract_tags(normalized_text, source, tags),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.docs.append(doc)
        self._persist()
        return "saved", doc

    def add_many(self, items: Sequence[tuple[str, str, str]]) -> int:
        """Batch-embed and append multiple `(category, text, source)` tuples with hash dedup."""
        self._check_stale_and_reload()
        raw = [(c, normalize_text(t), s) for (c, t, s) in items if t and normalize_text(t)]
        if not raw:
            return 0

        seen = {text_hash(d.get("text", "")) for d in self.docs}
        deduped: list[tuple[str, str, str]] = []
        for cat, norm_txt, src in raw:
            h = text_hash(norm_txt)
            if h in seen:
                continue
            seen.add(h)
            deduped.append((cat, norm_txt, src))
        if not deduped:
            return 0

        vecs = self.embedder.embed_passages([t for (_, t, _) in deduped])
        if self.index is None:
            self.index = faiss.IndexFlatIP(int(vecs.shape[1]))
        self.index.add(vecs.astype("float32"))

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for cat, norm_txt, src in deduped:
            c = (cat or "note").lower()
            if c not in ALLOWED_CATEGORIES:
                c = "note"
            self.docs.append(
                {
                    "id": "mem_" + uuid.uuid4().hex[:12],
                    "text": norm_txt,
                    "source": src or "seed",
                    "category": c,
                    "layer": CATEGORY_TO_LAYER.get(c, "L0"),
                    "pillar": CATEGORY_TO_PILLAR.get(c, "chat"),
                    "tags": self.extract_tags(norm_txt, src or "seed"),
                    "time": now,
                }
            )
        self._persist()
        return len(deduped)

    def search_raw(
        self,
        query: str,
        top_k: int = 5,
        category: str = "",
        min_score: float = 0.0,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Run a vector similarity query against this workspace branch."""
        self._check_stale_and_reload()
        if self.is_empty or self.index is None:
            return []

        qvec = self.embedder.embed_query(query)
        qvec_2d = np.asarray(qvec, dtype="float32").reshape(1, -1)
        k = min(max(1, top_k * 3 if category else top_k), int(self.index.ntotal))
        scores, positions = self.index.search(qvec_2d, k)

        cat_filter = category.strip().lower()
        results: list[tuple[float, dict[str, Any]]] = []
        for score, pos in zip(scores[0], positions[0], strict=False):
            idx = int(pos)
            if idx < 0 or idx >= len(self.docs):
                continue
            sim = float(score)
            if sim < min_score:
                continue
            doc = self.docs[idx]
            if cat_filter and doc.get("category", "").lower() != cat_filter:
                continue
            results.append((sim, doc))
            if len(results) >= top_k:
                break
        return results

    def delete_doc(self, doc_id: str) -> dict[str, Any]:
        """Remove a document by ID and reconstruct the FAISS index without re-embedding."""
        self._check_stale_and_reload()
        target_idx = -1
        for i, doc in enumerate(self.docs):
            if doc.get("id") == doc_id:
                target_idx = i
                break
        if target_idx < 0:
            return {"ok": False, "error": f"Document id={doc_id} not found in {self.name}"}

        removed_doc = self.docs.pop(target_idx)
        if self.index is not None and int(self.index.ntotal) == len(self.docs) + 1:
            dim = int(self.index.d)
            new_idx = faiss.IndexFlatIP(dim)
            if self.docs:
                kept = [
                    self.index.reconstruct(i)
                    for i in range(int(self.index.ntotal))
                    if i != target_idx
                ]
                new_idx.add(np.array(kept, dtype="float32"))
            self.index = new_idx
        elif self.docs:
            vecs = self.embedder.embed_passages([d.get("text", "") for d in self.docs])
            new_idx = faiss.IndexFlatIP(int(vecs.shape[1]))
            new_idx.add(vecs.astype("float32"))
            self.index = new_idx
        else:
            self.index = faiss.IndexFlatIP(self.embedder.dim)

        self._persist()
        return {"ok": True, "deleted": removed_doc, "remaining": len(self.docs)}

    def update_doc(
        self,
        doc_id: str,
        text: str | None = None,
        category: str | None = None,
        source: str | None = None,
        tags: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        """Update document text or metadata and refresh its vector in `index.faiss`."""
        self._check_stale_and_reload()
        target_idx = -1
        for i, doc in enumerate(self.docs):
            if doc.get("id") == doc_id:
                target_idx = i
                break
        if target_idx < 0:
            return {"ok": False, "error": f"Document id={doc_id} not found in {self.name}"}

        doc = self.docs[target_idx]
        old_text = str(doc.get("text", ""))
        new_text = normalize_text(text) if text is not None else old_text
        if not new_text:
            return {"ok": False, "error": "Document text cannot be empty"}

        new_cat = (category or str(doc.get("category", "note"))).lower()
        if new_cat not in ALLOWED_CATEGORIES:
            new_cat = "note"
        new_src = source if source is not None else str(doc.get("source", "manual"))
        new_tags = self.extract_tags(
            new_text,
            new_src,
            tags if tags is not None else doc.get("tags"),
        )

        if new_text != old_text:
            new_vec = self.embedder.embed_passages([new_text])[0]
            if self.index is not None and int(self.index.ntotal) == len(self.docs):
                dim = int(self.index.d)
                new_idx = faiss.IndexFlatIP(dim)
                vecs = [
                    new_vec if i == target_idx else self.index.reconstruct(i)
                    for i in range(len(self.docs))
                ]
                new_idx.add(np.array(vecs, dtype="float32"))
                self.index = new_idx

        doc["text"] = new_text
        doc["category"] = new_cat
        doc["source"] = new_src
        doc["layer"] = CATEGORY_TO_LAYER.get(new_cat, "L0")
        doc["pillar"] = CATEGORY_TO_PILLAR.get(new_cat, "chat")
        doc["tags"] = new_tags
        doc["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._persist()
        return {"ok": True, "doc": doc}

    def nearest_neighbors(
        self, doc_id: str, top_k: int = 6
    ) -> list[tuple[float, dict[str, Any]]]:
        """Return top-K nearest neighbors using the stored FAISS vector without re-embedding."""
        self._check_stale_and_reload()
        if self.is_empty or self.index is None:
            return []
        target_idx = -1
        for i, doc in enumerate(self.docs):
            if doc.get("id") == doc_id:
                target_idx = i
                break
        if target_idx < 0 or target_idx >= int(self.index.ntotal):
            return []

        qvec = self.index.reconstruct(target_idx)
        k = min(top_k + 1, int(self.index.ntotal))
        scores, positions = self.index.search(np.array([qvec], dtype="float32"), k)
        out: list[tuple[float, dict[str, Any]]] = []
        for score, pos in zip(scores[0], positions[0], strict=False):
            idx = int(pos)
            if idx < 0 or idx >= len(self.docs) or idx == target_idx:
                continue
            out.append((float(score), self.docs[idx]))
        return out[:top_k]

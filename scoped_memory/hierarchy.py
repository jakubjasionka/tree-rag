"""Two-tier hierarchical retrieval (`workspace` -> `global`) and Human-in-the-Loop escalation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from scoped_memory.index import WorkspaceIndex


@dataclass(frozen=True)
class SearchHit:
    """Single semantic retrieval hit with provenance and workspace label."""

    score: float
    doc: dict[str, Any]
    workspace: str

    @property
    def label(self) -> str:
        """Formatted workspace label (`[workspace=<name>]`)."""
        return f"[workspace={self.workspace}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "workspace": self.workspace,
            "label": self.label,
            **self.doc,
        }


@dataclass(frozen=True)
class EmptyContext:
    """Deterministic zero-pollution result returned when a workspace branch has no qualifying hits.

    Evaluates to `False` in boolean contexts (`bool(ctx) is False`), has `len(ctx) == 0`,
    injects 0 noise tokens, and signals via `requires_confirmation()` whether the autonomous
    agent must request Human-in-the-Loop approval before escalating to `scope="global"`.
    """

    workspace: str
    query: str
    scope: str = "workspace"
    reason: str = "empty_workspace"
    min_score: float = 0.25
    top_local_score: float = 0.0
    escalation_required: bool = True
    hits: tuple[SearchHit, ...] = ()

    @property
    def is_empty(self) -> bool:
        return True

    def requires_confirmation(self) -> bool:
        """Return `True` when local scope lacked hits above threshold and needs HITL."""
        return self.escalation_required

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[SearchHit]:
        return iter(())

    def format(self) -> str:
        if self.reason == "hitl_confirmation_required":
            return (
                f"[EmptyContext workspace={self.workspace}] Escalation to scope='global' "
                f"requires Human-in-the-Loop confirmation (pass confirmed=True)."
            )
        return (
            f"[EmptyContext workspace={self.workspace}] 0 hits above threshold "
            f"(min_score={self.min_score:.2f}, reason={self.reason}). "
            f"requires_confirmation={self.requires_confirmation()}"
        )


@dataclass(frozen=True)
class SearchResult:
    """Non-empty semantic search result from either a local workspace branch or global tree."""

    workspace: str
    query: str
    scope: str
    hits: list[SearchHit] = field(default_factory=list)
    min_score: float = 0.25

    @property
    def is_empty(self) -> bool:
        return len(self.hits) == 0

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    def requires_confirmation(self) -> bool:
        """Return `False` when local hits satisfy the similarity threshold."""
        if self.scope == "global":
            return False
        return len(self.hits) == 0 or self.top_score < self.min_score

    def __bool__(self) -> bool:
        return len(self.hits) > 0

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self) -> Iterator[SearchHit]:
        return iter(self.hits)

    def __getitem__(self, index: int) -> SearchHit:
        return self.hits[index]

    def format(self) -> str:
        if not self.hits:
            return f"memory_search ({self.scope}): 0 hits."
        nl = chr(10)
        header = (
            f'Found {len(self.hits)} hits (global, all workspaces) for: "{self.query}"{nl}'
            if self.scope == "global"
            else f'Found {len(self.hits)} hits in [{self.workspace}] for: "{self.query}"{nl}'
        )
        blocks = [header]
        for idx, hit in enumerate(self.hits, start=1):
            txt = str(hit.doc.get("text", ""))
            if len(txt) > 800:
                txt = txt[:800] + " [...]"
            cat = hit.doc.get("category", "note")
            src = hit.doc.get("source", "manual")
            ws_tag = f" {hit.label}" if self.scope == "global" else ""
            blocks.append(
                f"### #{idx} (score={hit.score:.3f}){ws_tag} [{cat}] src={src}{nl}{txt}"
            )
        sep = f"{nl}{nl}---{nl}{nl}"
        return sep.join(blocks)


def requires_confirmation(
    result_or_hits: (
        EmptyContext
        | SearchResult
        | list[SearchHit]
        | list[tuple[float, dict[str, Any]]]
        | None
    ),
    min_score: float = 0.25,
) -> bool:
    """Evaluate whether a local workspace query result requires HITL global confirmation.

    Returns `True` when:
    - The workspace returned `None` or an `EmptyContext`.
    - The hit list is empty.
    - The highest similarity score in the local workspace is strictly below `min_score`.
    """
    if result_or_hits is None:
        return True
    if isinstance(result_or_hits, EmptyContext):
        return result_or_hits.requires_confirmation()
    if isinstance(result_or_hits, SearchResult):
        if result_or_hits.is_empty:
            return True
        return result_or_hits.top_score < min_score
    if len(result_or_hits) == 0:
        return True
    first = result_or_hits[0]
    if isinstance(first, SearchHit):
        return first.score < min_score
    if isinstance(first, tuple) and len(first) >= 1:
        return float(first[0]) < min_score
    return True


class HierarchyEngine:
    """Executes two-tier (`workspace` -> `global`) search with Human-in-the-Loop escalation."""

    def __init__(
        self,
        min_score: float = 0.25,
        require_global_confirmation: bool = False,
        confirmation_callback: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.min_score = min_score
        self.require_global_confirmation = require_global_confirmation
        self.confirmation_callback = confirmation_callback

    def search_workspace(
        self,
        ws_index: WorkspaceIndex,
        query: str,
        top_k: int = 5,
        category: str = "",
        min_score: float | None = None,
    ) -> SearchResult | EmptyContext:
        """Search a single workspace branch; return `EmptyContext` on zero or sub-threshold hits."""
        threshold = self.min_score if min_score is None else min_score
        ws_index._check_stale_and_reload()
        if ws_index.is_empty:
            return EmptyContext(
                workspace=ws_index.name,
                query=query,
                scope="workspace",
                reason="empty_workspace",
                min_score=threshold,
                top_local_score=0.0,
                escalation_required=True,
            )

        raw_all = ws_index.search_raw(query=query, top_k=top_k, category=category, min_score=0.0)
        top_raw = raw_all[0][0] if raw_all else 0.0
        qualified = [
            SearchHit(score=score, doc=dict(doc), workspace=ws_index.name)
            for score, doc in raw_all
            if score >= threshold
        ]
        if not qualified:
            return EmptyContext(
                workspace=ws_index.name,
                query=query,
                scope="workspace",
                reason="below_similarity_threshold" if raw_all else "no_matching_category",
                min_score=threshold,
                top_local_score=top_raw,
                escalation_required=True,
            )

        return SearchResult(
            workspace=ws_index.name,
            query=query,
            scope="workspace",
            hits=qualified[:top_k],
            min_score=threshold,
        )

    def search_global(
        self,
        stores: dict[str, WorkspaceIndex],
        query: str,
        top_k: int = 5,
        category: str = "",
        min_score: float | None = None,
        origin_workspace: str = "global",
        confirmed: bool | None = None,
    ) -> SearchResult | EmptyContext:
        """Search across all workspace branches, tagging every result with `[workspace=<name>]`.

        If Human-in-the-Loop confirmation is enforced (`require_global_confirmation=True` or
        `confirmed=False`), blocks global traversal and returns an `EmptyContext` with
        `reason="hitl_confirmation_required"` and `requires_confirmation() == True` unless
        approved via `confirmed=True` or `confirmation_callback`.
        """
        threshold = self.min_score if min_score is None else min_score
        needs_gate = self.require_global_confirmation if confirmed is None else (not confirmed)
        if confirmed is True:
            needs_gate = False

        if needs_gate:
            approved = False
            if self.confirmation_callback is not None:
                approved = bool(self.confirmation_callback(origin_workspace, query))
            if not approved:
                return EmptyContext(
                    workspace=origin_workspace,
                    query=query,
                    scope="global",
                    reason="hitl_confirmation_required",
                    min_score=threshold,
                    top_local_score=0.0,
                    escalation_required=True,
                )

        merged: list[SearchHit] = []
        for ws_name, ws_index in sorted(stores.items()):
            raw_hits = ws_index.search_raw(
                query=query,
                top_k=top_k,
                category=category,
                min_score=threshold,
            )
            for score, doc in raw_hits:
                tagged_doc = dict(doc)
                tagged_doc["workspace"] = ws_name
                tagged_doc["workspace_label"] = f"[workspace={ws_name}]"
                merged.append(SearchHit(score=score, doc=tagged_doc, workspace=ws_name))

        merged.sort(key=lambda h: h.score, reverse=True)
        selected = merged[:top_k]
        if not selected:
            return EmptyContext(
                workspace="global",
                query=query,
                scope="global",
                reason="no_global_hits",
                min_score=threshold,
                top_local_score=0.0,
                escalation_required=False,
            )

        return SearchResult(
            workspace="global",
            query=query,
            scope="global",
            hits=selected,
            min_score=threshold,
        )

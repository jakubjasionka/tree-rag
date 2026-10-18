#!/usr/bin/env python3
"""Interactive / deterministic demonstration of Tree-RAG's Zero-Pollution & HITL Escalation.

Demonstrates:
1. Populating `workspace="payments_service"` with architecture decisions.
2. Querying an empty workspace (`workspace="auth_service"`) -> returns deterministic
   `EmptyContext` (0 noise tokens injected, `bool(ctx) is False`).
3. Checking `ctx.requires_confirmation()` -> signals `True` for Human-in-the-Loop escalation.
4. Attempting unconfirmed global search (`confirmed=False`) -> blocked by HITL gate.
5. Executing confirmed global search (`confirmed=True`) -> retrieves the cross-workspace
   entry clearly tagged with `[workspace=payments_service]`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scoped_memory import EmptyContext, MockEmbedder, ScopedMemoryStore


def run_demo() -> None:
    with tempfile.TemporaryDirectory(prefix="treerag_demo_") as tmp:
        store = ScopedMemoryStore(
            base_dir=Path(tmp),
            default_workspace="auth_service",
            embedder=MockEmbedder(dim=384),
            min_score=0.25,
            require_global_confirmation=True,
        )

        # 1. Seed knowledge in a sibling workspace (`payments_service`)
        store.save(
            category="decision",
            text="Idempotency keys for Stripe webhook retries are stored in Redis with a 24h TTL.",
            source="adr-004.md",
            workspace="payments_service",
        )
        store.save(
            category="bug",
            text="Fixed PostgreSQL connection pool exhaustion under concurrent webhook bursts.",
            source="pr-142",
            workspace="payments_service",
        )

        query = "Where are Stripe webhook idempotency keys stored and what is the TTL?"
        print(f"Query: {query!r}")
        print("-" * 72)

        # 2. Search in the active, empty workspace (`auth_service`)
        local_result = store.search(query, scope="workspace", workspace="auth_service")
        assert isinstance(local_result, EmptyContext)
        assert not local_result
        print(f"Step 1 (Local Workspace 'auth_service'): {local_result.format()}")
        print(
            f"  -> Zero-Pollution Guarantee: "
            f"len(hits)={len(local_result)}, bool={bool(local_result)}"
        )
        print(
            f"  -> HITL Escalation Signal: "
            f"requires_confirmation()={local_result.requires_confirmation()}"
        )

        # 3. Attempt global escalation WITHOUT confirmation
        blocked_global = store.search(
            query,
            scope="global",
            workspace="auth_service",
            confirmed=False,
        )
        assert isinstance(blocked_global, EmptyContext)
        print()
        print(f"Step 2 (Unconfirmed Global Search): {blocked_global.format()}")

        # 4. Escalate to global scope WITH Human-in-the-Loop confirmation
        confirmed_global = store.search(
            query,
            scope="global",
            workspace="auth_service",
            confirmed=True,
        )
        print()
        print("Step 3 (Confirmed Global Escalation):")
        print(confirmed_global.format())


if __name__ == "__main__":
    run_demo()

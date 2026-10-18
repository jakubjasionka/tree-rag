#!/usr/bin/env python3
"""CLI demonstration of standalone delta-based RAG synchronization (`RAGSyncEngine`).

Usage:
    python examples/sync_demo.py --base-a /tmp/sync-a --base-b /tmp/sync-b --direction bidirectional
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scoped_memory import (  # noqa: E402
    ConflictPolicy,
    LocalFsTransport,
    MockEmbedder,
    RAGSyncEngine,
    ScopedMemoryStore,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tree-RAG Universal Standalone Sync Demo (RAGSyncEngine)"
    )
    parser.add_argument(
        "--base-a",
        default="/tmp/sync-a",
        help="Base directory for local store A",
    )
    parser.add_argument(
        "--base-b",
        default="/tmp/sync-b",
        help="Base directory for peer store B",
    )
    parser.add_argument(
        "--direction",
        choices=["bidirectional", "push", "pull", "one_way"],
        default="bidirectional",
        help="Synchronization direction (default: bidirectional)",
    )
    parser.add_argument(
        "--policy",
        choices=["union", "first", "last"],
        default="union",
        help="Conflict resolution policy (union|first|last)",
    )
    parser.add_argument(
        "--workspace",
        default="shared_engineering",
        help="Workspace name to synchronize",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedder = MockEmbedder(dim=16)

    store_a = ScopedMemoryStore(
        base_dir=Path(args.base_a),
        default_workspace=args.workspace,
        embedder=embedder,
    )
    store_b = ScopedMemoryStore(
        base_dir=Path(args.base_b),
        default_workspace=args.workspace,
        embedder=embedder,
    )

    store_a.save(
        category="decision",
        text="Node A decision: delta sync transfers only documents modified since last_sync_hash.",
        source="node-a",
        workspace=args.workspace,
    )
    store_b.save(
        category="code",
        text="Node B note: every push and pull scrubs secrets and validates vector dimensionality.",
        source="node-b",
        workspace=args.workspace,
    )

    transport = LocalFsTransport(peer=store_b, conflict_policy=args.policy)
    engine = RAGSyncEngine(
        local_store=store_a,
        peer=store_b,
        transport=transport,
        conflict_policy=ConflictPolicy(
            {
                "union": "union_merge",
                "first": "first_write_wins",
                "last": "last_write_wins",
            }[args.policy]
        ),
    )

    report = engine.sync_workspace(args.workspace, direction=args.direction)
    print("=== Sync Report ===")
    print(json.dumps(report.to_dict(), indent=2))
    print(f"Store A docs count: {len(store_a.list(workspace=args.workspace))}")
    print(f"Store B docs count: {len(store_b.list(workspace=args.workspace))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

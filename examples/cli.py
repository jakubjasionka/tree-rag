#!/usr/bin/env python3
"""Command-line demo for `ScopedMemoryStore` (save / search / list / merge-pending / health)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scoped_memory import MockEmbedder, ScopedMemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tree-RAG CLI — Hierarchical Scoped Memory Engine"
    )
    parser.add_argument(
        "--base-dir",
        default="./.treerag_data",
        help="Base storage directory for workspace sub-indices",
    )
    parser.add_argument(
        "--workspace",
        default="default",
        help="Target workspace branch name",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_save = sub.add_parser("save", help="Save an entry into a workspace branch")
    p_save.add_argument(
        "--category",
        default="note",
        help="Category (code|bug|decision|assumption|note|skill|wiki)",
    )
    p_save.add_argument("--source", default="cli", help="Provenance source label")
    p_save.add_argument("text", help="Observation text to persist")

    p_search = sub.add_parser("search", help="Search within a workspace or globally")
    p_search.add_argument("--scope", choices=["workspace", "global"], default="workspace")
    p_search.add_argument("--category", default="", help="Optional category filter")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument(
        "--confirm-global",
        action="store_true",
        help="Approve HITL global escalation",
    )
    p_search.add_argument("query", help="Natural language search query")

    p_list = sub.add_parser("list", help="List recent entries")
    p_list.add_argument("--scope", choices=["workspace", "global"], default="workspace")
    p_list.add_argument("--category", default="")
    p_list.add_argument("--limit", type=int, default=20)

    sub.add_parser("merge-pending", help="Merge queued _pending/*.md files into FAISS")
    sub.add_parser("health", help="Print diagnostic health JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ScopedMemoryStore(
        base_dir=Path(args.base_dir),
        default_workspace=args.workspace,
        embedder=MockEmbedder(dim=384),
        require_global_confirmation=True,
    )

    if args.command == "save":
        res = store.save(
            category=args.category,
            text=args.text,
            source=args.source,
            workspace=args.workspace,
        )
        print(res["message"])
        return 0

    if args.command == "search":
        result = store.search(
            query=args.query,
            top_k=args.top_k,
            category=args.category,
            scope=args.scope,
            workspace=args.workspace,
            confirmed=args.confirm_global if args.scope == "global" else None,
        )
        print(result.format())
        return 0

    if args.command == "list":
        print(
            store.list_formatted(
                category=args.category,
                limit=args.limit,
                scope=args.scope,
                workspace=args.workspace,
            )
        )
        return 0

    if args.command == "merge-pending":
        merged = store.merge_pending(workspace=args.workspace)
        print(json.dumps(merged, indent=2))
        return 0

    if args.command == "health":
        print(json.dumps(store.health(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

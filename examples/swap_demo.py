#!/usr/bin/env python3
"""CLI demonstration of model-agnostic embedding swap (`plan_swap` + `execute_swap`).

Usage:
    python examples/swap_demo.py --from-dim 8 --to-dim 16 --base-dir /tmp/swap-demo
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
    EmbeddingModelConfig,
    MockEmbedder,
    ScopedMemoryStore,
    execute_swap,
    plan_swap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tree-RAG Model-Agnostic Embedding Swap Demo (plan_swap + execute_swap)"
    )
    parser.add_argument(
        "--from-dim",
        type=int,
        default=8,
        help="Initial vector dimension for seeding the store (default: 8)",
    )
    parser.add_argument(
        "--to-dim",
        type=int,
        default=16,
        help="Target vector dimension after swap (default: 16)",
    )
    parser.add_argument(
        "--base-dir",
        default="/tmp/swap-demo",
        help="Base storage directory for the demo workspace",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for checkpointed re-embedding",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_dir = Path(args.base_dir)

    store = ScopedMemoryStore(
        base_dir=base_dir,
        default_workspace="architecture_notes",
        embedder=MockEmbedder(dim=args.from_dim),
    )

    store.add_many(
        [
            (
                "decision",
                "Notes in docs.json are the single source of truth; FAISS vectors are derived.",
                "adr-010.md",
            ),
            (
                "code",
                "Atomic index swap writes to a temp file, fsyncs, and executes POSIX rename.",
                "swap.py",
            ),
            (
                "wiki",
                "Model-agnostic RAG allows upgrading from 384-d CPU models to 768-d GPU models.",
                "readme.md",
            ),
        ]
    )

    target_model = EmbeddingModelConfig(
        name=f"mock-embedder-{args.to_dim}d",
        dim=args.to_dim,
        model_id=f"mock-multilingual-{args.to_dim}d",
        server_url="http://embed.example:8089/v1",
        version="2.0.0",
        embedder=MockEmbedder(dim=args.to_dim),
    )

    plan = plan_swap(store, target_model)
    print("=== Swap Plan ===")
    print(json.dumps(plan.to_dict(), indent=2))

    report = execute_swap(store, target_model, batch_size=args.batch_size)
    print()
    print("=== Swap Report ===")
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

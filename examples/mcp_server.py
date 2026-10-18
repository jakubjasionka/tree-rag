#!/usr/bin/env python3
"""Example script launching the production FastMCP stdio server (`scoped_memory.mcp_server`).

Usage:
    python examples/mcp_server.py --print-tools
    python examples/mcp_server.py --stdio
"""

from __future__ import annotations

import sys

from scoped_memory.mcp_server import build_fastmcp_app, main

__all__ = ["build_fastmcp_app", "main"]

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

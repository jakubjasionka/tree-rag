# Memory MCP Hub — Hierarchical Tree RAG & Telemetry Dashboard

A standalone, zero-dependency Node.js web dashboard (`dashboard/`) for inspecting, querying, and managing **Tree-RAG (`scoped_memory`)** workspace indices in real time.

## Features

- **L0 → L3 Hierarchical Memory Pyramid**: Interactive filter across `L3 Persona (decision, skill)`, `L2 Scenario (code, wiki)`, `L1 Atomic Facts (bug, assumption)`, and `L0 Raw Context (note)`.
- **4 Memory Pillars**: Domain classification across **Chat & Context**, **Skills & Decisions**, **Knowledge & Wiki**, and **Code & Bug Graph**.
- **Interactive RAG Tree & 2D Graph Canvas**: Browse `Workspace → Category → Source` hierarchies in a collapsible tree or an interactive `<canvas>` radial/horizontal node graph.
- **Live Semantic Search & Auto-Highlighting**: Query across a single workspace or globally; matching workspace/category/source branches automatically highlight and expand in the RAG tree.
- **Drill-Down & Vector Neighbors**: Inspect any record's nearest semantic neighbors (reconstructed directly from `index.faiss`) alongside same-source sibling entries.
- **Full CRUD, Pending Queue Compaction & Mermaid Symbolic Canvas**: Add, edit, and delete records, merge queued `_pending/<workspace>/*.md` files, and export `flowchart LR` Mermaid diagrams.

---

## Architecture

```text
  Autonomous Agents (Cursor / Claude / CLI)
                    │
                    ▼  MCP over stdio (`python -m scoped_memory.mcp_server`)
        ┌───────────────────────┐
        │   ScopedMemoryStore   │
        │  (flock + SHA-256)    │
        └───────────┬───────────┘
                    │
                    ▼  On-disk workspace branches (`TREE_RAG_BASE_DIR`)
     ┌──────────────────────────────────────────────────┐
     │  <base_dir>/                                     │
     │    ├── <workspace>/docs.json                     │
     │    ├── <workspace>/index.faiss                   │
     │    ├── _pending/<workspace>/*.md                 │
     │    └── _telemetry.jsonl                          │
     └──────────────────────┬───────────────────────────┘
                            ▲
                            │  Direct JSON read + Python FAISS sidecar
        ┌───────────────────┴───────────────────┐
        │  dashboard/server.js (:8095)          │
        │    ├─ Static UI (index.html, app.js)  │
        │    ├─ dashboard/search_sidecar.py     │
        │    └─ Deterministic lexical fallback  │
        └───────────────────────────────────────┘
```

### Search & Indexing Backends

1. **Python FAISS Sidecar (`dashboard/search_sidecar.py`) — Default when `scoped_memory` is importable**:
   `dashboard/server.js` invokes `search_sidecar.py` over stdio JSON Lines to execute vector search, FAISS vector neighbor reconstruction, and atomic `docs.json` + `index.faiss` mutations via `ScopedMemoryStore` (using `CpuHttpEmbedder` when `TREE_RAG_EMBED_URL` is set, or `MockEmbedder` offline).
2. **Deterministic Lexical Fallback — Zero-Dependency Node Mode**:
   When Python or `scoped_memory` is unavailable (or `--lexical-only` / `TREE_RAG_FORCE_LEXICAL=1` is set), `server.js` falls back to deterministic token + character trigram cosine scoring directly over `docs.json`.

---

## Quick Start

Run directly with Node.js 18+ (no `npm install` required):

```bash
node dashboard/server.js --port 8095 --base-dir ./.treerag_data
```

Or try the synthetic demo dataset (`examples/sample_base`):

```bash
node dashboard/server.js --port 8095 --base-dir ./examples/sample_base
```

Then open `http://127.0.0.1:8095` in your browser.

### Run Smoke Tests

```bash
node dashboard/test/smoke.js
```

---

## HTTP API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/memory/status` | Engine status, workspace list, index byte sizes, pending count, embedder reachability |
| `GET` | `/api/memory/health` | Alias for `/api/memory/status` health snapshot |
| `GET` | `/api/memory/tree` | Full hierarchical RAG tree (`workspace → category → source`), L0–L3 counts, 4 pillars, tags, timeline, and telemetry |
| `GET` | `/api/memory/records` | Paginated record list (`?workspace=&category=&source=&layer=&pillar=&tag=&q=&limit=40&offset=0`) |
| `POST` | `/api/memory/search` | Vector/lexical search (`{ query, scope, workspace, category, mode, top_k }`) with tree branch highlights |
| `POST` | `/api/memory/neighbors` | Nearest vector neighbors + same-source siblings (`{ workspace, id, top_k }`) |
| `POST` | `/api/memory/save` | Create and index a new record (`{ workspace, category, source, tags, text }`) |
| `POST` | `/api/memory/update` | Update an existing record and refresh its vector (`{ id, workspace, category, source, tags, text }`) |
| `POST` | `/api/memory/delete` | Delete a record by ID and rebuild the workspace FAISS index (`{ workspace, id }`) |
| `GET` | `/api/memory/mermaid` | Generate a Mermaid `flowchart LR` symbolic canvas (`?workspace=global` or branch name) |
| `GET` | `/api/memory/pending` | Inspect queued markdown files in `_pending/<workspace>/*.md` |
| `POST` | `/api/memory/merge` | Compact queued `_pending` files into workspace FAISS indices |
| `POST` | `/api/memory/workspace-sync` | Heal sync conflicts and verify workspace indices |
| `GET` | `/api/memory/logs` | Tail recent MCP telemetry events from `_telemetry.jsonl` |

---

## Screenshots

> _Placeholder: Add screenshots of the L0→L3 Memory Pyramid, Interactive 2D RAG Graph Canvas, and FAISS Drill-Down Inspector under `docs/screenshots/`._

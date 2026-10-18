# Tree-RAG (`scoped_memory`) — Hierarchical Scoped Memory Engine for Autonomous Agents

![Status](https://img.shields.io/badge/Status-Public_Beta_%2F_Reference_Engine-success)
![VRAM](https://img.shields.io/badge/GPU_VRAM_Overhead-0_MB_(CPU_Offload)-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-informational)

> **Hierarchical scoped vector retrieval engine with deterministic zero-pollution guarantees, CPU embedding offloading, Human-in-the-Loop global escalation, and conflict-free multi-node persistence.**

---

## Key Engineering Takeaways (TL;DR)

- **Zero Context Pollution (`EmptyContext`):** Unlike flat vector databases where `top_k` queries unconditionally return $k$ low-relevance neighbors, Tree-RAG partitions memory into isolated per-workspace FAISS branches (`<base_dir>/<workspace>/index.faiss` + `docs.json`). Empty or below-threshold workspaces deterministically return `EmptyContext` (`bool(ctx) is False`, `0` noise tokens injected).
- **Human-in-the-Loop (HITL) Global Escalation:** When a local workspace branch misses (`ctx.requires_confirmation() is True`), the engine does not blindly search unrelated projects. It exposes an explicit escalation gate requiring agent/user confirmation (`confirmed=True` or approval callback) before traversing `scope="global"`.
- **0 MB GPU VRAM Overhead (CPU Offload):** Designed to pair with a compact multilingual embedding model (~100M parameters, 384-d L2-normalized vectors) running entirely on host/edge CPU, preserving 100% of GPU VRAM for generative LLM weights and KV-cache.
- **Crash-Safe Atomic Persistence & Non-Blocking Concurrency:** All index and metadata writes execute via `tmp + fsync + directory fsync + POSIX rename` guarded by per-workspace `flock` (`<workspace>/.lock`). Concurrent writers on different workspaces never serialize globally; same-workspace lock contention falls back losslessly to a markdown pending queue (`_pending/<workspace>/*.md`).

---

## The Problem: Always-Returns-Something & Context Pollution

Standard flat RAG pipelines store notes, architectural decisions, bug post-mortems, and code snippets across multiple repositories in a single shared vector index. When an autonomous coding agent starts work in a **brand-new or unrelated workspace** and runs `search(query, top_k=5)`:

1. **Unconditional `top-k` Retrieval:** Approximate Nearest Neighbor (ANN) and flat inner-product indices always return $k$ vectors—even when the highest cosine similarity is `0.12`.
2. **Cross-Project Hallucination:** Irrelevant architectural assumptions from Project A (e.g., Redis TTL rules, database schemas, or auth middleware) are injected into the prompt for Project B, poisoning the LLM context window and causing subtle architectural regressions.

---

## Zero-Pollution Guarantee

Tree-RAG enforces strict namespace isolation at the filesystem and index level:

- Every workspace is an independent branch (`<base_dir>/<workspace>/`) containing its own positionally aligned `index.faiss` (`IndexFlatIP`) and `docs.json` source of truth.
- Searching with `scope="workspace"` (the default) queries **only** the active workspace branch.
- If the branch is empty or no document meets `min_score`, `store.search()` returns an immutable `EmptyContext`:
  - `bool(result) is False` and `len(result) == 0`
  - **0 tokens** of cross-project noise enter the agent's prompt.
  - Content is deduplicated on write by SHA-256 hash (`text_hash`) of whitespace-normalized, secret-scrubbed text, plus cosine near-duplicate filtering (`>= 0.955`).

---

## Hierarchical Escalation + Human-in-the-Loop

Retrieval follows a strict two-tier hierarchy (`scope="workspace"` -> `scope="global"`):

1. **Tier 1 — Local Branch (`scope="workspace"`):** Fast, isolated lookup within the current repository namespace.
2. **Escalation Signal (`requires_confirmation()`):** When Tier 1 yields `EmptyContext`, `result.requires_confirmation()` returns `True`, signaling to the agent orchestrator that local memory had no high-confidence match.
3. **Tier 2 — Global Tree (`scope="global"`):** Cross-workspace search across all branches is gated by Human-in-the-Loop confirmation (`confirmed=True` or `confirmation_callback`). Every global hit is explicitly provenance-tagged with `[workspace=<branch_name>]` so the LLM knows the exact origin of the retrieved context.

---

## CPU Offloading for Embeddings (0 MB VRAM)

Local AI workstations and edge nodes are almost always **VRAM-constrained** by the primary LLM's weights and KV-cache. Loading a secondary embedding model onto the GPU steals 650 MB–2 GB of VRAM and introduces CUDA context-switching jitter.

Tree-RAG decouples embedding via the `Embedder` interface (`scoped_memory/embed.py`):

- **`CpuHttpEmbedder`:** Connects to a lightweight, OpenAI-compatible CPU embedding server running a compact multilingual embedding model (~100M parameters, 384 dimensions, L2-normalized) with **0 MB GPU VRAM** consumption.
- **`MockEmbedder`:** Deterministic, zero-dependency SHA-256 lexical/n-gram feature hashing embedder (384-d) for fast offline unit testing and CI pipelines.

---

## Conflict-Free Multi-Node Sync

When synchronizing agent memory across multiple machines (e.g., workstation and edge nodes):

- **Atomic Writes (`scoped_memory/atomic.py`):** `_atomic_write_json()` and `_atomic_write_faiss()` write to temporary sibling files, flush kernel buffers (`os.fsync`), `fsync` the parent directory descriptor, and execute an atomic POSIX `os.rename()`. File watchers never observe half-written files.
- **Read-Only Concurrency Guard & Pending Queue (`scoped_memory/guard.py`):** When `ConcurrencyGuard` detects an active concurrent writer, lock contention, or an unresolved `*.sync-conflict-*` artifact, `store.save()` automatically switches to non-destructive read-only fallback—writing markdown files with YAML frontmatter to `_pending/<workspace>/<timestamp>_<id>.md`.
- **Lossless Compaction & Union Merge:** `store.merge_pending()` drains `_pending/<ws>/*.md` into FAISS once locks are free. `merge_doc_collections()` and `heal_workspace_conflicts()` reconstruct 384-d vectors directly from existing FAISS indices (`index.reconstruct(i)`) without re-embedding unchanged documents, configured via abstract `PeerNode` (`http://peer-node.example:PORT/v1`) and `SyncPolicy`.

---

## Architecture

```mermaid
flowchart TD
    Query([Agent Query]) --> LocalScope[Tier 1: Workspace-Scope Lookup
scoped_memory/index.py]
    LocalScope --> CheckEmpty{Workspace Empty or
Score < min_score?}

    CheckEmpty -->|No: Qualifying Hits| ReturnLocal[Return SearchResult
 Local Workspace Context Only]
    CheckEmpty -->|Yes: 0 Qualifying Hits| EmptyCtx[Return Deterministic EmptyContext
0 Noise Tokens / requires_confirmation = True]

    EmptyCtx --> HITLGate{{Human-in-the-Loop Gate
confirmed=True or Callback Approved?}}
    HITLGate -->|Denied / Unconfirmed| SafeContinue[Abort Escalation
Zero Context Pollution]
    HITLGate -->|Approved| GlobalScope[Tier 2: Global-Scope Traversal
All Workspaces Tagged with workspace=...]
```

### Package Layout (`scoped_memory/`)

| Module | Responsibility |
| :--- | :--- |
| `scoped_memory/store.py` | `ScopedMemoryStore` facade: per-workspace `save`, `search`, `list`, `merge_pending`, `heal_workspace`, `health` |
| `scoped_memory/index.py` | `WorkspaceIndex`: per-branch FAISS `IndexFlatIP` + `docs.json`, optimistic `(mtime, size)` reload, vector reconstruction |
| `scoped_memory/hierarchy.py` | `HierarchyEngine`, `EmptyContext`, `SearchResult`, `SearchHit`, and `requires_confirmation()` HITL gate |
| `scoped_memory/atomic.py` | `_atomic_write_json()` and `_atomic_write_faiss()` (`tmp + fsync + dir fsync + rename`) |
| `scoped_memory/locks.py` | Per-workspace non-blocking POSIX `flock` (`<ws>/.lock`) and global merger lock (`.global.lock`) |
| `scoped_memory/guard.py` | `ConcurrencyGuard`, `_pending/<ws>/*.md` queue, conflict-free union merge, `PeerNode` & `SyncPolicy` |
| `scoped_memory/embed.py` | `Embedder` ABC, `CpuHttpEmbedder` (~100M params CPU offload), `MockEmbedder`, sentence-boundary `chunk_text` |
| `scoped_memory/sanitize.py` | `_scrub_secrets()` (redacts API/OAuth tokens), `sanitize()` workspace slug normalizer, `text_hash()` |

---

## Quickstart (<3 Steps)

```bash
# 1. Install the package and development dependencies
pip install -e ".[dev]"

# 2. Run the Zero-Pollution & Human-in-the-Loop escalation demo
python examples/demo_zero_pollution.py

# 3. Run the CPU unit test suite (MockEmbedder, zero external servers required)
pytest -v
```

---

## Benchmarks

> **Note (Illustrative / Reference Values):** The figures below are **illustrative / expected reference values** from architectural characterization of a compact multilingual CPU embedding model (~100M parameters, 384-d `IndexFlatIP`) vs. a standard GPU-resident embedding pipeline. They are provided as reference design targets rather than hardware-bound claims.

| Metric / Scenario | Flat RAG (GPU Embedder) | **Tree-RAG (CPU Offload — Reference Values)** | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **GPU VRAM Allocated to Embeddings** | 650 – 1,400 MB VRAM | **0 MB VRAM** *(reference value, 100% CPU)* | 100% of GPU VRAM preserved for primary LLM KV-cache |
| **Single Query Embed Latency (`v1`, batch=1)** | ~25 ms | **~160 ms** *(illustrative / expected on CPU)* | Negligible in agentic tool loops; zero GPU preemption |
| **Batch Embed Latency (`batch=10` passages)** | ~45 ms | **~300 ms** *(illustrative / expected, ~30 ms/vec)* | Sub-linear amortization during `_pending` queue merge |
| **False-Positive Pollution on Empty Workspace** | 100% (returns $k$ arbitrary neighbors) | **0.0%** (`EmptyContext` + HITL gate) | Eliminates cross-project context poisoning |

---

## Security / Sanitization Policy

1. **Automatic Pre-Persistence Secret Scrubbing (`scoped_memory/sanitize.py`):** Every document passed to `save()` is scrubbed by `_scrub_secrets()` before hashing, embedding, or writing to disk (`docs.json` or `_pending/*.md`). Recognized patterns include GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`), OpenAI/Anthropic-style keys (`sk-`), serverless keys (`ak-`, `as-`), Slack tokens (`xox*`), X/Twitter bearer/access tokens, and LinkedIn OAuth cookies (`li_at=`).
2. **Strict Namespace Sanitization:** Workspace identifiers are canonicalized via `sanitize()` into lowercase alphanumeric slugs (`[a-z0-9_]+`), preventing path traversal (`../`) or hidden dot-directory injection.
3. **RFC 2606 Configuration Placeholders:** All network endpoints in `.env.example` and `PeerNode` defaults use reserved example domains (`http://embed.example:PORT/v1`, `http://peer-node.example:PORT/v1`). Never commit `.env` files or runtime `.treerag_data/` directories.

---

## License

Released under the [MIT License](LICENSE).

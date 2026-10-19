# Measured results: embedding latency

Server: `intfloat/multilingual-e5-small` (384-dim), served on loopback `127.0.0.1:8089`
on a host with an NVIDIA RTX 5090. The device that ran the model was **not verified**,
so these numbers are not attributed to CPU or GPU. Client on the same host (no network hop).

Command: `TREE_RAG_EMBED_URL=http://127.0.0.1:8089/v1 python benchmarks/embed_latency.py`

| Batch size | Samples | p50 | p95 | min |
| :--- | :--- | :--- | :--- | :--- |
| 1 (single query) | 200 | 13.4 ms | 15.4 ms | 11.4 ms |
| 10 (passages) | 100 | 31.4 ms | 34.5 ms | 24.1 ms |

Not measured: concurrent throughput, the edge node, or any CPU-vs-GPU comparison.

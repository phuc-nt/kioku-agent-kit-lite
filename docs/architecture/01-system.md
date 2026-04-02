# System Architecture — kioku-agent-kit-lite

> Last updated: 2026-04-02 (v0.1.29-dev)

## Overview

kioku-agent-kit-lite is a personal memory engine for AI agents with a **zero-dependency** design: no Docker, no external servers. All storage is in a single SQLite file, and embedding runs locally via ONNX.

**Design philosophy:**
- **Agent-driven KG** — kioku-lite never calls an LLM. The agent extracts entities → calls `kg-index` to store them in the graph.
- **SQLite-everything** — BM25 (FTS5) + vector (sqlite-vec) + knowledge graph all in one `.db` file.
- **Offline-capable** — FastEmbed ONNX runs without internet once the model has been downloaded.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     INTERFACE LAYER                          │
│                                                              │
│   ┌───────────────────────────────────────────────────────┐  │
│   │  cli.py  (Typer CLI — 16 commands)                   │  │
│   │  Write: save kg-index kg-invalidate kg-alias        │  │
│   │  Query: search recall connect timeline entities      │  │
│   │  Dedup: dedup-scan merge                             │  │
│   │  Consolidate: consolidate clusters cluster           │  │
│   │  System: users setup init install-profile export     │  │
│   └──────────────────────────┬────────────────────────────┘  │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│               KiokuLiteService  (service.py)                 │
│   save_memory() │ search() │ kg_index() │ delete_memory()   │
└────────┬─────────────────┬─────────────────────┬─────────────┘
         │                 │                     │
         ▼                 ▼                     ▼
  MarkdownStore        Embedder              KiokuDB
  ~/memory/*.md       FastEmbed             (single .db)
  (human backup)      ONNX local    ┌────────────────────────┐
                                    │  SQLiteStore           │
                                    │  ├── memories (FTS5)   │
                                    │  └── memory_vec        │
                                    │      (sqlite-vec)      │
                                    │                        │
                                    │  GraphStore            │
                                    │  ├── kg_nodes          │
                                    │  ├── kg_edges          │
                                    │  └── kg_aliases        │
                                    └────────────────────────┘
```

## Module Structure

```
src/kioku_lite/
├── __init__.py
├── config.py          # Settings (Pydantic) — env vars KIOKU_LITE_*
├── service.py         # KiokuLiteService — all business logic
├── cli.py             # CLI (Typer) — 16 commands (write, query, dedup, consolidate)
│
├── pipeline/          # WRITE path + consolidation pipeline
│   ├── db.py          # KiokuDB — facade for SQLiteStore + GraphStore
│   ├── sqlite_store.py # FTS5 (BM25) + sqlite-vec (vector) + temporal tables
│   ├── graph_store.py  # SQLite KG: entities, relations, aliases, temporal validity
│   ├── embedder.py    # FastEmbedder (ONNX) | OllamaEmbedder | FakeEmbedder
│   ├── consolidation.py # Decay & merge detection (confidence decay pipeline)
│   ├── dedup.py       # Dual-threshold entity deduplication (vec + name sim)
│   ├── clustering.py  # Connected component detection (BFS-based)
│   └── models.py      # Data models (ConsolidationReport, DedupeResult, etc.)
│
├── search/            # READ path
│   ├── bm25.py        # BM25 keyword search (SQLite FTS5)
│   ├── semantic.py    # Vector similarity (sqlite-vec ANN)
│   ├── graph.py       # Graph traversal (SQLite BFS, supports --include-historical)
│   ├── pagerank.py    # Personalized PageRank (PPR) for entity-focused search
│   └── reranker.py    # Reciprocal Rank Fusion (RRF)
│
└── storage/
    └── markdown.py    # Markdown file I/O (source of truth)
```

## Data Flow: Save

See [02-write-save-kg-index.md](02-write-save-kg-index.md) for full details.

| Step | Component | Output |
|---|---|---|
| 1 | SHA256(text) | `content_hash` — universal dedup key |
| 2 | Embedder.embed("passage: " + text) | 1024-dim vector |
| 3a | MarkdownStore | `~/memory/YYYY-MM/hash.md` |
| 3b | SQLiteStore.upsert_memory | FTS5 indexed row |
| 3c | SQLiteStore.upsert_vector | sqlite-vec row |
| 4 | *(Agent calls kg-index separately)* | entities + relations in GraphStore |

## Data Flow: Search

See [03-search.md](03-search.md) for full details.

| Step | Component | Output |
|---|---|---|
| 1 | Embedder.embed("query: " + text) | 1024-dim query vector |
| 2a | BM25Search | top-K BM25 results |
| 2b | SemanticSearch | top-K cosine similarity results |
| 2c | GraphSearch | entity-linked memories |
| 3 | RRF Reranker | fused, deduplicated top-N |

## Data Flow: KG Index (Agent-Driven, 3-Step)

See [02-write-save-kg-index.md](02-write-save-kg-index.md) for full details.

```
Step 1 — Disambiguate:
  kioku-lite entities --limit 50
  → Compare against existing canonical names, reuse matches

Step 2 — Extract from context:
  entities      → [{"name": "Alice", "type": "PERSON"}, ...]
  relationships → [{"source": "Alice", "rel_type": "WORKS_ON", "target": "Kioku", "evidence": "..."}]
  event_time    → parse relative dates to YYYY-MM-DD

Step 3 — Index:
  kioku-lite kg-index <hash> --entities '...' --relationships '...' --event-time YYYY-MM-DD
        ↓
  GraphStore.upsert_node()  → kg_nodes
  GraphStore.upsert_edge()  → kg_edges (with source_hash + event_time)
```

## Configuration

All settings via environment variables with prefix `KIOKU_LITE_`:

| Variable | Default | Purpose |
|---|---|---|
| `KIOKU_LITE_USER_ID` | `default` | User isolation |
| `KIOKU_LITE_DATA_DIR` | `~/.kioku-lite/data` | SQLite DB location |
| `KIOKU_LITE_MEMORY_DIR` | `~/.kioku-lite/memory` | Markdown files |
| `KIOKU_LITE_EMBED_PROVIDER` | `fastembed` | `fastembed` \| `ollama` \| `fake` |
| `KIOKU_LITE_EMBED_MODEL` | `intfloat/multilingual-e5-large` | Model name |
| `KIOKU_LITE_EMBED_DIM` | `1024` | Embedding dimensions |
| `KIOKU_LITE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama URL (if using Ollama) |

## Multi-User Isolation

```
~/.kioku-lite/
├── users/
│   ├── default/
│   │   ├── data/kioku.db     ← SQLite (FTS5 + vec + KG)
│   │   └── memory/           ← Markdown files
│   │       └── 2026-02/
│   │           └── abc123.md
│   └── alice/
│       ├── data/kioku.db
│       └── memory/
└── ...
```

## Graceful Degradation

| Component Down | Impact |
|---|---|
| Embedder / FastEmbed | Vector indexing skipped. BM25 + KG still work. |
| sqlite-vec extension | Vector search skipped. BM25 + KG still work. |
| GraphStore | KG search skipped. BM25 + Vector still work. |
| SQLite | ❌ Critical — all search operations fail |

## Temporal Validity Layer (v0.1.29)

Facts in a personal knowledge graph age and become superseded. kioku-lite tracks validity windows:

| Feature | Implementation |
|---|---|
| **Storage** | `kg_edges.valid_from`, `kg_edges.valid_until` |
| **Invalidation** | `kg-invalidate` CLI: mark edge as no longer valid with date + reason |
| **Query control** | `--include-historical` flag on `search`, `recall`, `connect` |
| **Default behavior** | Queries exclude superseded edges unless `--include-historical` is set |
| **Use case** | Agent marks outdated facts (job change, breakup, resolved issue) without deletion |

Example: "Phuc works at LINE" (valid 2023-01-01 to 2026-03-31) → agent marks `valid_until=2026-03-31` when he changes jobs.

## Confidence Decay & Consolidation (v0.1.29)

Memory systems accumulate stale or weakly-reinforced facts. Consolidation surfaces them for agent action:

| Feature | Formula | Purpose |
|---|---|---|
| **Confidence decay** | `weight_t = weight * 0.5^(days_since_reinforced / half_life)` | Reduce signal from rarely-reinforced edges |
| **Stale detection** | Memories last updated > `--older-than` days | Surface old entries for summarization |
| **Merge suggestions** | Dual-threshold: vec_sim ≥ 0.98 AND jaro_winkler ≥ 0.85 | Find duplicate entity pairs |
| **Output** | JSON report: `{decay, merge_suggestions, stale_memories}` | Agent-driven (agent reviews, then acts) |

Run `kioku-lite consolidate --half-life 90 --older-than 30` to surface candidates.

## Entity Resolution & Deduplication (v0.1.29)

Personal KGs suffer from entity aliases and misspellings. kioku-lite auto-dedupes:

| Feature | Details |
|---|---|
| **Detection** | Dual threshold: cosine_sim(embedding) ≥ 0.98 AND jaro_winkler(name) ≥ 0.85 |
| **Auto-merge** | `dedup-scan --auto` merges qualifying pairs automatically |
| **Manual merge** | `merge <source> <target>` consolidates with audit trail in `kg_merge_log` |
| **Confidence scoring** | Entities track confidence (0.0–1.0), set to MAX on upsert/merge |
| **Audit trail** | `kg_merge_log` table records all merges with source, target, timestamp |

Example: "Phuc" (10 mentions) + "Phúc" (3 mentions, 0.99 similarity) → auto-merge into "Phúc".

## Personalized PageRank for Entity-Focused Search (v0.1.29)

When user asks about relationships between specific people/projects, BFS alone misses important indirect connections. PPR reorders by relevance to seed entities:

| Aspect | Details |
|---|---|
| **Activation** | When `search --entities "Alice,Bob,Charlie"` is called |
| **Algorithm** | Pure Python power iteration, damping=0.85 |
| **Kept untouched** | `recall` and `connect` remain BFS-based (simpler, focused) |
| **Benefit** | Better multi-hop associative recall (e.g., "who is Alice likely to know?") |

## Community/Cluster Detection (v0.1.29)

KGs fragment into communities (friend groups, project clusters, work domains). Detection helps consolidation:

| Command | Purpose |
|---|---|
| `kioku-lite clusters` | List all detected clusters with auto-assigned labels (from most common entity type) |
| `kioku-lite cluster PERSON` | Show all entities + memories in the PERSON cluster |
| **Integration** | Clusters included in `consolidate` report for structural analysis |
| **Method** | Connected components via BFS, O(V+E) |

Example output: `{label: "PERSON", entity_count: 42, memory_count: 156}`

## Graph Search: Hub Node Fixes (v0.1.27–0.1.28)

In personal knowledge graphs, the user's own entity (e.g. "Alice") appears in nearly every memory. Without mitigation, graph search traversing from this "hub" returns 90%+ of all memories — providing no signal.

Three layers of protection were added:

| Fix | Version | Description |
|---|---|---|
| **1A — Self-entity exclusion** | v0.1.27 | `get_top_entity()` detects the hub (highest `mention_count`). If other seeds exist, hub is excluded from BFS traversal. |
| **1C — Adaptive hop limit** | v0.1.27 | `get_degree(entity)` counts edges. If `degree > 15` → `effective_hops = 1`, else use `max_hops`. |
| **2E — Multi-entity intersection** | v0.1.28 | When 2+ entity seeds: only return memories reachable from ALL seeds. Fallback to union if intersection empty. |

See [proposals/2026-03-03-hub-node-problem.md](../proposals/2026-03-03-hub-node-problem.md) for detailed analysis.

## Comparison: kioku-lite vs kioku-server

kioku-server (the planned enterprise version, inheriting core logic from kioku-lite) is designed for **enterprise and multi-tenant** cloud deployments.

| | kioku-lite (current) | kioku-server (planned) |
|---|---|---|
| **Target use case** | Personal, edge, single-agent | Enterprise, multi-tenant, team memory |
| **Interface** | CLI + SKILL.md | MCP Server (JSON-RPC) |
| **Vector store** | sqlite-vec (in-process) | ChromaDB (dedicated container) |
| **Graph store** | SQLite BFS (adaptive hops, intersection) | FalkorDB (property graph, Cypher) |
| **Embedding** | FastEmbed ONNX (local) | Ollama / cloud API (configurable) |
| **KG extraction** | Agent-driven (no built-in LLM) | Agent-driven (same design) |
| **Search algorithms** | Tri-hybrid + RRF | Same tri-hybrid + RRF core |
| **Setup** | `pipx install "kioku-lite[cli]"` | Docker Compose / Kubernetes |
| **Offline capability** | ✅ (after model download) | Configurable |
| **Multi-tenant** | Profile-based isolation | Full multi-tenant with API keys |
| **Status** | ✅ Available (v0.1.28) | 🔨 In development |

See [blog/2026-03-04-kioku-server-roadmap-en.md](../blog/2026-03-04-kioku-server-roadmap-en.md) for the full roadmap and comparison with Anthropic's MCP Memory Server.

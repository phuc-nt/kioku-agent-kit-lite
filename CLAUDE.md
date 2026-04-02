# kioku-agent-kit-lite — Dev Guide

## What This Is

Python package `kioku-lite` on PyPI. Local-first memory engine: tri-hybrid search (BM25 + vector + KG) in a single SQLite file, zero Docker, zero LLM calls internally.

## Development

```bash
pip install -e ".[cli,dev]"
pytest                          # 149+ tests, uses FakeEmbedder (no model download)
ruff check . && ruff format .   # lint — 100-char line, Python 3.11+
```

## Key Paths

- `src/kioku_lite/service.py` — core orchestrator (start here)
- `src/kioku_lite/cli.py` — Typer CLI (16 commands)
- `src/kioku_lite/pipeline/` — write path (DB, embedder, stores)
  - `consolidation.py` — decay & merge detection
  - `dedup.py` — dual-threshold entity deduplication
  - `clustering.py` — connected component detection
- `src/kioku_lite/search/` — read path (BM25, vector, PPR graph, RRF reranker)
  - `pagerank.py` — personalized PageRank for entity-focused search
- `src/kioku_lite/resources/` — agent skill files & persona profiles
- `docs/proposals/` — feature research & proposals
- `docs/architecture/` — design docs (6 files)

## Release Flow

1. All tests pass (`pytest`)
2. Bump version in `pyproject.toml`
3. Update `CHANGELOG.md`
4. Build & publish: `python -m build && twine upload dist/*`
5. After release — update `kioku-lite-landing/` (separate repo)

## Architecture Quick Ref

```
save → Markdown + FTS5 + sqlite-vec embedding → content_hash

kg-index → agent extracts entities → upsert nodes/edges with temporal validity (agent-driven, no internal LLM)

kg-invalidate → mark edge as superseded with valid_until date & reason

search → BM25 ∪ Vector ∪ PPR(entities) → RRF rerank → hydrate by content_hash
  - PPR activates when --entities provided (replaces BFS)
  - BFS kept for recall/connect (untouched)
  - --include-historical includes superseded edges in traversal

consolidate → decay weights, find duplicate pairs, surface stale memories (agent-driven report)
  - weight_t = weight * 0.5^(days_since_reinforced / half_life)
  - output: {decay, merge_suggestions, stale_memories}

clusters / cluster → detect and explore connected components in KG
  - auto-labeled from most common entity type
```

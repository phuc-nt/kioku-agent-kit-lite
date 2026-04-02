# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Temporal Fact Management**: `valid_from` / `valid_until` columns on `kg_edges`
  - `kg-invalidate` CLI command: mark facts as superseded with `--source`, `--target`, `--rel-type`, `--date`, `--reason`
  - `--include-historical` flag on `search`, `recall`, `connect` to include superseded facts
  - Agent can mark facts as no longer valid and track temporal validity
- **Memory Consolidation**: decay stale edges and surface old memories
  - `consolidate` CLI command: `--half-life` (days), `--older-than` (days), `--auto-merge`
  - Confidence decay: `weight * 0.5^(days/half_life)` with `last_reinforced` tracking
  - JSON report output: decayed edges, merge suggestions, stale memories (agent-driven action)
- **Entity Resolution + Auto-Dedup**:
  - Dual-threshold dedup: vector similarity ≥0.98 AND Jaro-Winkler name ≥0.85 → auto-merge
  - `dedup-scan` CLI command with optional `--auto` flag
  - `merge` CLI command: source → target consolidation with audit log
  - Confidence scoring on entities (0.0–1.0, MAX on upsert)
  - `kg_merge_log` audit table tracks all merges
  - `kg-index` response now includes `auto_merged` and `dedup_candidates`
- **Personalized PageRank (PPR)**:
  - PPR replaces BFS for entity-focused search (when `--entities` provided)
  - BFS retained for `recall` / `connect` (untouched)
  - Pure Python power iteration, damping=0.85
  - Better multi-hop associative recall for entity-seeded queries
- **Community/Cluster Detection**:
  - `clusters` CLI command: list all detected clusters with suggested labels
  - `cluster <label>` CLI command: show entities and connected memories in a cluster
  - Connected components via BFS, auto-labeled from most common entity type
  - Integrated into `consolidate` report for structural analysis
- 375 tests passing across all new modules

## [0.1.28] — 2026-03-03

### Fixed
- **Graph search — multi-entity intersection (Task 2E)**: `graph_search()` with 2+ seed entities now returns only memories reachable from **all** seeds (intersection), instead of any seed (union). Prevents 2-hop traversal from expanding through the hub node indirectly. Falls back to union if no memories co-occur across all seeds.

### Changed
- **SKILL.md (Task 2H)**: Added `--entities` rules — do NOT include user's own name (auto-excluded by engine), do NOT pass a single entity (use `recall` instead). Added concrete examples contrasting correct vs incorrect usage.
- **TOOLS.md** (companion + mentor profiles): Same rules added to the "Search Enrichment" section.

### Added
- 5 new tests in `TestMultiEntityIntersection`: intersection precision, union fallback, single seed, 3-entity, token-based no-entities

## [0.1.27] — 2026-03-03

### Fixed
- **Graph search — hub node problem (Task 1A)**: `graph_search()` now detects the user's self-entity (highest `mention_count`) and excludes it from BFS seeds when other entities are present. Prevents hub node from flooding results with 90%+ of all memories. Fallback: hub is kept if it is the only entity passed.
- **Graph search — adaptive hop limit (Task 1C)**: `traverse()` now uses `effective_hops = 1` for any entity with `degree > 15` (hub nodes), vs full `max_hops` for normal nodes. Defense-in-depth against supernode traversal explosion (e.g., direct `recall` on hub).

### Added
- `GraphStore.get_top_entity()` — returns entity name with highest `mention_count`
- `GraphStore.get_degree(entity_name)` — counts total edges (in + out) for an entity
- 19 new tests: `TestGetTopEntity`, `TestGetDegree`, `TestAdaptiveHopLimit` (in `test_graph_store.py`) and `TestGraphSearchBasic`, `TestSelfEntityExclusion` (new `test_graph_search.py`)

## [0.1.26] — 2026-03-03

### Fixed
- **Search**: SKILL.md now mandates `--entities` param to activate graph backend
- **KG indexing**: Added rule to prefer proper names over generic labels ("Phong" not "Con trai")

## [0.1.25] — 2026-03-03

### Fixed
- **Export graph**: Fixed duplicate title in HTML export (pyvis heading rendered both as visible h1 and `<title>` tag)

## [0.1.24] — 2026-03-03

### Changed
- **CRITICAL directive**: All SOUL.md, TOOLS.md, and AGENTS.md templates now explicitly state that `kioku-lite` is the **ONLY** memory system. Agents must NOT use USER.md, notes, or files to store user information — everything goes through `kioku-lite save` + `kg-index`.

## [0.1.23] — 2026-03-03

### Fixed
- **Search date filter now uses `event_time`** when available, falling back to `date`. Previously all temporal queries ("năm 2019") returned 0 results because memories were filtered by processing date (today) instead of event date.

### Changed
- `FTSResult` and `SearchResult` now carry `event_time` field through the search pipeline
- Search output includes `event_time` in results JSON
- SKILL.md: `--event-time` is now marked as REQUIRED on `save` (not just `kg-index`)

## [0.1.22] — 2026-03-03

### Changed
- **Restructured SKILL.md** — clearer sections, less redundancy, same 11 sections but more substance
- **kg-index now a 3-step process**: disambiguate (check existing entities) → extract → index
- Added `--event-time` documentation to kg-index with relative date parsing guide
- Added entity disambiguation guidance: check `kioku-lite entities` before extracting
- Clarified `evidence` field: must be exact quote from saved text
- Updated OpenClaw TOOLS.md templates (companion + mentor) with same improvements

## [0.1.21] — 2026-03-03

### Added
- **Entry Splitting Strategy** in SKILL.md — quantifiable criteria for when agents should split large saves into multiple focused entries (≥3 topics, ≥10 entities, ≥2 time phases, >300 words + multiple topics)
- Entry splitting rules added to OpenClaw TOOLS.md templates (companion + mentor profiles)

### Changed
- Decision tree updated: save step now includes splitting check before kg-index

## [0.1.18] — 2026-03-02

### Added
- Setup guide for OpenClaw agent integration (`docs/guides/setup-guide-for-openclaw-agent.md`)
- Telegram-specific setup guide (`docs/guides/openclaw-telegram-setup.md`)
- Architecture documentation: system overview, write pipeline, search pipeline, KG open schema

### Changed
- Removed PATH symlink requirement for OpenClaw — `~/.local/bin` is already in OpenClaw's LaunchAgent PATH
- Cleaned up redundant generic entity types from OpenClaw profile TOOLS.md files

## [0.1.15] — 2026-03-01

### Added
- Agent Profile System: built-in personas via `kioku-lite install-profile <name>`
  - `companion` — emotional companion with schema: EMOTION, LIFE_EVENT, TRIGGERED_BY
  - `mentor` — business & career mentor with schema: DECISION, LESSON_LEARNED, LED_TO_LESSON
- Profile files: each persona includes pre-written `AGENTS.md` + `SKILL.md`, deployable instantly

### Fixed
- `kioku-lite init` now creates `AGENTS.md` instead of `CLAUDE.md` (open standard compatible)
- Skill directory changed from `.claude/skills/` to `.agents/skills/` (works with Claude Code, Cursor, Windsurf)

## [0.1.14] — 2026-02-28

### Fixed
- **`connect` always returned empty `source_memories`** (two-part bug):
  - `find_path()` in `graph_store.py` did not fetch `source_hash` from DB
  - `explain_connection()` in `service.py` used `.values()` instead of `.items()`, losing hash keys
- `source_memories` now includes `content`, `date`, `mood`, and `content_hash`

## [0.1.13] — 2026-02-28

### Fixed
- SKILL.md: `explain-connection` → `connect` (3 occurrences corrected to match actual CLI command)

## [0.1.12] — 2026-02-28

### Added
- Enriched search workflow in SKILL.md (Section 6): 5-step decision tree with 6 query case types
- Agent query enrichment: pronoun resolution, implicit entities, type inference, temporal ranges

## [0.1.11] — 2026-02-28

### Fixed
- **BM25 search always returned 0 results**: FTS5 was doing phrase match instead of term match
- **`content_hash` missing from search/recall/timeline output**: agent could not reference memories for `kg-index`
- FastEmbed pooling-method warning suppressed (cosmetic, not functional)

## [0.1.0] — 2026-02-27

### Added
- Initial release of kioku-lite
- Tri-hybrid search: BM25 (SQLite FTS5) + Vector (sqlite-vec) + Knowledge Graph
- FastEmbed ONNX embedder — local, offline-capable (`intfloat/multilingual-e5-large`)
- OllamaEmbedder — HTTP-based for dev/benchmark comparison
- CLI commands: `save`, `search`, `kg-index`, `recall`, `connect`, `entities`, `timeline`, `users`
- Agent-driven KG indexing via `kg-index` command
- Multi-user support via `kioku-lite users` with profile isolation
- SQLite-based graph store (BFS traversal, entity aliases, open schema)
- Markdown file storage (human-readable backup, source of truth)
- Comprehensive test suite: 149+ tests across 5 modules
- PyPI-ready packaging via Hatchling

### Architecture Decisions
- Zero Docker — all storage in a single SQLite file
- Agent-driven KG: kioku-lite stores what the agent provides; no built-in LLM calls
- Embedding default: `intfloat/multilingual-e5-large` (1024-dim, multilingual, 100+ languages)
- E5 instruction format: `passage:` for indexing, `query:` for search
- Open KG schema: entity types and relationship types are plain strings, not fixed enums

### Benchmark (vs kioku-agent-kit full Docker)
- Search latency: **1.2s vs 2–3s** (kioku-lite 1.7–7.6× faster)
- Precision@3: **0.60 = 0.60** (equal quality with same KG extraction)
- Infrastructure: `pip install` vs 3 Docker containers

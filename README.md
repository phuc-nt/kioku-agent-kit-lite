# kioku-lite

> Personal memory engine for AI agents. Tri-hybrid search, zero Docker, single SQLite file.

[![PyPI](https://img.shields.io/pypi/v/kioku-lite)](https://pypi.org/project/kioku-lite/)
[![Downloads](https://static.pepy.tech/badge/kioku-lite)](https://pepy.tech/project/kioku-lite)
[![Python](https://img.shields.io/pypi/pyversions/kioku-lite)](https://pypi.org/project/kioku-lite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**kioku-lite** — *kioku* (記憶) means "memory" in Japanese: 記 (ki) to record, 憶 (oku) to remember. A lightweight, fully local memory engine that gives AI agents long-term memory with causal reasoning, using tri-hybrid search (BM25 + Vector + Knowledge Graph) in a single SQLite file.

**[Homepage](https://phuc-nt.github.io/kioku-lite-landing/)** · **[Docs](https://phuc-nt.github.io/kioku-lite-landing/blog.html)** · **[The Story](https://phuc-nt.github.io/kioku-lite-landing/blog.html#kioku-intro)** · **[PyPI](https://pypi.org/project/kioku-lite/)**

> **Read the intro:** [Why I built a Knowledge Graph memory engine for my AI agents](https://phuc-nt.github.io/kioku-lite-landing/blog.html#kioku-intro) — also in [日本語](https://phuc-nt.github.io/kioku-lite-landing/blog.html#kioku-intro-ja) and [Tiếng Việt](https://phuc-nt.github.io/kioku-lite-landing/blog.html#kioku-intro-vn)

![kioku-lite](https://raw.githubusercontent.com/phuc-nt/kioku-agent-kit-lite/main/docs/blog/img/image.png)

---

## Why kioku-lite

Most agent memory systems store flat text or vectors. They can't answer *"Why was I stressed last month?"* because they don't track how events, emotions, and decisions connect. kioku-lite solves this with a Knowledge Graph on top of traditional search — running 100% local with no LLM calls.

| System | Infrastructure | LLM required | Search | Knowledge Graph |
|---|---|---|---|---|
| **Mem0** | Cloud-managed | Yes (every write) | Vector + Graph | ✅ Managed |
| **Claude Code** | Flat markdown files | No | Context window only | ❌ |
| **OpenClaw** | SQLite per-agent | No | Semantic (embedding) | ❌ |
| **kioku-lite** | Single SQLite file | Agent-driven, zero extra | Tri-hybrid (BM25 + vector + KG) | ✅ Agent-driven |

> **vs Mem0:** Mem0 targets production apps — managed cloud infra, automatic LLM extraction on every write. kioku-lite targets personal use — fully local, fully offline, the agent is already an LLM so no extra call needed. Full comparison: [docs](https://phuc-nt.github.io/kioku-lite-landing/blog.html#memory-comparison)

## Features

- **Tri-hybrid search** — BM25 (FTS5) + Vector (sqlite-vec) + Knowledge Graph (PPR for entity-focused queries)
- **Temporal fact management** — mark facts as superseded, track validity windows with `--include-historical` flag
- **Memory consolidation** — detect stale edges via confidence decay, surface old memories for summarization
- **Entity resolution** — auto-dedup via dual-threshold (vector + name similarity), manual merge with audit trail
- **Community detection** — cluster detection and analysis integrated into consolidation pipeline
- **Zero infrastructure** — no Docker, no ChromaDB, no external servers
- **Fully offline** — FastEmbed ONNX embedding, no API keys needed
- **Agent-driven KG** — agent extracts entities and indexes them (no built-in LLM)
- **CLI + Python API** — works with any agent that runs shell commands
- **Built-in personas** — companion (emotion tracking) and mentor (decision tracking)
- **Multilingual** — 100+ languages via multilingual-e5-large

## Getting started

Install and connect to your agent in 3 commands:

```bash
pipx install "kioku-lite[cli]"
kioku-lite init --global
kioku-lite install-profile companion
```

That's it. Your agent now has long-term memory with Knowledge Graph.

Full setup guides for each agent type:
- **[Claude Code / Cursor / Windsurf](https://phuc-nt.github.io/kioku-lite-landing/agent-setup.html)** — copy-paste the guide, agent does the rest
- **[OpenClaw](https://phuc-nt.github.io/kioku-lite-landing/openclaw-setup.html)** — auto-generates SOUL.md + TOOLS.md

## Architecture

```
Agent (Claude Code, Cursor, …)
  │
  ├─ save "..."                ──→  SQLite FTS5 + sqlite-vec + Markdown backup
  ├─ kg-index <hash>           ──→  GraphStore (nodes, edges, aliases, temporal validity)
  ├─ kg-invalidate / merge      ──→  Temporal tracking, entity resolution audit
  ├─ search "..."              ──→  BM25 ∪ Vector ∪ PPR(entities) → RRF rerank
  ├─ consolidate               ──→  Decay stale edges, detect duplicates, surface old memories
  └─ clusters / cluster        ──→  Community detection & analysis
```

**Temporal & Dedup Layers:**
- Temporal validity: `kg_edges.valid_from` / `valid_until` — query with `--include-historical`
- Entity consolidation: auto-dedup via dual threshold (0.98 vector sim + 0.85 name sim)
- Confidence decay: `weight * 0.5^(days/half_life)` for stale edge detection
- Cluster-aware consolidation: groups related entities during dedup

> kioku-lite **never calls an LLM**. The agent extracts entities from its own context, keeping the memory engine 100% local and LLM-agnostic.

Deep dive: [System Architecture](https://phuc-nt.github.io/kioku-lite-landing/blog.html#system-architecture) · [Write Pipeline](https://phuc-nt.github.io/kioku-lite-landing/blog.html#write-save-kg-index) · [Search Pipeline](https://phuc-nt.github.io/kioku-lite-landing/blog.html#search-architecture) · [KG Open Schema](https://phuc-nt.github.io/kioku-lite-landing/blog.html#kg-open-schema)

## Development

```bash
git clone https://github.com/phuc-nt/kioku-agent-kit-lite
cd kioku-agent-kit-lite
pip install -e ".[cli,dev]"
pytest
```

## License

[MIT](LICENSE) © 2026 Phuc Nguyen

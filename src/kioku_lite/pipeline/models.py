"""Shared data classes for Kioku Lite pipeline and search layers."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Memory ─────────────────────────────────────────────────────────────────────

@dataclass
class FTSResult:
    """A result from FTS5 keyword search."""
    rowid: int
    content: str
    date: str
    mood: str
    timestamp: str
    rank: float
    content_hash: str = ""
    event_time: str = ""


# ── Knowledge Graph ─────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    name: str
    type: str
    mention_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    confidence: float = 1.0


@dataclass
class GraphEdge:
    """An edge (relationship) in the knowledge graph."""
    source: str
    target: str
    rel_type: str
    weight: float = 0.5
    evidence: str = ""
    source_hash: str = ""   # links back to SQLite memories for hydration
    valid_from: str = ""           # when fact became true ('' = unknown)
    valid_until: str | None = None  # when fact stopped being true (None = current)
    last_reinforced: str = ""      # when edge was last re-confirmed ('' = legacy/unknown)


@dataclass
class GraphSearchResult:
    """Result from a graph traversal or path search."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)

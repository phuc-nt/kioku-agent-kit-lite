"""Community/Cluster Detection — connected components on the KG.

Pure Python BFS-based connected components. No external dependencies.

Algorithm:
  1. Load all active edges from kg_edges (valid_until IS NULL by default)
  2. Build adjacency list from source+target node names
  3. BFS to find connected components
  4. For each component, suggest a label from the most common entity type

Usage:
    communities = find_communities(conn)
    # → [Community(id=0, label="PERSON", entities=["Alice","Bob"], edge_count=3), ...]
"""

from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Community:
    """A single cluster of connected entities."""

    id: int
    label: str
    entities: list[str] = field(default_factory=list)
    edge_count: int = 0


# ── Core algorithm ──────────────────────────────────────────────────────────────


def _build_adjacency(conn: sqlite3.Connection, include_historical: bool) -> dict[str, set[str]]:
    """Load edges from DB and build an undirected adjacency list (lowercased keys)."""
    cur = conn.cursor()
    validity_clause = "" if include_historical else "WHERE valid_until IS NULL"
    cur.execute(f"SELECT source, target FROM kg_edges {validity_clause}")
    adj: dict[str, set[str]] = {}
    for src, tgt in cur.fetchall():
        adj.setdefault(src.lower(), set()).add(tgt)
        adj.setdefault(tgt.lower(), set()).add(src)
    return adj


def _all_nodes(conn: sqlite3.Connection) -> list[str]:
    """Return all entity names from kg_nodes."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM kg_nodes")
    return [row[0] for row in cur.fetchall()]


def _bfs_component(start: str, adj: dict[str, set[str]], visited: set[str]) -> list[str]:
    """BFS from start, returning all reachable node names."""
    component: list[str] = []
    queue: deque[str] = deque([start])
    visited.add(start.lower())
    while queue:
        node = queue.popleft()
        component.append(node)
        for neighbor in adj.get(node.lower(), set()):
            if neighbor.lower() not in visited:
                visited.add(neighbor.lower())
                queue.append(neighbor)
    return component


def _suggest_label(entities: list[str], conn: sqlite3.Connection) -> str:
    """Suggest a cluster label from the most common entity type in the component."""
    if not entities:
        return "UNKNOWN"
    cur = conn.cursor()
    placeholders = ",".join("?" * len(entities))
    cur.execute(
        f"SELECT type, COUNT(*) as cnt FROM kg_nodes WHERE name IN ({placeholders}) "
        f"AND type != '' GROUP BY type ORDER BY cnt DESC LIMIT 1",
        entities,
    )
    row = cur.fetchone()
    return row[0] if row else "UNKNOWN"


def find_communities(
    conn: sqlite3.Connection,
    include_historical: bool = False,
) -> list[Community]:
    """Find connected components in the KG.

    Args:
        conn: SQLite connection (shared KiokuDB connection).
        include_historical: If True, include edges with valid_until set.

    Returns:
        List of Community objects, sorted by entity count descending.
    """
    adj = _build_adjacency(conn, include_historical)
    all_node_names = _all_nodes(conn)

    # Ensure isolated nodes (no edges) are included
    for name in all_node_names:
        if name.lower() not in adj:
            adj[name.lower()] = set()

    visited: set[str] = set()
    communities: list[Community] = []
    component_id = 0

    for node in all_node_names:
        if node.lower() in visited:
            continue
        component = _bfs_component(node, adj, visited)
        if not component:
            continue

        # Count edges within this component
        member_set = {n.lower() for n in component}
        edge_count = sum(
            1
            for n in component
            for neighbor in adj.get(n.lower(), set())
            if neighbor.lower() in member_set and neighbor.lower() > n.lower()
        )

        label = _suggest_label(component, conn)
        communities.append(
            Community(
                id=component_id,
                label=label,
                entities=component,
                edge_count=edge_count,
            )
        )
        component_id += 1

    # Sort by size descending so the largest cluster is first
    communities.sort(key=lambda c: len(c.entities), reverse=True)
    # Re-assign IDs after sort
    for i, c in enumerate(communities):
        c.id = i

    return communities

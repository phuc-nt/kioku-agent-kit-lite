"""Personalized PageRank (PPR) on knowledge graph edges.

Power iteration seeded by query entities — follows HippoRAG approach.
No external deps: pure Python dict operations on SQLite adjacency.
"""

from __future__ import annotations

import sqlite3

from kioku_lite.search.bm25 import SearchResult


def personalized_pagerank(
    conn: sqlite3.Connection,
    seeds: list[str],
    damping: float = 0.85,
    iterations: int = 10,
    include_historical: bool = False,
) -> dict[str, float]:
    """Power-iteration PPR seeded by `seeds`.

    Returns entity_name → score mapping for all reachable nodes.
    Empty dict when graph has no edges or seeds is empty.
    """
    if not seeds:
        return {}

    validity_clause = "" if include_historical else "AND valid_until IS NULL"
    cur = conn.cursor()
    cur.execute(
        f"SELECT source, target, weight FROM kg_edges WHERE 1=1 {validity_clause}"
    )
    rows = cur.fetchall()
    if not rows:
        return {}

    # Build undirected adjacency: node → [(neighbor, weight)]
    adj: dict[str, list[tuple[str, float]]] = {}
    for src, tgt, w in rows:
        s, t = src.lower(), tgt.lower()
        weight = float(w) if w else 1.0
        adj.setdefault(s, []).append((t, weight))
        adj.setdefault(t, []).append((s, weight))

    # Normalize outgoing weights per node so they sum to 1.0
    out_weight: dict[str, float] = {}
    for node, neighbors in adj.items():
        total = sum(w for _, w in neighbors)
        out_weight[node] = total if total > 0 else 1.0

    # Collect all known nodes
    all_nodes = set(adj.keys())

    # Teleport vector: uniform over (case-insensitive) seeds present in graph
    seed_lower = [s.lower() for s in seeds]
    active_seeds = [s for s in seed_lower if s in all_nodes]
    if not active_seeds:
        # Seeds not in graph — return zero scores (only teleport, no propagation)
        # Assign uniform teleport score to seeds even if disconnected
        return {s: 1.0 / len(seeds) for s in seed_lower}

    teleport_weight = 1.0 / len(active_seeds)
    teleport: dict[str, float] = {s: teleport_weight for s in active_seeds}

    # Initialize PPR scores: uniform over all nodes
    n = len(all_nodes)
    scores: dict[str, float] = {node: 1.0 / n for node in all_nodes}

    # Power iteration
    for _ in range(iterations):
        new_scores: dict[str, float] = {}
        # Teleport component
        for node in all_nodes:
            new_scores[node] = (1.0 - damping) * teleport.get(node, 0.0)

        # Transition component: for each node push score to neighbors
        for node, neighbors in adj.items():
            node_score = scores[node]
            total_out = out_weight[node]
            for neighbor, w in neighbors:
                new_scores[neighbor] = new_scores.get(neighbor, 0.0) + damping * node_score * (w / total_out)

        scores = new_scores

    return {node: score for node, score in scores.items() if score > 0}


def ppr_to_results(
    conn: sqlite3.Connection,
    ppr_scores: dict[str, float],
    limit: int = 20,
    include_historical: bool = False,
) -> list[SearchResult]:
    """Convert PPR scores → SearchResult list via kg_edges.source_hash.

    Each result score = edge.weight * ppr_score[entity].
    Deduplicates by source_hash (keeps highest score).
    """
    if not ppr_scores:
        return []

    # Sort by score descending, take top candidates
    top_entities = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:limit * 2]

    validity_clause = "" if include_historical else "AND valid_until IS NULL"
    cur = conn.cursor()

    # Accumulate scores per source_hash: a memory reachable from multiple seeds
    # gets additive score (shared memory ranks higher than single-seed memory).
    hash_scores: dict[str, float] = {}
    hash_content: dict[str, tuple[str, str]] = {}  # hash → (evidence, source_hash)
    orphans: dict[str, SearchResult] = {}
    seen_edge_entity: set[tuple[str, str]] = set()  # (source_hash, entity) dedup

    for entity, ppr_score in top_entities:
        cur.execute(
            f"""
            SELECT source, target, weight, evidence, source_hash
            FROM kg_edges
            WHERE (source = ? COLLATE NOCASE OR target = ? COLLATE NOCASE)
            {validity_clause}
            """,
            (entity, entity),
        )
        for row in cur.fetchall():
            _src, _tgt, w, evidence, source_hash = row
            edge_weight = float(w) if w else 1.0
            if source_hash:
                key = (source_hash, entity)
                if key not in seen_edge_entity:
                    seen_edge_entity.add(key)
                    hash_scores[source_hash] = hash_scores.get(source_hash, 0.0) + edge_weight * ppr_score
                    hash_content.setdefault(source_hash, (evidence or "", source_hash))
            else:
                ev_key = evidence or ""
                if ev_key and ev_key not in orphans:
                    orphans[ev_key] = SearchResult(
                        content=evidence or "",
                        date="", mood="", timestamp="",
                        score=edge_weight * ppr_score,
                        source="graph",
                        content_hash="",
                    )

    seen_hashes = {
        sh: SearchResult(
            content=hash_content[sh][0],
            date="", mood="", timestamp="",
            score=score,
            source="graph",
            content_hash=sh,
        )
        for sh, score in hash_scores.items()
    }

    results = list(seen_hashes.values()) + list(orphans.values())
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]

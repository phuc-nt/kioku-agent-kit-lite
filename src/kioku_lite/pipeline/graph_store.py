"""Knowledge Graph store — SQLite-backed entity nodes, edges, and traversal.

Tables: kg_nodes, kg_edges, kg_aliases
No Cypher, no FalkorDB — pure Python BFS on top of SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import deque

from kioku_lite.pipeline.models import GraphEdge, GraphNode, GraphSearchResult

log = logging.getLogger(__name__)


class GraphStore:
    """Knowledge graph backed by three SQLite tables.

    Shares a sqlite3.Connection with MemoryStore (created by KiokuDB).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ── Upsert ─────────────────────────────────────────────────────────────────

    def upsert_node(self, name: str, entity_type: str, date: str, confidence: float = 1.0) -> None:
        """Insert or update entity node, incrementing mention_count.

        Confidence uses MAX strategy: re-upserting with a higher value raises it,
        but never lowers it.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO kg_nodes (name, type, mention_count, first_seen, last_seen, confidence)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                mention_count = mention_count + 1,
                last_seen = excluded.last_seen,
                type = CASE WHEN excluded.type != '' THEN excluded.type ELSE type END,
                confidence = MAX(confidence, excluded.confidence)
            """,
            (name, entity_type, date, date, confidence),
        )
        self.conn.commit()

    def upsert_edge(
        self,
        source: str,
        target: str,
        rel_type: str,
        weight: float,
        evidence: str,
        source_hash: str,
        event_time: str = "",
        valid_from: str = "",
    ) -> None:
        """Insert or update relationship edge, averaging weights on conflict.

        ON CONFLICT does NOT touch valid_from/valid_until — re-indexing same
        fact should not un-invalidate it.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO kg_edges
                (source, target, rel_type, weight, evidence, source_hash, event_time, valid_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, target, rel_type) DO UPDATE SET
                weight = (weight + excluded.weight) / 2,
                evidence = excluded.evidence,
                source_hash = excluded.source_hash,
                event_time = excluded.event_time
            """,
            (source, target, rel_type, weight, evidence, source_hash, event_time, valid_from),
        )
        self.conn.commit()

    def invalidate_edge(
        self,
        valid_until: str,
        source: str | None = None,
        target: str | None = None,
        rel_type: str | None = None,
    ) -> int:
        """Mark edge(s) as no longer valid. Returns count of rows updated."""
        clauses: list[str] = []
        params: list[str] = [valid_until]  # SET valid_until = ? comes first
        if source:
            clauses.append("source = ? COLLATE NOCASE")
            params.append(source)
        if target:
            clauses.append("target = ? COLLATE NOCASE")
            params.append(target)
        if rel_type:
            clauses.append("rel_type = ? COLLATE NOCASE")
            params.append(rel_type)
        if not clauses:
            return 0
        sql = f"UPDATE kg_edges SET valid_until = ? WHERE {' AND '.join(clauses)}"
        cur = self.conn.cursor()
        cur.execute(sql, params)
        self.conn.commit()
        return cur.rowcount

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register an alias → canonical SAME_AS mapping. Skips if alias == canonical."""
        if alias.strip().lower() == canonical.strip().lower():
            log.debug("Skipping self-alias: '%s' == '%s'", alias, canonical)
            return
        cur = self.conn.cursor()
        for name in (alias, canonical):
            cur.execute(
                "INSERT OR IGNORE INTO kg_nodes (name, type, mention_count) VALUES (?, 'PERSON', 0)",
                (name,),
            )
        cur.execute("UPDATE kg_nodes SET is_canonical = 1 WHERE name = ?", (canonical,))
        cur.execute(
            "INSERT OR IGNORE INTO kg_aliases (alias, canonical) VALUES (?, ?)",
            (alias, canonical),
        )
        self.conn.commit()
        log.info("Linked alias '%s' → canonical '%s'", alias, canonical)

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_canonical_entities(self, limit: int = 50) -> list[dict]:
        """Top entities by mention_count, with their aliases included."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT n.name, n.type, n.mention_count, n.confidence,
                   GROUP_CONCAT(a.alias, '|||') AS aliases
            FROM kg_nodes n
            LEFT JOIN kg_aliases a ON a.canonical = n.name COLLATE NOCASE
            GROUP BY n.name
            ORDER BY n.mention_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "name": r[0],
                "type": r[1] or "",
                "mentions": r[2] or 0,
                "confidence": r[3] if r[3] is not None else 1.0,
                "aliases": [x for x in (r[4] or "").split("|||") if x],
            }
            for r in cur.fetchall()
        ]

    def search_nodes(self, query: str, limit: int = 30, min_confidence: float = 0.0) -> list[GraphNode]:
        """Case-insensitive substring search, re-ranked by match quality.

        min_confidence: if > 0, excludes nodes with confidence below threshold.
        """
        cur = self.conn.cursor()
        if min_confidence > 0.0:
            cur.execute(
                """
                SELECT name, type, mention_count, first_seen, last_seen, confidence
                FROM kg_nodes
                WHERE name LIKE ? COLLATE NOCASE AND confidence >= ?
                ORDER BY mention_count DESC
                LIMIT ?
                """,
                (f"%{query}%", min_confidence, limit),
            )
        else:
            cur.execute(
                """
                SELECT name, type, mention_count, first_seen, last_seen, confidence
                FROM kg_nodes
                WHERE name LIKE ? COLLATE NOCASE
                ORDER BY mention_count DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )
        nodes = [
            GraphNode(
                name=r[0], type=r[1] or "", mention_count=r[2] or 0,
                first_seen=r[3] or "", last_seen=r[4] or "",
                confidence=r[5] if r[5] is not None else 1.0,
            )
            for r in cur.fetchall()
        ]
        return self._rerank_nodes(nodes, query)

    @staticmethod
    def _rerank_nodes(nodes: list[GraphNode], query: str) -> list[GraphNode]:
        """Re-rank nodes: exact > starts-with > whole-word > substring."""
        q = query.lower()
        is_single = " " not in q.strip()

        def _key(n: GraphNode) -> tuple:
            nl = n.name.lower()
            if nl == q:
                return (0, -n.mention_count)
            if nl.startswith(q + " ") or (not is_single and nl.endswith(" " + q)):
                return (1, -n.mention_count)
            if q + " " in nl or " " + q in nl:
                return (2, -n.mention_count)
            if is_single and nl.endswith(" " + q):
                return (2, -n.mention_count)
            return (3, -n.mention_count)

        nodes.sort(key=_key)
        return nodes

    # ── Traversal ──────────────────────────────────────────────────────────────

    def get_top_entity(self) -> str | None:
        """Return the entity name with the highest mention_count (self/hub node)."""
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM kg_nodes ORDER BY mention_count DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None

    def get_degree(self, entity_name: str) -> int:
        """Count the number of edges connected to an entity (in + out)."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM kg_edges "
            "WHERE source = ? COLLATE NOCASE OR target = ? COLLATE NOCASE",
            (entity_name, entity_name),
        )
        row = cur.fetchone()
        return row[0] if row else 0

    def traverse(
        self,
        entity_name: str,
        max_hops: int = 2,
        limit: int = 20,
        include_historical: bool = False,
    ) -> GraphSearchResult:
        """BFS traversal from seed entity, following SAME_AS aliases.

        Uses adaptive hop limit (Task 1C): hub nodes with degree > 15 are
        capped at 1 hop to avoid returning the majority of the DB.
        By default only follows currently-valid edges (valid_until IS NULL).
        """
        seeds = self._resolve_names(entity_name)

        # Adaptive hop limit: high-degree (hub) nodes get only 1 hop
        degree = self.get_degree(entity_name)
        effective_hops = 1 if degree > 15 else max_hops

        nodes_map: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        seen: set[str] = set()

        for seed in seeds:
            self._bfs(seed, effective_hops, limit, nodes_map, edges, seen, include_historical)

        return GraphSearchResult(nodes=list(nodes_map.values()), edges=edges[:limit])

    def _resolve_names(self, name: str) -> list[str]:
        """Expand a name to include its canonical and all known aliases."""
        names: dict[str, str] = {name.lower(): name}
        cur = self.conn.cursor()
        # alias → find canonical
        cur.execute("SELECT canonical FROM kg_aliases WHERE alias = ? COLLATE NOCASE", (name,))
        for row in cur.fetchall():
            names[row[0].lower()] = row[0]
        # canonical → find all aliases
        for n in list(names.values()):
            cur.execute("SELECT alias FROM kg_aliases WHERE canonical = ? COLLATE NOCASE", (n,))
            for row in cur.fetchall():
                if row[0].lower() not in names:
                    names[row[0].lower()] = row[0]
        return list(names.values())

    def _bfs(
        self,
        start: str,
        max_hops: int,
        limit: int,
        nodes_map: dict[str, GraphNode],
        edges: list[GraphEdge],
        seen: set[str],
        include_historical: bool = False,
    ) -> None:
        """BFS from start, collecting up to `limit` edges.

        By default only traverses valid edges (valid_until IS NULL).
        """
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        visited: set[str] = {start.lower()}
        cur = self.conn.cursor()

        validity_clause = "" if include_historical else "AND valid_until IS NULL"

        while queue and len(edges) < limit:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            cur.execute(
                f"""
                SELECT source, target, rel_type, weight, evidence, source_hash,
                       valid_from, valid_until
                FROM kg_edges
                WHERE (source = ? COLLATE NOCASE OR target = ? COLLATE NOCASE)
                    {validity_clause}
                ORDER BY weight DESC LIMIT 50
                """,
                (current, current),
            )
            for row in cur.fetchall():
                src, tgt, rel, weight, evidence, src_hash, vf, vu = row
                key = f"{src.lower()}|{tgt.lower()}|{rel}|{src_hash}"
                if key not in seen:
                    seen.add(key)
                    edges.append(GraphEdge(
                        source=src, target=tgt, rel_type=rel,
                        weight=weight, evidence=evidence or "", source_hash=src_hash or "",
                        valid_from=vf or "", valid_until=vu,
                    ))
                    nodes_map[src.lower()] = GraphNode(name=src, type="")
                    nodes_map[tgt.lower()] = GraphNode(name=tgt, type="")
                    neighbor = tgt if src.lower() == current.lower() else src
                    if neighbor.lower() not in visited:
                        visited.add(neighbor.lower())
                        queue.append((neighbor, depth + 1))

        # Enrich nodes with metadata
        self._enrich_nodes(nodes_map)

    def _enrich_nodes(self, nodes_map: dict[str, GraphNode]) -> None:
        """Fill in type/mention_count/dates/confidence for collected nodes."""
        cur = self.conn.cursor()
        for key, node in list(nodes_map.items()):
            cur.execute(
                "SELECT name, type, mention_count, first_seen, last_seen, confidence "
                "FROM kg_nodes WHERE name = ? COLLATE NOCASE",
                (node.name,),
            )
            row = cur.fetchone()
            if row:
                nodes_map[key] = GraphNode(
                    name=row[0], type=row[1] or "", mention_count=row[2] or 0,
                    first_seen=row[3] or "", last_seen=row[4] or "",
                    confidence=row[5] if row[5] is not None else 1.0,
                )

    # ── Merge ──────────────────────────────────────────────────────────────────

    def merge_entities(
        self,
        source: str,
        target: str,
        merge_type: str = "auto",
        vector_sim: float = 0.0,
        name_sim: float = 0.0,
    ) -> dict:
        """Atomically merge source entity into target.

        - Re-points all edges from source to target
        - Deduplicates edges after re-pointing (keep lowest id per src+tgt+rel_type)
        - Accumulates mention_count from source into target
        - Registers source as alias of target
        - Logs merge to kg_merge_log
        - Deletes source node

        Returns status dict. No-ops if source == target (case-insensitive) or
        source does not exist.
        """
        if source.strip().lower() == target.strip().lower():
            return {"status": "skipped", "reason": "self-merge"}

        cur = self.conn.cursor()
        cur.execute(
            "SELECT mention_count FROM kg_nodes WHERE name = ? COLLATE NOCASE",
            (source,),
        )
        src_row = cur.fetchone()
        if not src_row:
            return {"status": "skipped", "reason": "source not found"}

        cur.execute(
            "SELECT name FROM kg_nodes WHERE name = ? COLLATE NOCASE",
            (target,),
        )
        if not cur.fetchone():
            return {"status": "skipped", "reason": "target not found"}

        src_mentions = src_row[0] or 0

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cur.execute("SAVEPOINT merge_entity")
        try:
            # Re-point edges where source is the source
            cur.execute(
                "UPDATE kg_edges SET source = ? WHERE source = ? COLLATE NOCASE",
                (target, source),
            )
            # Re-point edges where source is the target
            cur.execute(
                "UPDATE kg_edges SET target = ? WHERE target = ? COLLATE NOCASE",
                (target, source),
            )
            # Remove self-loops created by re-pointing (source->source becomes target->target)
            cur.execute(
                "DELETE FROM kg_edges WHERE source = ? AND target = ? COLLATE NOCASE",
                (target, target),
            )
            # Dedup edges: keep lowest id per (source, target, rel_type)
            cur.execute(
                """
                DELETE FROM kg_edges WHERE id NOT IN (
                    SELECT MIN(id) FROM kg_edges GROUP BY source, target, rel_type
                )
                """
            )
            # Accumulate mention_count into target
            cur.execute(
                "UPDATE kg_nodes SET mention_count = mention_count + ? WHERE name = ? COLLATE NOCASE",
                (src_mentions, target),
            )
            # Register source as alias of target (reuse existing method logic inline
            # to avoid commit inside SAVEPOINT)
            cur.execute(
                "INSERT OR IGNORE INTO kg_aliases (alias, canonical) VALUES (?, ?)",
                (source, target),
            )
            cur.execute(
                "UPDATE kg_nodes SET is_canonical = 1 WHERE name = ? COLLATE NOCASE",
                (target,),
            )
            # Log merge
            cur.execute(
                """
                INSERT INTO kg_merge_log
                    (source_name, target_name, merge_type, vector_sim, name_sim, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, target, merge_type, vector_sim, name_sim, ts),
            )
            # Delete source node
            cur.execute(
                "DELETE FROM kg_nodes WHERE name = ? COLLATE NOCASE",
                (source,),
            )
            cur.execute("RELEASE merge_entity")
            self.conn.commit()
        except Exception:
            cur.execute("ROLLBACK TO merge_entity")
            cur.execute("RELEASE merge_entity")
            raise

        log.info("Merged '%s' → '%s' (%s, vec=%.3f, name=%.3f)",
                 source, target, merge_type, vector_sim, name_sim)
        return {"status": "merged", "source": source, "target": target,
                "merge_type": merge_type, "vector_sim": vector_sim, "name_sim": name_sim}

    # ── Export ─────────────────────────────────────────────────────────────────

    def get_all_nodes(self) -> list[dict]:
        """Return all nodes with metadata for graph export."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT n.name, n.type, n.mention_count, n.first_seen, n.last_seen,
                   n.confidence, GROUP_CONCAT(a.alias, '|||') AS aliases
            FROM kg_nodes n
            LEFT JOIN kg_aliases a ON a.canonical = n.name COLLATE NOCASE
            GROUP BY n.name
            ORDER BY n.mention_count DESC
            """
        )
        return [
            {
                "id": r[0],
                "name": r[0],
                "type": r[1] or "UNKNOWN",
                "mentions": r[2] or 0,
                "first_seen": r[3] or "",
                "last_seen": r[4] or "",
                "confidence": r[5] if r[5] is not None else 1.0,
                "aliases": [x for x in (r[6] or "").split("|||") if x],
            }
            for r in cur.fetchall()
        ]

    def get_all_edges(self) -> list[dict]:
        """Return all edges for graph export (includes temporal validity)."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT source, target, rel_type, weight, evidence, event_time,
                   valid_from, valid_until
            FROM kg_edges
            ORDER BY weight DESC
            """
        )
        return [
            {
                "source": r[0],
                "target": r[1],
                "relation": r[2] or "",
                "weight": r[3] or 0.5,
                "evidence": r[4] or "",
                "event_time": r[5] or "",
                "valid_from": r[6] or "",
                "valid_until": r[7],
            }
            for r in cur.fetchall()
        ]

    def find_path(
        self, source: str, target: str, include_historical: bool = False,
    ) -> GraphSearchResult:
        """BFS shortest path between two entities (undirected)."""
        cur = self.conn.cursor()
        validity_clause = "" if include_historical else "WHERE valid_until IS NULL"
        cur.execute(
            f"SELECT source, target, rel_type, evidence, source_hash FROM kg_edges {validity_clause}"
        )
        adj: dict[str, list[tuple[str, str, str, str]]] = {}
        for row in cur.fetchall():
            s, t, rel, ev, sh = row[0], row[1], row[2] or "", row[3] or "", row[4] or ""
            adj.setdefault(s.lower(), []).append((t, rel, ev, sh))
            adj.setdefault(t.lower(), []).append((s, rel, ev, sh))

        queue: deque[tuple[str, list[str]]] = deque([(source.lower(), [source])])
        visited = {source.lower()}

        while queue:
            current, path = queue.popleft()
            if current == target.lower():
                nodes = [GraphNode(name=n, type="") for n in path]
                edges = []
                for i in range(len(path) - 1):
                    a, b = path[i].lower(), path[i + 1].lower()
                    for nb, rel, ev, sh in adj.get(a, []):
                        if nb.lower() == b:
                            edges.append(GraphEdge(source=path[i], target=path[i + 1], rel_type=rel, evidence=ev, source_hash=sh))
                            break
                return GraphSearchResult(nodes=nodes, edges=edges, paths=[path])
            for neighbor, _, _, _ in adj.get(current, []):
                if neighbor.lower() not in visited:
                    visited.add(neighbor.lower())
                    queue.append((neighbor.lower(), path + [neighbor]))

        return GraphSearchResult()

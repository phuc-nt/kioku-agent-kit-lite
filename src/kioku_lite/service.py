"""Kioku Lite Service — single source of truth, zero cloud LLM dependency.

Architecture difference from kioku-agent-kit:
  - Entity extraction is NOT done here.
  - The agent (Claude Code / OpenClaw) calls `kg-index` separately,
    providing its own entities and relationships extracted from conversation context.
  - This keeps kioku-lite 100% local: SQLite + FastEmbed (ONNX), no API calls.
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# Suppress fastembed pooling-method change warning (cosmetic, not functional)
warnings.filterwarnings(
    "ignore",
    message=".*mean pooling.*CLS embedding.*",
    category=UserWarning,
)

from kioku_lite.config import Settings
from kioku_lite.pipeline.db import KiokuDB
from kioku_lite.pipeline.dedup import DedupEngine
from kioku_lite.pipeline.embedder import make_embedder
from kioku_lite.search.bm25 import bm25_search
from kioku_lite.search.graph import graph_search
from kioku_lite.search.reranker import rrf_rerank
from kioku_lite.search.semantic import vector_search
from kioku_lite.storage.markdown import save_entry

log = logging.getLogger(__name__)
JST = timezone(timedelta(hours=7))


# ── Input data classes (passed by agent via CLI) ───────────────────────────────

@dataclass
class EntityInput:
    name: str
    type: str = "TOPIC"
    confidence: float = 1.0


@dataclass
class RelationshipInput:
    source: str
    target: str
    rel_type: str = "TOPICAL"
    weight: float = 0.5
    evidence: str = ""


# ── Service ────────────────────────────────────────────────────────────────────

class KiokuLiteService:
    """Core business logic for Kioku Lite. Used by CLI only (no MCP)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_dirs()

        self.db = KiokuDB(self.settings.db_path, embed_dim=self.settings.embed_dim)
        ollama_model = getattr(self.settings, "ollama_model", "bge-m3")
        ollama_url   = getattr(self.settings, "ollama_base_url", "http://localhost:11434")
        self.embedder = make_embedder(
            provider=self.settings.embed_provider,
            model=ollama_model if self.settings.embed_provider == "ollama" else self.settings.embed_model,
            dim=self.settings.embed_dim,
            base_url=ollama_url,
        )

    # ── save_memory ────────────────────────────────────────────────────────────

    def save_memory(
        self,
        text: str,
        mood: str | None = None,
        tags: list[str] | None = None,
        event_time: str | None = None,
    ) -> dict:
        """Save a memory entry.

        Stores to Markdown (source of truth) + FTS5 (BM25) + sqlite-vec (vector).
        Does NOT extract entities — agent calls `kg_index` separately.

        Returns content_hash so agent can call kg_index immediately after.
        """
        date = datetime.now(JST).strftime("%Y-%m-%d")
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        # 1. Markdown (source of truth)
        entry = save_entry(
            self.settings.memory_dir,
            text,
            mood=mood,
            tags=tags,
            event_time=event_time,
        )

        # 2. FTS5 (BM25)
        self.db.memory.insert(
            content=text,
            date=date,
            timestamp=entry.timestamp,
            mood=mood or "",
            tags=tags,
            content_hash=content_hash,
            event_time=event_time or "",
        )

        # 3. Vector (sqlite-vec + FastEmbed)
        try:
            embedding = self.embedder.embed(text)
            self.db.memory.insert_vector(content_hash, embedding)
            vector_indexed = self.db.vec_enabled
        except Exception as e:
            log.warning("Vector indexing failed: %s", e)
            vector_indexed = False

        return {
            "status": "saved",
            "content_hash": content_hash,
            "timestamp": entry.timestamp,
            "date": date,
            "mood": mood,
            "tags": tags,
            "event_time": event_time,
            "vector_indexed": vector_indexed,
            "hint": "Run `kg-index` to add knowledge graph entries for this memory.",
        }

    # ── kg_index ───────────────────────────────────────────────────────────────

    def kg_index(
        self,
        content_hash: str,
        entities: list[EntityInput],
        relationships: list[RelationshipInput],
        event_time: str | None = None,
    ) -> dict:
        """Index entities and relationships for a previously saved memory.

        Called by the agent AFTER save_memory, providing its own extraction.
        The agent already has full context — no need for kioku to call any LLM.
        """
        date = datetime.now(JST).strftime("%Y-%m-%d")
        et = event_time or date

        for entity in entities:
            self.db.graph.upsert_node(entity.name, entity.type, date, confidence=entity.confidence)

        # Dedup: find and auto-merge near-duplicate entities
        dedup_engine = DedupEngine()
        all_auto_merged = []
        all_candidates = []
        for entity in entities:
            try:
                dedup_result = dedup_engine.find_similar(
                    entity.name, self.embedder, self.db.conn
                )
                for action in dedup_result.auto_merged:
                    merge_out = self.db.graph.merge_entities(
                        action.source_name, action.target_name,
                        "auto", action.vector_sim, action.name_sim,
                    )
                    if merge_out.get("status") == "merged":
                        all_auto_merged.append({
                            "source": action.source_name,
                            "target": action.target_name,
                            "vector_sim": round(action.vector_sim, 4),
                            "name_sim": round(action.name_sim, 4),
                        })
                for cand in dedup_result.candidates:
                    all_candidates.append({
                        "source": cand.source_name,
                        "target": cand.target_name,
                        "vector_sim": round(cand.vector_sim, 4),
                        "name_sim": round(cand.name_sim, 4),
                    })
            except Exception as e:
                log.warning("Dedup failed for entity '%s': %s", entity.name, e)

        for rel in relationships:
            self.db.graph.upsert_edge(
                source=rel.source,
                target=rel.target,
                rel_type=rel.rel_type,
                weight=rel.weight,
                evidence=rel.evidence,
                source_hash=content_hash,
                event_time=et,
                valid_from=et,
            )

        log.info(
            "KG indexed: %d entities, %d relationships for hash %s",
            len(entities), len(relationships), content_hash[:8],
        )
        return {
            "status": "indexed",
            "content_hash": content_hash,
            "entities_added": len(entities),
            "relationships_added": len(relationships),
            "auto_merged": all_auto_merged,
            "dedup_candidates": all_candidates,
        }

    # ── kg_invalidate ──────────────────────────────────────────────────────────

    def kg_invalidate(
        self,
        source: str,
        target: str,
        rel_type: str | None = None,
        valid_until: str | None = None,
        reason: str = "",
    ) -> dict:
        """Mark edge(s) as no longer valid (superseded by newer facts)."""
        date = valid_until or datetime.now(JST).strftime("%Y-%m-%d")
        count = self.db.graph.invalidate_edge(
            valid_until=date, source=source, target=target, rel_type=rel_type,
        )
        return {
            "status": "invalidated" if count > 0 else "no_match",
            "edges_updated": count,
            "valid_until": date,
            "reason": reason,
        }

    # ── kg_alias ───────────────────────────────────────────────────────────────

    def kg_alias(self, canonical: str, aliases: list[str]) -> dict:
        """Register SAME_AS aliases for a canonical entity name."""
        added = []
        for alias in aliases:
            if alias != canonical:
                self.db.graph.add_alias(alias, canonical)
                added.append(alias)
        return {"status": "ok", "canonical": canonical, "aliases_added": added}

    # ── dedup ──────────────────────────────────────────────────────────────────

    def dedup_scan(self, auto_merge: bool = False) -> dict:
        """Scan all entities for near-duplicates.

        If auto_merge=True, automatically merges pairs that meet both
        vec_auto + name_auto thresholds.
        """
        engine = DedupEngine()
        candidates = engine.scan_all(self.db.conn, self.embedder)
        merged = []
        remaining_candidates = []

        for cand in candidates:
            if (auto_merge
                    and cand.vector_sim >= engine.vec_auto
                    and cand.name_sim >= engine.name_auto):
                result = self.db.graph.merge_entities(
                    cand.source_name, cand.target_name,
                    "auto", cand.vector_sim, cand.name_sim,
                )
                if result.get("status") == "merged":
                    merged.append({
                        "source": cand.source_name,
                        "target": cand.target_name,
                        "vector_sim": round(cand.vector_sim, 4),
                        "name_sim": round(cand.name_sim, 4),
                    })
                    continue
            remaining_candidates.append({
                "source": cand.source_name,
                "target": cand.target_name,
                "vector_sim": round(cand.vector_sim, 4),
                "name_sim": round(cand.name_sim, 4),
            })

        return {
            "candidates": remaining_candidates,
            "auto_merged": merged,
            "total_scanned": len(candidates),
        }

    def merge_entities(self, source: str, target: str) -> dict:
        """Manually merge source entity into target (alias + edge re-point)."""
        return self.db.graph.merge_entities(source, target, merge_type="manual")

    # ── search_memories ────────────────────────────────────────────────────────

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        date_from: str | None = None,
        date_to: str | None = None,
        entities: list[str] | None = None,
        include_historical: bool = False,
    ) -> dict:
        """Tri-hybrid search: BM25 + Vector + Graph → RRF rerank → SQLite hydration."""
        clean_query = re.sub(r"[^\w\s]", " ", query)

        # Auto-detect temporal range from query
        if not date_from and not date_to:
            date_from, date_to = self._extract_temporal_range(query)

        if entities:
            # Entity-focused mode
            import re as _re
            safe = [_re.sub(r'[&|*"^()]', ' ', e).strip() for e in entities]
            bm25_q = " ".join(e for e in safe if e)
            bm25_results = bm25_search(self.db.memory, bm25_q, limit=limit * 3) if bm25_q else []

            vec_all = vector_search(self.db.memory, self.embedder, query, limit=limit * 5)
            entity_lower = [e.lower() for e in entities]
            vec_results = [r for r in vec_all if any(ent in r.content.lower() for ent in entity_lower)]

            kg_results = graph_search(self.db.graph, query, limit=limit * 3, entities=entities, include_historical=include_historical)
        else:
            bm25_results = bm25_search(self.db.memory, clean_query, limit=limit * 3)
            vec_results = vector_search(self.db.memory, self.embedder, query, limit=limit * 3)
            kg_results = graph_search(self.db.graph, query, limit=limit * 3, include_historical=include_historical)

        results = rrf_rerank(bm25_results, vec_results, kg_results, limit=limit)

        # Date filter — prefer event_time (when it happened) over date (when saved)
        if date_from or date_to:
            def _match_date(r) -> bool:
                # Use event_time if available, fallback to processing date
                d = getattr(r, "event_time", None) or r.date
                if not d:
                    return True  # keep results with no date info
                if date_from and d < date_from:
                    return False
                if date_to and d > date_to:
                    return False
                return True
            results = [r for r in results if _match_date(r)]

        # Hydrate from SQLite via content_hash (Phase 7 pattern)
        hashes = list({r.content_hash for r in results if r.content_hash})
        hydrated = self.db.memory.get_by_hashes(hashes) if hashes else {}

        output = []
        for r in results:
            if r.content_hash and r.content_hash in hydrated:
                entry = hydrated[r.content_hash]
                output.append({
                    "content": entry["text"],
                    "date": entry.get("date", r.date),
                    "mood": entry.get("mood", r.mood),
                    "event_time": entry.get("event_time", "") or r.event_time,
                    "score": round(r.score, 4),
                    "source": r.source,
                    "content_hash": r.content_hash,
                })
            else:
                output.append({
                    "content": r.content,
                    "date": r.date,
                    "mood": r.mood,
                    "event_time": r.event_time,
                    "score": round(r.score, 4),
                    "source": r.source,
                    "content_hash": r.content_hash or "",
                })

        # Graph context for entity-focused search
        response: dict = {
            "query": query,
            "entities_used": entities or [],
            "count": len(output),
            "results": output,
        }
        if entities:
            response["graph_context"] = self._graph_context(entities, results)

        return response

    def _graph_context(self, entities: list[str], text_results) -> dict:
        """Fetch graph nodes + extra evidence edges for the given entities."""
        graph_nodes: dict[str, dict] = {}
        all_edges = []
        for ent in entities:
            traversal = self.db.graph.traverse(ent, max_hops=2, limit=20)
            for n in traversal.nodes:
                graph_nodes.setdefault(n.name, {"name": n.name, "type": n.type, "mention_count": n.mention_count})
            all_edges.extend(traversal.edges)

        text_hashes = {r.content_hash for r in text_results if r.content_hash}
        seen: set[str] = set()
        unique_edges = []
        for e in sorted(all_edges, key=lambda x: x.weight, reverse=True):
            if e.source_hash and e.source_hash not in text_hashes and e.source_hash not in seen:
                seen.add(e.source_hash)
                unique_edges.append(e)

        top_edges = unique_edges[:max(0, 20 - len(text_results))]
        edge_hashes = [e.source_hash for e in top_edges if e.source_hash]
        edge_hydrated = self.db.memory.get_by_hashes(edge_hashes) if edge_hashes else {}

        return {
            "nodes": list(graph_nodes.values()),
            "evidence": [
                {
                    "source": e.source, "target": e.target, "type": e.rel_type,
                    "weight": round(e.weight, 2),
                    "evidence": edge_hydrated.get(e.source_hash, {}).get("text", e.evidence or ""),
                }
                for e in top_edges
            ],
        }

    # ── recall / explain ───────────────────────────────────────────────────────

    def recall_entity(
        self, entity: str, max_hops: int = 2, limit: int = 10, include_historical: bool = False,
    ) -> dict:
        """Recall everything related to an entity via graph traversal + hydration."""
        result = self.db.graph.traverse(entity, max_hops=max_hops, limit=limit, include_historical=include_historical)
        hashes = list({e.source_hash for e in result.edges if e.source_hash})
        hydrated = self.db.memory.get_by_hashes(hashes) if hashes else {}

        return {
            "entity": entity,
            "connected_count": len(result.nodes),
            "nodes": [{"name": n.name, "type": n.type, "mention_count": n.mention_count} for n in result.nodes],
            "relationships": [{"source": e.source, "target": e.target, "type": e.rel_type, "weight": round(e.weight, 2)} for e in result.edges],
            "source_memories": [
                {"content": v["text"], "date": v.get("date", ""), "mood": v.get("mood", ""), "content_hash": k}
                for k, v in hydrated.items()
            ],
        }

    def explain_connection(
        self, entity_a: str, entity_b: str, include_historical: bool = False,
    ) -> dict:
        """Find and explain the path between two entities in the graph."""
        result = self.db.graph.find_path(entity_a, entity_b, include_historical=include_historical)
        hashes = list({e.source_hash for e in result.edges if e.source_hash})
        hydrated = self.db.memory.get_by_hashes(hashes) if hashes else {}

        return {
            "from": entity_a, "to": entity_b,
            "connected": len(result.paths) > 0,
            "paths": result.paths,
            "nodes": [{"name": n.name, "type": n.type} for n in result.nodes],
            "source_memories": [{"content": v["text"], "date": v.get("date", ""), "mood": v.get("mood", ""), "content_hash": k} for k, v in hydrated.items()],
        }

    # ── list / timeline ────────────────────────────────────────────────────────

    def list_entities(self, limit: int = 50) -> dict:
        entities = self.db.graph.get_canonical_entities(limit=limit)
        return {"count": len(entities), "entities": entities}

    def list_memory_dates(self) -> dict:
        dates = self.db.memory.get_dates()
        return {"count": len(dates), "dates": dates}

    def get_timeline(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        sort_by: str = "processing_time",
    ) -> dict:
        entries = self.db.memory.get_timeline(start_date, end_date, limit, sort_by=sort_by)
        return {"count": len(entries), "sort_by": sort_by, "timeline": entries}

    # ── temporal helper ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_temporal_range(query: str) -> tuple[str | None, str | None]:
        """Detect Vietnamese year/month patterns → (date_from, date_to)."""
        import calendar
        now = datetime.now(JST)
        q = query.lower()
        if re.search(r"năm\s+nay", q):
            y = now.year
            return f"{y}-01-01", f"{y}-12-31"
        if re.search(r"năm\s+(?:ngoái|trước)", q):
            y = now.year - 1
            return f"{y}-01-01", f"{y}-12-31"
        m = re.search(r"tháng\s+(\d{1,2})\s*(?:/|năm)\s*(\d{4})", q)
        if m:
            month, year = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12 and 1900 <= year <= 2100:
                last = calendar.monthrange(year, month)[1]
                return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"
        m = re.search(r"(?:năm|year)\s+(\d{4})", q) or re.search(r"\b(20\d{2}|19\d{2})\b", q)
        if m:
            y = int(m.group(1))
            if 1900 <= y <= 2100:
                return f"{y}-01-01", f"{y}-12-31"
        return None, None

    # ── consolidate ────────────────────────────────────────────────────────────

    def consolidate(
        self,
        half_life_days: int = 90,
        older_than_days: int = 30,
        auto_merge: bool = False,
    ) -> dict:
        """Run all consolidation steps. Returns report for agent to act on.

        Steps:
          1. Apply confidence decay to KG edges
          2. Scan for near-duplicate entities (reuse dedup engine)
          3. Surface stale memories older than `older_than_days`
        """
        from kioku_lite.pipeline.consolidation import (
            build_decay_report,
            build_stale_report,
            stale_cutoff,
        )

        # Step 1: confidence decay
        decayed = self.db.graph.apply_confidence_decay(half_life_days=half_life_days)

        # Step 2: dedup scan
        dedup_result = self.dedup_scan(auto_merge=auto_merge)

        # Step 3: stale memories
        cutoff = stale_cutoff(older_than_days)
        stale = self.db.memory.get_timeline(end_date=cutoff.isoformat(), limit=50)

        return {
            "decay": build_decay_report(decayed, half_life_days),
            "merge_suggestions": {
                "candidates": dedup_result.get("candidates", []),
                "auto_merged": dedup_result.get("auto_merged", []),
            },
            "stale_memories": build_stale_report(stale, cutoff),
        }

    # ── export ─────────────────────────────────────────────────────────────────

    def get_graph_data(self) -> dict:
        """Return all nodes and edges for graph export (D3 node-link format)."""
        nodes = self.db.graph.get_all_nodes()
        links = self.db.graph.get_all_edges()
        return {"nodes": nodes, "links": links}

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.db.close()

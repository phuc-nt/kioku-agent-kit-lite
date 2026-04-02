"""Knowledge graph search via BFS traversal on GraphStore.

Entity-focused queries (entities param provided) use Personalized PageRank (PPR).
Token-based queries (no entities) fall back to BFS for exact-match recall.
"""

from __future__ import annotations

import re

from kioku_lite.pipeline.graph_store import GraphStore
from kioku_lite.search.bm25 import SearchResult
from kioku_lite.search.pagerank import personalized_pagerank, ppr_to_results

_STOPWORDS = {
    "là", "và", "của", "có", "cho", "với", "được", "này", "đó", "các",
    "một", "những", "trong", "để", "từ", "theo", "về", "hay", "hoặc",
    "nhưng", "mà", "nếu", "khi", "thì", "đã", "sẽ", "đang", "rồi",
    "nào", "gì", "thế", "sao", "tại", "vì", "bị", "do", "qua", "lại",
    "như", "hơn", "nhất", "rất", "quá", "cũng", "vẫn", "còn", "chỉ",
    "tôi", "anh", "em", "bạn", "mình", "chúng", "họ", "ai",
    "the", "is", "are", "was", "were", "what", "who", "how", "why",
}


def graph_search(
    store: GraphStore,
    query: str,
    limit: int = 20,
    entities: list[str] | None = None,
    include_historical: bool = False,
) -> list[SearchResult]:
    """Search the knowledge graph by entity traversal.

    If `entities` provided (agent pre-extracted): use them as seeds.
    Otherwise: tokenize query and search per meaningful token.
    """
    seed_map: dict[str, object] = {}

    if entities:
        for entity_name in entities:
            pat = r"(?<!\w)" + re.escape(entity_name) + r"(?!\w)"
            for node in store.search_nodes(entity_name, limit=5):
                if re.search(pat, node.name, re.IGNORECASE):
                    seed_map.setdefault(node.name, node)
    else:
        tokens = re.findall(r"\w+", query.lower())
        meaningful = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
        if not meaningful:
            return []
        for token in meaningful:
            pat = r"(?<!\w)" + re.escape(token) + r"(?!\w)"
            for node in store.search_nodes(token, limit=5):
                if re.search(pat, node.name, re.IGNORECASE):
                    seed_map.setdefault(node.name, node)

    if not seed_map:
        return []

    ranked_seeds = sorted(
        seed_map.values(),
        key=lambda e: getattr(e, "mention_count", 0),
        reverse=True,
    )[:5]

    # Task 1A: exclude self-entity (hub node) from seeds — it connects to
    # everything and adds no signal. Only skip if other seeds are available.
    self_entity = store.get_top_entity()
    if self_entity:
        filtered = [e for e in ranked_seeds if e.name.lower() != self_entity.lower()]
        if filtered:  # fallback: keep all when self is the only seed
            ranked_seeds = filtered

    # --- PPR path (entity-focused search) ---
    # When entities param is provided, use Personalized PageRank: naturally
    # handles multi-seed and hub dilution without BFS intersection heuristics.
    if entities and ranked_seeds:
        seed_names = [e.name for e in ranked_seeds]
        ppr_scores = personalized_pagerank(
            store.conn, seed_names, include_historical=include_historical,
        )
        results = ppr_to_results(
            store.conn, ppr_scores, limit=limit,
            include_historical=include_historical,
        )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # --- BFS path (token-based fallback, no entities param) ---
    # Collect per-entity traversal results
    # results_by_hash: dedup by source_hash (memory-backed edges)
    # orphan_results:  edges without source_hash, always included via union
    results_by_hash: dict[str, SearchResult] = {}
    orphan_keys: set[str] = set()
    orphan_results: list[SearchResult] = []

    for entity in ranked_seeds:
        traversal = store.traverse(entity.name, max_hops=2, limit=limit, include_historical=include_historical)
        for edge in traversal.edges:
            r = SearchResult(
                content=edge.evidence or "",
                date="", mood="", timestamp="",
                score=edge.weight,
                source="graph",
                content_hash=edge.source_hash,
            )
            if edge.source_hash:
                results_by_hash.setdefault(edge.source_hash, r)
            else:
                key = edge.evidence
                if key and key not in orphan_keys:
                    orphan_keys.add(key)
                    orphan_results.append(r)

    results = list(results_by_hash.values()) + orphan_results
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]

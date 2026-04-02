"""Tests for Personalized PageRank: unit tests for algorithm functions
and integration tests verifying graph_search routes correctly to PPR vs BFS.
"""

from __future__ import annotations

import pytest

from kioku_lite.pipeline.graph_store import GraphStore
from kioku_lite.search.graph import graph_search
from kioku_lite.search.pagerank import personalized_pagerank, ppr_to_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_node(graph: GraphStore, name: str, n: int = 1) -> None:
    for _ in range(n):
        graph.upsert_node(name, "PERSON", "2026-01-01")


def _add_edge(
    graph: GraphStore,
    src: str,
    tgt: str,
    evidence: str = "",
    h: str = "",
    weight: float = 0.7,
    valid_until: str | None = None,
) -> None:
    graph.upsert_edge(src, tgt, "KNOWS", weight, evidence, h)
    if valid_until:
        graph.invalidate_edge(valid_until, source=src, target=tgt, rel_type="KNOWS")


# ── Unit Tests: personalized_pagerank() ───────────────────────────────────────

class TestPersonalizedPageRank:
    def test_empty_graph_returns_empty(self, graph: GraphStore) -> None:
        """No edges → empty scores."""
        result = personalized_pagerank(graph.conn, ["A"])
        assert result == {}

    def test_empty_seeds_returns_empty(self, graph: GraphStore) -> None:
        """Empty seed list → empty scores immediately."""
        _add_node(graph, "A")
        _add_edge(graph, "A", "B", h="h1")
        result = personalized_pagerank(graph.conn, [])
        assert result == {}

    def test_single_seed_linear_chain_proximity(self, graph: GraphStore) -> None:
        """A→B→C chain seeded at A: B must score higher than C (closer to seed)."""
        _add_node(graph, "A"); _add_node(graph, "B"); _add_node(graph, "C")
        _add_edge(graph, "A", "B", h="h_ab")
        _add_edge(graph, "B", "C", h="h_bc")
        scores = personalized_pagerank(graph.conn, ["A"])
        assert "b" in scores and "c" in scores
        assert scores["b"] > scores["c"], "Closer node B should score higher than distant C"

    def test_multi_seed_both_neighbors_score_high(self, graph: GraphStore) -> None:
        """Seeds A and B: neighbors of A and B both score highly."""
        _add_node(graph, "A"); _add_node(graph, "B")
        _add_node(graph, "NA"); _add_node(graph, "NB")
        _add_edge(graph, "A", "NA", h="h1")
        _add_edge(graph, "B", "NB", h="h2")
        scores = personalized_pagerank(graph.conn, ["A", "B"])
        # Both neighborhoods should receive score
        assert scores.get("na", 0) > 0
        assert scores.get("nb", 0) > 0

    def test_disconnected_seed_returns_teleport_score(self, graph: GraphStore) -> None:
        """Seed not in graph edge set → still returns its teleport score."""
        # Graph has A-B only; seed is "Ghost" not connected to anything
        _add_node(graph, "A"); _add_node(graph, "B")
        _add_edge(graph, "A", "B", h="h1")
        scores = personalized_pagerank(graph.conn, ["Ghost"])
        # Ghost not in adjacency → returns teleport score dict
        assert "ghost" in scores
        assert scores["ghost"] == pytest.approx(1.0, abs=0.01)

    def test_case_insensitive_seeds(self, graph: GraphStore) -> None:
        """Seeds matched case-insensitively against graph nodes."""
        _add_node(graph, "Alice"); _add_node(graph, "Bob")
        _add_edge(graph, "Alice", "Bob", h="h1")
        scores_lower = personalized_pagerank(graph.conn, ["alice"])
        scores_upper = personalized_pagerank(graph.conn, ["Alice"])
        assert scores_lower == scores_upper

    def test_respects_edge_weights(self, graph: GraphStore) -> None:
        """Higher-weight edges propagate more score to their target."""
        _add_node(graph, "Root")
        _add_node(graph, "Heavy"); _add_node(graph, "Light")
        # Create via direct SQL to bypass upsert averaging
        graph.conn.execute(
            "INSERT INTO kg_edges (source, target, rel_type, weight, evidence, source_hash) "
            "VALUES ('Root', 'Heavy', 'W', 0.9, 'heavy edge', 'hh')"
        )
        graph.conn.execute(
            "INSERT INTO kg_edges (source, target, rel_type, weight, evidence, source_hash) "
            "VALUES ('Root', 'Light', 'W', 0.1, 'light edge', 'hl')"
        )
        graph.conn.commit()
        scores = personalized_pagerank(graph.conn, ["Root"])
        assert scores.get("heavy", 0) > scores.get("light", 0), \
            "Heavy edge target must score higher than light edge target"

    def test_historical_flag_includes_invalidated_edges(self, graph: GraphStore) -> None:
        """include_historical=True includes edges with valid_until set."""
        _add_node(graph, "A"); _add_node(graph, "B")
        _add_edge(graph, "A", "B", h="h1", valid_until="2025-01-01")
        # Without historical: no valid edges → empty
        scores_current = personalized_pagerank(graph.conn, ["A"], include_historical=False)
        assert scores_current == {}
        # With historical: edge included
        scores_hist = personalized_pagerank(graph.conn, ["A"], include_historical=True)
        assert "b" in scores_hist

    def test_hub_dilution(self, graph: GraphStore) -> None:
        """Hub connected to many nodes gets diluted PPR score per neighbor."""
        _add_node(graph, "Hub")
        _add_node(graph, "Specific"); _add_node(graph, "SpecificNeighbor")
        # Hub → 10 nodes (spreads score thin)
        for i in range(10):
            _add_node(graph, f"HubN{i}")
            _add_edge(graph, "Hub", f"HubN{i}", h=f"hh{i}")
        # Specific → 1 node (concentrates score)
        _add_edge(graph, "Specific", "SpecificNeighbor", h="hs")

        scores_hub = personalized_pagerank(graph.conn, ["Hub"])
        scores_spec = personalized_pagerank(graph.conn, ["Specific"])

        # SpecificNeighbor should get higher share of score from Specific than
        # any single HubN gets from Hub
        specific_n_score = scores_spec.get("specificneighbor", 0)
        best_hub_n = max(scores_hub.get(f"hubn{i}", 0) for i in range(10))
        assert specific_n_score > best_hub_n, \
            "Focused seed gives higher per-neighbor score than hub"


# ── Unit Tests: ppr_to_results() ──────────────────────────────────────────────

class TestPprToResults:
    def test_basic_mapping_to_search_result(self, graph: GraphStore) -> None:
        """Results are SearchResult with source='graph' and correct content_hash."""
        _add_node(graph, "A"); _add_node(graph, "B")
        _add_edge(graph, "A", "B", evidence="test evidence", h="h1")
        scores = personalized_pagerank(graph.conn, ["A"])
        results = ppr_to_results(graph.conn, scores)
        assert len(results) >= 1
        r = results[0]
        assert r.source == "graph"
        assert r.content_hash == "h1"
        assert r.content == "test evidence"

    def test_empty_scores_returns_empty(self, graph: GraphStore) -> None:
        """Empty PPR scores → empty results list."""
        results = ppr_to_results(graph.conn, {})
        assert results == []

    def test_limit_enforcement(self, graph: GraphStore) -> None:
        """Returns at most `limit` results."""
        _add_node(graph, "Hub")
        for i in range(10):
            _add_node(graph, f"N{i}")
            _add_edge(graph, "Hub", f"N{i}", h=f"h{i}")
        scores = personalized_pagerank(graph.conn, ["Hub"])
        results = ppr_to_results(graph.conn, scores, limit=3)
        assert len(results) <= 3

    def test_dedup_by_source_hash(self, graph: GraphStore) -> None:
        """Same source_hash from multiple entities appears only once (additive score)."""
        _add_node(graph, "A"); _add_node(graph, "B"); _add_node(graph, "Shared")
        _add_edge(graph, "A", "Shared", evidence="shared memory", h="h_shared")
        _add_edge(graph, "B", "Shared", evidence="shared memory", h="h_shared")
        scores = personalized_pagerank(graph.conn, ["A", "B"])
        results = ppr_to_results(graph.conn, scores)
        hashes = [r.content_hash for r in results]
        assert hashes.count("h_shared") == 1, "source_hash must appear exactly once"

    def test_shared_memory_scores_higher_than_single(self, graph: GraphStore) -> None:
        """Memory connected to both seeds scores higher than single-seed memory."""
        _add_node(graph, "A"); _add_node(graph, "B"); _add_node(graph, "Shared")
        _add_edge(graph, "A", "Shared", h="h_shared")
        _add_edge(graph, "B", "Shared", h="h_shared")
        _add_node(graph, "OnlyA")
        _add_edge(graph, "A", "OnlyA", h="h_only_a")
        scores = personalized_pagerank(graph.conn, ["A", "B"])
        results = ppr_to_results(graph.conn, scores)
        score_map = {r.content_hash: r.score for r in results}
        assert score_map["h_shared"] > score_map.get("h_only_a", 0), \
            "Shared memory (additive) must score higher than single-seed memory"


# ── Integration Tests: graph_search() PPR vs BFS routing ──────────────────────

class TestPprIntegration:
    def test_graph_search_with_entities_returns_results(self, graph: GraphStore) -> None:
        """graph_search with entities param uses PPR and returns SearchResults."""
        _add_node(graph, "Alice", 3); _add_node(graph, "Bob", 1)
        _add_edge(graph, "Alice", "Bob", evidence="Alice knows Bob", h="h1")
        results = graph_search(graph, "alice", entities=["Alice"])
        assert len(results) >= 1
        assert all(r.source == "graph" for r in results)

    def test_graph_search_without_entities_uses_bfs(self, graph: GraphStore) -> None:
        """graph_search without entities param uses token-based BFS fallback."""
        _add_node(graph, "Techbase", 2); _add_node(graph, "Brain", 1)
        _add_edge(graph, "Techbase", "Brain", evidence="tech brain", h="h_tb")
        results = graph_search(graph, "Techbase Brain")
        assert len(results) >= 1
        assert results[0].source == "graph"

    def test_graph_search_ppr_empty_seeds(self, graph: GraphStore) -> None:
        """Entity provided but not in graph → empty results."""
        results = graph_search(graph, "query", entities=["NoSuchEntity"])
        assert results == []

    def test_graph_search_ppr_hub_handling(self, graph: GraphStore) -> None:
        """With PPR, focused entity's direct neighbor scores higher than hub's node."""
        _add_node(graph, "Hub", 20)
        # Hub connected to 15 nodes (high degree)
        for i in range(15):
            _add_node(graph, f"HubN{i}", 1)
            _add_edge(graph, "Hub", f"HubN{i}", h=f"hh{i}")
        # Focused entity connected to just 1 node
        _add_node(graph, "Focused", 2); _add_node(graph, "Target", 1)
        _add_edge(graph, "Focused", "Target", evidence="direct connection", h="h_target")
        # Search with Focused only (Hub auto-excluded as top entity)
        results = graph_search(graph, "query", entities=["Focused", "Hub"])
        assert len(results) >= 1
        # h_target must be in results
        hashes = {r.content_hash for r in results}
        assert "h_target" in hashes

    def test_graph_search_ppr_limit_respected(self, graph: GraphStore) -> None:
        """graph_search respects limit param when using PPR path."""
        _add_node(graph, "Center", 1)
        for i in range(10):
            _add_node(graph, f"N{i}", 1)
            _add_edge(graph, "Center", f"N{i}", h=f"h{i}")
        results = graph_search(graph, "query", entities=["Center"], limit=4)
        assert len(results) <= 4

    def test_graph_search_ppr_include_historical(self, graph: GraphStore) -> None:
        """include_historical=True expands PPR to invalidated edges."""
        _add_node(graph, "A"); _add_node(graph, "B")
        _add_edge(graph, "A", "B", evidence="old memory", h="h_old", valid_until="2025-01-01")
        # Without historical: no results (edge invalidated)
        results_current = graph_search(graph, "query", entities=["A"], include_historical=False)
        assert len(results_current) == 0
        # With historical: includes the invalidated edge
        results_hist = graph_search(graph, "query", entities=["A"], include_historical=True)
        assert len(results_hist) >= 1
        assert results_hist[0].content_hash == "h_old"

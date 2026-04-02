"""Tests for graph_search(): entity seeding, self-entity exclusion (Task 1A)."""

from __future__ import annotations

import pytest

from kioku_lite.pipeline.graph_store import GraphStore
from kioku_lite.search.graph import graph_search


def _add_node(graph: GraphStore, name: str, n: int = 1) -> None:
    for _ in range(n):
        graph.upsert_node(name, "PERSON", "2026-01-01")


def _add_edge(graph: GraphStore, src: str, tgt: str, evidence: str = "", h: str = "") -> None:
    graph.upsert_edge(src, tgt, "KNOWS", 0.7, evidence, h)


# ── basic graph_search ─────────────────────────────────────────────────────────

class TestGraphSearchBasic:
    def test_returns_results_for_known_entity(self, graph):
        _add_node(graph, "Mẹ", 3)
        _add_node(graph, "Phúc", 1)
        _add_edge(graph, "Mẹ", "Phúc", "memory about Mẹ", "h1")
        results = graph_search(graph, "mẹ", entities=["Mẹ"])
        assert len(results) >= 1
        assert all(r.source == "graph" for r in results)

    def test_empty_graph_returns_empty(self, graph):
        results = graph_search(graph, "anything", entities=["Unknown"])
        assert results == []

    def test_no_entities_falls_back_to_token_search(self, graph):
        _add_node(graph, "Phong", 2)
        _add_node(graph, "Phúc", 1)
        _add_edge(graph, "Phong", "Phúc", "played together", "h1")
        results = graph_search(graph, "Phong")
        assert len(results) >= 1

    def test_results_respect_limit(self, graph):
        _add_node(graph, "Hub", 5)
        for i in range(10):
            _add_node(graph, f"Node{i}", 1)
            _add_edge(graph, "Hub", f"Node{i}", f"ev{i}", f"h{i}")
        results = graph_search(graph, "hub", entities=["Hub"], limit=3)
        assert len(results) <= 3


# ── Task 1A: self-entity exclusion ────────────────────────────────────────────

class TestSelfEntityExclusion:
    def test_hub_excluded_when_other_seeds_present(self, graph):
        """When hub (highest mention_count) is passed alongside other entities,
        hub is dropped from seeds so Mẹ's direct neighbor scores highest.

        With PPR (replaces BFS): Phúc is excluded from the teleport set so its
        edges score lower than Mẹ's direct neighbors. h_me_sato must appear;
        h_irr may appear but must score lower than h_me_sato (hub diluted).
        """
        _add_node(graph, "Phúc", 10)
        _add_node(graph, "Mẹ", 3)
        _add_node(graph, "Sato", 1)

        _add_edge(graph, "Mẹ", "Sato", "Mẹ gặp Sato", "h_me_sato")
        _add_node(graph, "IrrelevantNode", 1)
        _add_edge(graph, "Phúc", "IrrelevantNode", "unrelated memory", "h_irr")

        results = graph_search(graph, "mẹ tình cảm", entities=["Mẹ", "Phúc"])
        hashes = {r.content_hash for r in results}

        assert "h_me_sato" in hashes, "Mẹ→Sato edge should be in results"
        # PPR: h_me_sato must score higher than h_irr (seed=Mẹ, Phúc is excluded from teleport)
        scores = {r.content_hash: r.score for r in results}
        if "h_irr" in scores:
            assert scores["h_me_sato"] >= scores["h_irr"], "Mẹ's direct neighbor must score >= hub's edge"

    def test_hub_kept_when_it_is_the_only_seed(self, graph):
        """Fallback: if hub is the only entity passed, keep it (don't return empty)."""
        _add_node(graph, "Phúc", 10)
        _add_node(graph, "TBV", 1)
        _add_edge(graph, "Phúc", "TBV", "works at TBV", "h_tbv")

        results = graph_search(graph, "công việc", entities=["Phúc"])
        assert len(results) >= 1, "Should still return results when hub is the only seed"

    def test_correct_entity_selected_as_top(self, graph):
        """get_top_entity() picks entity with most mentions — ensures correct hub detection."""
        _add_node(graph, "LowMention", 1)
        _add_node(graph, "MidMention", 5)
        _add_node(graph, "HubPerson", 20)

        assert graph.get_top_entity() == "HubPerson"

    def test_exclusion_uses_case_insensitive_match(self, graph):
        """Hub exclusion should be case-insensitive.

        With PPR: 'phúc' (lowercase) should still be recognized as the hub and
        excluded from the teleport set. Mẹ's direct edge h_mb must appear and
        score higher than the hub's edge h_noise.
        """
        _add_node(graph, "Phúc", 10)  # stored as "Phúc"
        _add_node(graph, "Mẹ", 3)
        _add_node(graph, "Bạn bè", 1)
        _add_edge(graph, "Mẹ", "Bạn bè", "evidence", "h_mb")
        _add_node(graph, "Noise", 1)
        _add_edge(graph, "Phúc", "Noise", "noise", "h_noise")

        # Pass "phúc" (lowercase) — should still be recognized as hub
        results = graph_search(graph, "mẹ", entities=["Mẹ", "phúc"])
        hashes = {r.content_hash for r in results}
        assert "h_mb" in hashes, "Mẹ's direct neighbor should be in results"
        # h_mb must score at least as high as h_noise (Phúc excluded from seeds)
        scores = {r.content_hash: r.score for r in results}
        if "h_noise" in scores:
            assert scores["h_mb"] >= scores["h_noise"], "Mẹ edge must score >= hub's edge"


# ── Task 2E: multi-entity intersection ────────────────────────────────────────

class TestMultiEntityIntersection:
    def test_intersection_keeps_only_common_memories(self, graph):
        """PPR replaces strict intersection: shared memory scores highest.

        With PPR + multi-seed teleport: h_common receives score from BOTH A and B
        seeds, so it must rank first. Single-entity memories may also appear but
        must score lower than the shared memory.
        """
        # Hub entity — will be auto-excluded by Task 1A
        _add_node(graph, "User", 10)

        # Entity A → memories h1, h2, h_common
        _add_node(graph, "A", 2)
        _add_node(graph, "NodeA1", 1)
        _add_node(graph, "NodeA2", 1)
        _add_node(graph, "Shared", 1)
        _add_edge(graph, "A", "NodeA1", "only A sees this", "h1")
        _add_edge(graph, "A", "NodeA2", "only A sees this too", "h2")
        _add_edge(graph, "A", "Shared", "A and B both see this", "h_common")

        # Entity B → memories h4, h_common
        _add_node(graph, "B", 2)
        _add_node(graph, "NodeB1", 1)
        _add_edge(graph, "B", "NodeB1", "only B sees this", "h4")
        _add_edge(graph, "B", "Shared", "A and B both see this", "h_common")

        results = graph_search(graph, "test", entities=["A", "B"])
        hashes = {r.content_hash for r in results}

        assert "h_common" in hashes, "Shared memory must be in PPR results"
        # h_common should rank at or near the top (highest score)
        top_result = results[0]
        assert top_result.content_hash == "h_common", "Shared memory should be ranked first by PPR"

    def test_intersection_fallback_to_union_when_no_common(self, graph):
        """When no memory is reachable from ALL seeds, fall back to union.
        A clear hub ('User') is added so neither X nor Y is auto-excluded by 1A.
        """
        _add_node(graph, "User", 10)  # hub — auto-excluded by 1A

        _add_node(graph, "X", 2)
        _add_node(graph, "Y", 2)
        _add_node(graph, "NX", 1)
        _add_node(graph, "NY", 1)
        _add_edge(graph, "X", "NX", "X memory", "hx")
        _add_edge(graph, "Y", "NY", "Y memory", "hy")

        # X and Y share no common memories → fallback to union
        results = graph_search(graph, "test", entities=["X", "Y"])
        hashes = {r.content_hash for r in results}

        assert "hx" in hashes, "Union fallback should include X memories"
        assert "hy" in hashes, "Union fallback should include Y memories"

    def test_single_seed_uses_union(self, graph):
        """Single entity → no intersection applied, return all reachable memories."""
        _add_node(graph, "Solo", 2)
        for i in range(3):
            _add_node(graph, f"N{i}", 1)
            _add_edge(graph, "Solo", f"N{i}", f"memory {i}", f"h{i}")

        results = graph_search(graph, "test", entities=["Solo"])
        hashes = {r.content_hash for r in results}

        assert len(hashes) == 3, "All 3 memories should be returned for single entity"

    def test_three_entity_intersection(self, graph):
        """PPR with 3 seeds: the shared memory scores highest.

        h_all receives teleport pressure from all 3 seeds → ranked first.
        h_12 connected to only E1+E2 may appear but must score lower than h_all.
        """
        for name in ["E1", "E2", "E3"]:
            _add_node(graph, name, 2)

        # Shared node connected to all 3
        _add_node(graph, "Common", 1)
        _add_edge(graph, "E1", "Common", "all three meet here", "h_all")
        _add_edge(graph, "E2", "Common", "all three meet here", "h_all")
        _add_edge(graph, "E3", "Common", "all three meet here", "h_all")

        # Exclusive memories — connected to only E1 and E2
        _add_node(graph, "Only12", 1)
        _add_edge(graph, "E1", "Only12", "only E1 and E2", "h_12")
        _add_edge(graph, "E2", "Only12", "only E1 and E2", "h_12")

        results = graph_search(graph, "test", entities=["E1", "E2", "E3"])
        hashes = {r.content_hash for r in results}

        assert "h_all" in hashes, "Memory reachable from all 3 should be in PPR results"
        # h_all should rank above h_12 since it receives score from all 3 seeds
        scores = {r.content_hash: r.score for r in results}
        if "h_12" in scores:
            assert scores["h_all"] >= scores["h_12"], "Triple-seed memory must score >= dual-seed memory"

    def test_intersection_not_applied_without_explicit_entities(self, graph):
        """Token-based fallback (no entities param) should still return results."""
        _add_node(graph, "Techbase", 2)
        _add_node(graph, "Brain", 2)
        _add_edge(graph, "Techbase", "Brain", "career memory", "h_career")

        results = graph_search(graph, "Techbase Brain career")
        assert len(results) >= 1

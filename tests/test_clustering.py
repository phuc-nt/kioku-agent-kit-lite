"""Tests for Community/Cluster Detection (Phase 3).

Covers:
  - Connected component algorithm (single, multiple, isolated, empty graph)
  - Label suggestion logic
  - GraphStore cluster persistence (save/get/get_entities)
  - Service: detect_clusters, get_cluster
  - CLI: clusters command, cluster command
  - Consolidate includes cluster info
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kioku_lite.cli import app
from kioku_lite.pipeline.clustering import Community, find_communities
from kioku_lite.pipeline.graph_store import GraphStore

runner = CliRunner()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_cli_singleton():
    """Reset global CLI service singleton between tests."""
    import kioku_lite.cli as _cli_module
    import kioku_lite.config as _cfg_module
    _cli_module._svc = None
    _orig = _cfg_module.settings
    yield
    if _cli_module._svc is not None:
        try:
            _cli_module._svc.close()
        except Exception:
            pass
    _cli_module._svc = None
    _cfg_module.settings = _orig


def make_env(tmp_path: Path) -> dict:
    return {
        "KIOKU_LITE_USER_ID": "cluster_test",
        "KIOKU_LITE_EMBED_PROVIDER": "fake",
        "KIOKU_LITE_EMBED_DIM": "128",
        "KIOKU_LITE_MEMORY_DIR": str(tmp_path / "memory"),
        "KIOKU_LITE_DATA_DIR": str(tmp_path / "data"),
    }


def _add_node(graph: GraphStore, name: str, entity_type: str = "PERSON") -> None:
    graph.upsert_node(name, entity_type, "2026-04-01")


def _add_edge(graph: GraphStore, src: str, tgt: str) -> None:
    graph.upsert_edge(src, tgt, "KNOWS", 0.7, "", "hash_test")


# ── Algorithm: connected components ───────────────────────────────────────────

class TestConnectedComponents:

    def test_empty_graph_returns_empty(self, graph: GraphStore):
        """No nodes → no communities."""
        communities = find_communities(graph.conn)
        assert communities == []

    def test_single_isolated_node_is_one_community(self, graph: GraphStore):
        """A single node with no edges forms its own cluster."""
        _add_node(graph, "Alice")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert "Alice" in communities[0].entities

    def test_two_connected_nodes_form_one_community(self, graph: GraphStore):
        """Two nodes joined by an edge → one community with both."""
        _add_node(graph, "Alice")
        _add_node(graph, "Bob")
        _add_edge(graph, "Alice", "Bob")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert set(communities[0].entities) == {"Alice", "Bob"}

    def test_two_disjoint_pairs_form_two_communities(self, graph: GraphStore):
        """Two disconnected pairs → two communities."""
        _add_node(graph, "Alice")
        _add_node(graph, "Bob")
        _add_node(graph, "Carol")
        _add_node(graph, "Dave")
        _add_edge(graph, "Alice", "Bob")
        _add_edge(graph, "Carol", "Dave")
        communities = find_communities(graph.conn)
        assert len(communities) == 2
        # Each community has exactly 2 entities
        sizes = sorted(len(c.entities) for c in communities)
        assert sizes == [2, 2]

    def test_isolated_node_gets_own_community(self, graph: GraphStore):
        """Node with no edges gets its own community (not merged with others)."""
        _add_node(graph, "Alice")
        _add_node(graph, "Bob")
        _add_node(graph, "Isolated")
        _add_edge(graph, "Alice", "Bob")
        communities = find_communities(graph.conn)
        assert len(communities) == 2
        entity_sets = [set(c.entities) for c in communities]
        assert {"Alice", "Bob"} in entity_sets
        assert {"Isolated"} in entity_sets

    def test_chain_of_nodes_is_one_community(self, graph: GraphStore):
        """Alice→Bob→Carol→Dave (chain) → one community with all four."""
        for name in ("Alice", "Bob", "Carol", "Dave"):
            _add_node(graph, name)
        _add_edge(graph, "Alice", "Bob")
        _add_edge(graph, "Bob", "Carol")
        _add_edge(graph, "Carol", "Dave")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert set(communities[0].entities) == {"Alice", "Bob", "Carol", "Dave"}

    def test_include_historical_includes_superseded_edges(self, graph: GraphStore):
        """With include_historical=True, invalid edges are still used for clustering."""
        _add_node(graph, "Alice")
        _add_node(graph, "Bob")
        _add_edge(graph, "Alice", "Bob")
        # Invalidate the edge
        graph.invalidate_edge("2025-01-01", source="Alice", target="Bob", rel_type="KNOWS")

        # Without historical: two separate communities
        without = find_communities(graph.conn, include_historical=False)
        assert len(without) == 2

        # With historical: one community
        with_hist = find_communities(graph.conn, include_historical=True)
        assert len(with_hist) == 1

    def test_communities_sorted_by_size_descending(self, graph: GraphStore):
        """Largest cluster is listed first."""
        for name in ("A", "B", "C"):
            _add_node(graph, name)
        _add_node(graph, "Lone")
        _add_edge(graph, "A", "B")
        _add_edge(graph, "B", "C")
        communities = find_communities(graph.conn)
        assert len(communities) == 2
        assert len(communities[0].entities) >= len(communities[1].entities)

    def test_edge_count_is_correct(self, graph: GraphStore):
        """edge_count reflects the number of edges within a cluster."""
        for name in ("Alice", "Bob", "Carol"):
            _add_node(graph, name)
        _add_edge(graph, "Alice", "Bob")
        _add_edge(graph, "Bob", "Carol")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert communities[0].edge_count == 2


# ── Label suggestion ──────────────────────────────────────────────────────────

class TestLabelSuggestion:

    def test_majority_type_wins(self, graph: GraphStore):
        """Most frequent entity type in cluster becomes the label."""
        graph.upsert_node("Alice", "PERSON", "2026-04-01")
        graph.upsert_node("Bob", "PERSON", "2026-04-01")
        graph.upsert_node("Acme", "ORG", "2026-04-01")
        _add_edge(graph, "Alice", "Bob")
        _add_edge(graph, "Bob", "Acme")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert communities[0].label == "PERSON"

    def test_single_node_uses_its_own_type(self, graph: GraphStore):
        """A single node in a cluster uses its entity type as label."""
        graph.upsert_node("ProjectX", "PROJECT", "2026-04-01")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert communities[0].label == "PROJECT"

    def test_unknown_label_for_no_type(self, graph: GraphStore):
        """Node with empty type → label is UNKNOWN."""
        graph.upsert_node("Mystery", "", "2026-04-01")
        communities = find_communities(graph.conn)
        assert len(communities) == 1
        assert communities[0].label == "UNKNOWN"


# ── GraphStore: cluster persistence ──────────────────────────────────────────

class TestGraphStoreClusters:

    def test_save_and_get_clusters(self, graph: GraphStore):
        """save_clusters persists and get_clusters retrieves them."""
        clusters = [
            Community(id=0, label="PERSON", entities=["Alice", "Bob"], edge_count=1),
            Community(id=1, label="ORG", entities=["Acme"], edge_count=0),
        ]
        graph.save_clusters(clusters)
        result = graph.get_clusters()
        assert len(result) == 2
        labels = {r["label"] for r in result}
        assert labels == {"PERSON", "ORG"}

    def test_get_cluster_entities_by_label(self, graph: GraphStore):
        """get_cluster_entities returns entities for the named label."""
        clusters = [
            Community(id=0, label="PERSON", entities=["Alice", "Bob"], edge_count=1),
        ]
        graph.save_clusters(clusters)
        entities = graph.get_cluster_entities("PERSON")
        assert set(entities) == {"Alice", "Bob"}

    def test_get_cluster_entities_not_found(self, graph: GraphStore):
        """get_cluster_entities returns empty list for unknown label."""
        entities = graph.get_cluster_entities("NONEXISTENT")
        assert entities == []

    def test_save_clusters_replaces_previous(self, graph: GraphStore):
        """Second save_clusters call replaces all previous entries."""
        graph.save_clusters([Community(id=0, label="OLD", entities=["X"], edge_count=0)])
        graph.save_clusters([Community(id=0, label="NEW", entities=["Y", "Z"], edge_count=1)])
        result = graph.get_clusters()
        assert len(result) == 1
        assert result[0]["label"] == "NEW"
        assert result[0]["entity_count"] == 2


# ── Service: detect_clusters / get_cluster ────────────────────────────────────

class TestClusterService:

    def test_detect_clusters_empty_db(self, service):
        """detect_clusters on empty DB returns zero clusters."""
        result = service.detect_clusters()
        assert result["status"] == "clustered"
        assert result["cluster_count"] == 0
        assert result["total_entities"] == 0

    def test_detect_clusters_returns_summary(self, service):
        """detect_clusters runs correctly and returns expected keys."""
        from kioku_lite.service import EntityInput, RelationshipInput

        mem = service.save_memory("Alice works at Acme")
        service.kg_index(
            mem["content_hash"],
            entities=[EntityInput("Alice", "PERSON"), EntityInput("Acme", "ORG")],
            relationships=[RelationshipInput("Alice", "Acme", "WORKS_AT", 0.8, "Alice at Acme")],
        )
        result = service.detect_clusters()
        assert result["status"] == "clustered"
        assert result["cluster_count"] >= 1
        assert result["total_entities"] >= 2
        assert "clusters" in result

    def test_get_cluster_not_found(self, service):
        """get_cluster on missing label returns not_found status."""
        result = service.get_cluster("NONEXISTENT")
        assert result["status"] == "not_found"
        assert result["entities"] == []

    def test_get_cluster_returns_entities(self, service):
        """get_cluster returns entities after detect_clusters is called."""
        from kioku_lite.service import EntityInput, RelationshipInput

        mem = service.save_memory("Alice and Bob work together")
        service.kg_index(
            mem["content_hash"],
            entities=[EntityInput("Alice", "PERSON"), EntityInput("Bob", "PERSON")],
            relationships=[RelationshipInput("Alice", "Bob", "WORKS_WITH", 0.8, "work together")],
        )
        service.detect_clusters()
        result = service.get_cluster("PERSON")
        assert result["status"] == "ok"
        assert "Alice" in result["entities"] or "Bob" in result["entities"]


# ── CLI: clusters / cluster commands ─────────────────────────────────────────

class TestClusterCLI:

    def test_clusters_empty_db(self, tmp_path: Path):
        """kioku-lite clusters on empty DB returns valid JSON with zero clusters."""
        env = make_env(tmp_path)
        result = runner.invoke(app, ["clusters"], env=env)
        assert result.exit_code == 0, f"output: {result.output}"
        data = json.loads(result.output)
        assert data["status"] == "clustered"
        assert data["cluster_count"] == 0

    def test_clusters_with_data(self, tmp_path: Path):
        """kioku-lite clusters returns clusters after data is indexed."""
        env = make_env(tmp_path)
        # Save and index data first
        save_result = runner.invoke(app, ["save", "Alice met Bob today"], env=env)
        assert save_result.exit_code == 0
        content_hash = json.loads(save_result.output)["content_hash"]

        entities_json = json.dumps([
            {"name": "Alice", "type": "PERSON"},
            {"name": "Bob", "type": "PERSON"},
        ])
        rels_json = json.dumps([
            {"source": "Alice", "target": "Bob", "rel_type": "MET", "weight": 0.8, "evidence": "met today"},
        ])
        runner.invoke(app, ["kg-index", content_hash,
                            "--entities", entities_json,
                            "--relationships", rels_json], env=env)

        result = runner.invoke(app, ["clusters"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["cluster_count"] >= 1

    def test_cluster_not_found(self, tmp_path: Path):
        """kioku-lite cluster <label> on unknown label returns not_found."""
        env = make_env(tmp_path)
        result = runner.invoke(app, ["cluster", "NONEXISTENT"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "not_found"

    def test_cluster_found_after_detect(self, tmp_path: Path):
        """kioku-lite cluster <label> returns entities after clusters are detected."""
        env = make_env(tmp_path)
        save_result = runner.invoke(app, ["save", "Carol works at XYZ"], env=env)
        content_hash = json.loads(save_result.output)["content_hash"]

        entities_json = json.dumps([
            {"name": "Carol", "type": "PERSON"},
            {"name": "XYZ", "type": "ORG"},
        ])
        rels_json = json.dumps([
            {"source": "Carol", "target": "XYZ", "rel_type": "WORKS_AT", "weight": 0.9, "evidence": "works at"},
        ])
        runner.invoke(app, ["kg-index", content_hash,
                            "--entities", entities_json,
                            "--relationships", rels_json], env=env)
        # Detect clusters first
        runner.invoke(app, ["clusters"], env=env)

        result = runner.invoke(app, ["cluster", "PERSON"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert "Carol" in data["entities"]


# ── Consolidate includes cluster info ─────────────────────────────────────────

class TestConsolidateIncludesClusters:

    def test_consolidate_includes_clusters_section(self, service):
        """consolidate() result includes a 'clusters' key with summary info."""
        result = service.consolidate()
        assert "clusters" in result
        assert "cluster_count" in result["clusters"]
        assert "total_entities" in result["clusters"]

    def test_consolidate_cli_includes_clusters(self, tmp_path: Path):
        """kioku-lite consolidate JSON output has clusters key."""
        env = make_env(tmp_path)
        result = runner.invoke(app, ["consolidate"], env=env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "clusters" in data
        assert "cluster_count" in data["clusters"]

"""Tests for memory consolidation: confidence decay, consolidate() service, CLI command."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kioku_lite.cli import app
from kioku_lite.pipeline.graph_store import GraphStore

runner = CliRunner()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_cli_singleton():
    """Reset global CLI service singleton between tests."""
    import kioku_lite.cli as _cli_module
    import kioku_lite.config as _cfg_module
    _cli_module._svc = None
    _orig_settings = _cfg_module.settings
    yield
    if _cli_module._svc is not None:
        try:
            _cli_module._svc.close()
        except Exception:
            pass
    _cli_module._svc = None
    _cfg_module.settings = _orig_settings


def make_env(tmp_path: Path) -> dict:
    return {
        "KIOKU_LITE_USER_ID": "consol_test",
        "KIOKU_LITE_EMBED_PROVIDER": "fake",
        "KIOKU_LITE_EMBED_DIM": "128",
        "KIOKU_LITE_MEMORY_DIR": str(tmp_path / "memory"),
        "KIOKU_LITE_DATA_DIR": str(tmp_path / "data"),
    }


def _add_edge(
    graph: GraphStore,
    src: str = "Alice",
    tgt: str = "Acme",
    rel: str = "WORKS_AT",
    weight: float = 0.8,
) -> None:
    graph.upsert_edge(src, tgt, rel, weight, "evidence", "hash001")


def _backdate_edge(graph: GraphStore, src: str, tgt: str, rel: str, lr_date: str) -> None:
    """Force last_reinforced to a past date for decay testing."""
    cur = graph.conn.cursor()
    cur.execute(
        "UPDATE kg_edges SET last_reinforced = ? WHERE source = ? AND target = ? AND rel_type = ?",
        (lr_date, src, tgt, rel),
    )
    graph.conn.commit()


# ── Phase 1: Confidence Decay ─────────────────────────────────────────────────

class TestConfidenceDecay:

    def test_basic_decay_halves_weight_at_half_life(self, graph: GraphStore):
        """Edge reinforced exactly half_life_days ago → weight should be ~halved."""
        _add_edge(graph, weight=0.8)
        past = (date.today() - timedelta(days=30)).isoformat()
        _backdate_edge(graph, "Alice", "Acme", "WORKS_AT", past)

        ref = date.today().isoformat()
        decayed = graph.apply_confidence_decay(half_life_days=30, min_weight=0.1, reference_date=ref)

        assert len(decayed) == 1
        record = decayed[0]
        assert record["source"] == "Alice"
        assert record["target"] == "Acme"
        assert record["old_weight"] == pytest.approx(0.8, rel=1e-4)
        assert record["new_weight"] == pytest.approx(0.4, rel=1e-3)
        assert record["days_since_reinforced"] == 30

    def test_no_decay_for_fresh_edges(self, graph: GraphStore):
        """Edge reinforced today (last_reinforced == today) → no decay."""
        _add_edge(graph, weight=0.8)
        # last_reinforced is already set to today by upsert_edge

        ref = date.today().isoformat()
        decayed = graph.apply_confidence_decay(half_life_days=30, reference_date=ref)

        assert decayed == []

    def test_skip_legacy_edges_with_empty_last_reinforced(self, graph: GraphStore):
        """Edges with last_reinforced = '' (legacy) are not touched."""
        _add_edge(graph, weight=0.9)
        # Force to empty string (simulating pre-migration edge)
        cur = graph.conn.cursor()
        cur.execute("UPDATE kg_edges SET last_reinforced = ''")
        graph.conn.commit()

        ref = date.today().isoformat()
        decayed = graph.apply_confidence_decay(half_life_days=30, reference_date=ref)

        assert decayed == []

    def test_min_weight_clamping(self, graph: GraphStore):
        """Edge that would decay below min_weight is clamped to min_weight."""
        _add_edge(graph, weight=0.2)
        past = (date.today() - timedelta(days=180)).isoformat()
        _backdate_edge(graph, "Alice", "Acme", "WORKS_AT", past)

        ref = date.today().isoformat()
        decayed = graph.apply_confidence_decay(
            half_life_days=30, min_weight=0.1, reference_date=ref
        )

        assert len(decayed) == 1
        # 0.2 * 0.5^(180/30) = 0.2 * 0.5^6 = 0.2 * (1/64) ≈ 0.003 → clamped to 0.1
        assert decayed[0]["new_weight"] == pytest.approx(0.1, rel=1e-4)

    def test_multiple_edges_each_decay_proportionally(self, graph: GraphStore):
        """Multiple edges with different ages each decay independently."""
        graph.upsert_edge("Alice", "Acme", "WORKS_AT", 0.8, "", "h1")
        graph.upsert_edge("Bob", "Corp", "WORKS_AT", 0.6, "", "h2")

        today = date.today()
        _backdate_edge(graph, "Alice", "Acme", "WORKS_AT", (today - timedelta(days=30)).isoformat())
        _backdate_edge(graph, "Bob", "Corp", "WORKS_AT", (today - timedelta(days=60)).isoformat())

        ref = today.isoformat()
        decayed = graph.apply_confidence_decay(half_life_days=30, min_weight=0.01, reference_date=ref)

        assert len(decayed) == 2
        alice_rec = next(r for r in decayed if r["source"] == "Alice")
        bob_rec = next(r for r in decayed if r["source"] == "Bob")

        # Alice: 30 days → weight * 0.5^1 = 0.4
        assert alice_rec["new_weight"] == pytest.approx(0.4, rel=1e-3)
        # Bob: 60 days → weight * 0.5^2 = 0.6 * 0.25 = 0.15
        assert bob_rec["new_weight"] == pytest.approx(0.15, rel=1e-3)

    def test_zero_half_life_returns_empty(self, graph: GraphStore):
        """half_life_days=0 is a guard condition — returns empty list, no crash."""
        _add_edge(graph)
        past = (date.today() - timedelta(days=10)).isoformat()
        _backdate_edge(graph, "Alice", "Acme", "WORKS_AT", past)

        decayed = graph.apply_confidence_decay(half_life_days=0, reference_date=date.today().isoformat())
        assert decayed == []

    def test_half_life_formula_correctness(self, graph: GraphStore):
        """Verify exponential formula: weight * 0.5^(days/half_life)."""
        _add_edge(graph, weight=1.0)
        past = (date.today() - timedelta(days=45)).isoformat()
        _backdate_edge(graph, "Alice", "Acme", "WORKS_AT", past)

        ref = date.today().isoformat()
        decayed = graph.apply_confidence_decay(half_life_days=30, min_weight=0.01, reference_date=ref)

        expected = 1.0 * (0.5 ** (45 / 30))  # ≈ 0.3969
        assert len(decayed) == 1
        assert decayed[0]["new_weight"] == pytest.approx(expected, rel=1e-3)


# ── Phase 1: last_reinforced tracking ────────────────────────────────────────

class TestLastReinforcedTracking:

    def test_new_edge_insert_sets_last_reinforced(self, graph: GraphStore):
        """First upsert_edge call sets last_reinforced = today."""
        today = date.today().isoformat()
        graph.upsert_edge("Alice", "Acme", "WORKS_AT", 0.8, "", "h1")

        cur = graph.conn.cursor()
        cur.execute("SELECT last_reinforced FROM kg_edges WHERE source='Alice' AND target='Acme'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == today

    def test_re_upsert_refreshes_last_reinforced(self, graph: GraphStore):
        """Second upsert_edge call (ON CONFLICT) refreshes last_reinforced."""
        graph.upsert_edge("Alice", "Acme", "WORKS_AT", 0.8, "", "h1")

        # Backdate to simulate a stale edge
        past = (date.today() - timedelta(days=30)).isoformat()
        _backdate_edge(graph, "Alice", "Acme", "WORKS_AT", past)

        # Re-upsert — should refresh last_reinforced to today
        today = date.today().isoformat()
        graph.upsert_edge("Alice", "Acme", "WORKS_AT", 0.9, "new evidence", "h2")

        cur = graph.conn.cursor()
        cur.execute("SELECT last_reinforced FROM kg_edges WHERE source='Alice' AND target='Acme'")
        row = cur.fetchone()
        assert row[0] == today


# ── Phase 2: consolidate() service method ────────────────────────────────────

class TestConsolidateService:

    def test_empty_kg_returns_empty_sections(self, service):
        """Consolidate on empty DB returns all sections with zero counts, no error."""
        result = service.consolidate(half_life_days=90, older_than_days=30)

        assert "decay" in result
        assert "merge_suggestions" in result
        assert "stale_memories" in result
        assert result["decay"]["edges_decayed"] == 0
        assert result["decay"]["edges"] == []
        assert result["merge_suggestions"]["candidates"] == []
        assert result["merge_suggestions"]["auto_merged"] == []
        assert result["stale_memories"]["count"] == 0
        assert result["stale_memories"]["memories"] == []

    def test_decay_section_populated_when_stale_edges_exist(self, service):
        """Edges with old last_reinforced appear in decay section."""
        from kioku_lite.service import EntityInput, RelationshipInput

        # Save a memory and index an edge
        result = service.save_memory("Alice works at Acme")
        content_hash = result["content_hash"]
        service.kg_index(
            content_hash,
            entities=[EntityInput("Alice", "PERSON"), EntityInput("Acme", "ORG")],
            relationships=[RelationshipInput("Alice", "Acme", "WORKS_AT", 0.8, "Alice works at Acme")],
        )

        # Backdate the edge
        past = (date.today() - timedelta(days=60)).isoformat()
        _backdate_edge(service.db.graph, "Alice", "Acme", "WORKS_AT", past)

        result = service.consolidate(half_life_days=30, older_than_days=90)
        assert result["decay"]["edges_decayed"] == 1
        assert result["decay"]["half_life_days"] == 30
        edge = result["decay"]["edges"][0]
        assert edge["source"] == "Alice"
        assert edge["target"] == "Acme"
        assert edge["days_since_reinforced"] == 60

    def test_stale_memories_section_populated(self, service):
        """Old memories appear in stale_memories section."""
        from datetime import datetime, timezone

        # Insert memory directly with old date
        old_date = (date.today() - timedelta(days=60)).isoformat()
        service.db.memory.insert(
            content="An old memory",
            date=old_date,
            timestamp=old_date + "T00:00:00Z",
            content_hash="oldhash001",
        )

        result = service.consolidate(older_than_days=30)
        stale = result["stale_memories"]
        assert stale["count"] >= 1
        texts = [m["text"] for m in stale["memories"]]
        assert "An old memory" in texts

    def test_auto_merge_flag_passed_through(self, service):
        """auto_merge=True is passed through to dedup_scan (no crash, dict returned)."""
        result = service.consolidate(auto_merge=True)
        # Even with empty DB, result should have merge_suggestions
        assert "candidates" in result["merge_suggestions"]
        assert "auto_merged" in result["merge_suggestions"]


# ── Phase 3: CLI command ──────────────────────────────────────────────────────

class TestConsolidateCLI:

    def test_default_args_returns_valid_json(self, tmp_path: Path):
        """kioku-lite consolidate returns exit code 0 and parseable JSON."""
        env = make_env(tmp_path)
        result = runner.invoke(app, ["consolidate"], env=env)

        assert result.exit_code == 0, f"stderr: {result.output}"
        data = json.loads(result.output)
        assert "decay" in data
        assert "merge_suggestions" in data
        assert "stale_memories" in data

    def test_custom_half_life_reflected_in_output(self, tmp_path: Path):
        """--half-life is reflected in the decay section of output."""
        env = make_env(tmp_path)
        result = runner.invoke(app, ["consolidate", "--half-life", "7"], env=env)

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["decay"]["half_life_days"] == 7

    def test_older_than_reflected_in_stale_cutoff(self, tmp_path: Path):
        """--older-than changes the older_than date in stale_memories output."""
        env = make_env(tmp_path)
        result = runner.invoke(app, ["consolidate", "--older-than", "60"], env=env)

        assert result.exit_code == 0
        data = json.loads(result.output)
        expected_cutoff = (date.today() - timedelta(days=60)).isoformat()
        assert data["stale_memories"]["older_than"] == expected_cutoff

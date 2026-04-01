"""CLI integration tests via typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kioku_lite.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_service_singleton():
    """Reset the global _svc singleton and config between tests for proper isolation."""
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
    """Env vars for isolated, fake-embed CLI tests."""
    return {
        "KIOKU_LITE_USER_ID": "cli_test",
        "KIOKU_LITE_EMBED_PROVIDER": "fake",
        "KIOKU_LITE_EMBED_DIM": "128",
        "KIOKU_LITE_MEMORY_DIR": str(tmp_path / "memory"),
        "KIOKU_LITE_DATA_DIR": str(tmp_path / "data"),
    }


def invoke(args: list[str], env: dict, input_text: str | None = None):
    return runner.invoke(app, args, env=env, input=input_text)


# ── save ───────────────────────────────────────────────────────────────────────

class TestCLISave:
    def test_save_exits_0(self, tmp_path):
        result = invoke(["save", "test memory"], make_env(tmp_path))
        assert result.exit_code == 0, result.output

    def test_save_outputs_json(self, tmp_path):
        result = invoke(["save", "json output test"], make_env(tmp_path))
        data = json.loads(result.output)
        assert data["status"] == "saved"

    def test_save_returns_content_hash(self, tmp_path):
        result = invoke(["save", "hash test"], make_env(tmp_path))
        data = json.loads(result.output)
        assert "content_hash" in data
        assert len(data["content_hash"]) == 64

    def test_save_with_mood(self, tmp_path):
        result = invoke(["save", "happy day", "--mood", "happy"], make_env(tmp_path))
        data = json.loads(result.output)
        assert data["mood"] == "happy"

    def test_save_with_tags(self, tmp_path):
        result = invoke(["save", "tagged", "--tags", "work,meeting"], make_env(tmp_path))
        data = json.loads(result.output)
        assert data["tags"] == ["work", "meeting"]

    def test_save_with_event_time(self, tmp_path):
        result = invoke(["save", "past event", "--event-time", "2026-01-15"], make_env(tmp_path))
        data = json.loads(result.output)
        assert data["event_time"] == "2026-01-15"


# ── kg-index ───────────────────────────────────────────────────────────────────

class TestCLIKgIndex:
    def test_kg_index_exits_0(self, tmp_path):
        env = make_env(tmp_path)
        save_result = invoke(["save", "Gặp Hùng tại TBV"], env)
        h = json.loads(save_result.output)["content_hash"]
        entities = json.dumps([{"name": "Hùng", "type": "PERSON"}, {"name": "TBV", "type": "ORGANIZATION"}])
        result = invoke(["kg-index", h, "--entities", entities], env)
        assert result.exit_code == 0, result.output

    def test_kg_index_returns_indexed_status(self, tmp_path):
        env = make_env(tmp_path)
        save_result = invoke(["save", "Phúc làm ở LINE"], env)
        h = json.loads(save_result.output)["content_hash"]
        entities = json.dumps([{"name": "Phúc", "type": "PERSON"}])
        rels = json.dumps([{"source": "Phúc", "target": "LINE", "rel_type": "WORKS_AT", "weight": 0.9}])
        result = invoke(["kg-index", h, "--entities", entities, "--relationships", rels], env)
        data = json.loads(result.output)
        assert data["status"] == "indexed"
        assert data["entities_added"] == 1

    def test_kg_index_invalid_json_exits_1(self, tmp_path):
        env = make_env(tmp_path)
        result = invoke(["kg-index", "fakehash", "--entities", "not-json"], env)
        assert result.exit_code == 1

    def test_kg_index_no_entities_ok(self, tmp_path):
        env = make_env(tmp_path)
        save_result = invoke(["save", "no entities"], env)
        h = json.loads(save_result.output)["content_hash"]
        result = invoke(["kg-index", h], env)
        assert result.exit_code == 0


# ── kg-alias ───────────────────────────────────────────────────────────────────

class TestCLIKgAlias:
    def test_kg_alias_exits_0(self, tmp_path):
        env = make_env(tmp_path)
        aliases = json.dumps(["tôi", "mình"])
        result = invoke(["kg-alias", "Phúc", "--aliases", aliases], env)
        assert result.exit_code == 0, result.output

    def test_kg_alias_returns_ok(self, tmp_path):
        env = make_env(tmp_path)
        aliases = json.dumps(["phuc-nt"])
        result = invoke(["kg-alias", "Nguyễn Trọng Phúc", "--aliases", aliases], env)
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["canonical"] == "Nguyễn Trọng Phúc"

    def test_kg_alias_invalid_json_exits_1(self, tmp_path):
        result = invoke(["kg-alias", "Phúc", "--aliases", "bad-json"], make_env(tmp_path))
        assert result.exit_code == 1


# ── search ─────────────────────────────────────────────────────────────────────

class TestCLISearch:
    def test_search_exits_0(self, tmp_path):
        env = make_env(tmp_path)
        invoke(["save", "hello world"], env)
        result = invoke(["search", "hello"], env)
        assert result.exit_code == 0, result.output

    def test_search_outputs_json(self, tmp_path):
        env = make_env(tmp_path)
        invoke(["save", "search test entry"], env)
        result = invoke(["search", "search test"], env)
        data = json.loads(result.output)
        assert "results" in data
        assert "count" in data

    def test_search_with_limit(self, tmp_path):
        env = make_env(tmp_path)
        for i in range(10):
            invoke(["save", f"searchable content number {i}"], env)
        result = invoke(["search", "searchable content", "--limit", "3"], env)
        data = json.loads(result.output)
        assert data["count"] <= 3

    def test_search_no_results(self, tmp_path):
        env = make_env(tmp_path)
        # Insert content with no shared tokens with the query
        invoke(["save", "completely unrelated zebra content"], env)
        result = invoke(["search", "a4f8b2c7e1d09356_no_match"], env)
        data = json.loads(result.output)
        # BM25 must return 0 — vector may fuzzy-match (acceptable for FakeEmbedder)
        bm25_count = sum(1 for r in data["results"] if r["source"] == "bm25")
        assert bm25_count == 0

    def test_search_with_entities(self, tmp_path):
        env = make_env(tmp_path)
        result = invoke(["search", "query", "--entities", "Phúc,TBV"], env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["entities_used"] == ["Phúc", "TBV"]


# ── recall ─────────────────────────────────────────────────────────────────────

class TestCLIRecall:
    def test_recall_exits_0(self, tmp_path):
        result = invoke(["recall", "AnyEntity"], make_env(tmp_path))
        assert result.exit_code == 0, result.output

    def test_recall_outputs_json(self, tmp_path):
        result = invoke(["recall", "SomePerson"], make_env(tmp_path))
        data = json.loads(result.output)
        assert "entity" in data
        assert "nodes" in data


# ── connect ────────────────────────────────────────────────────────────────────

class TestCLIConnect:
    def test_connect_exits_0(self, tmp_path):
        result = invoke(["connect", "A", "B"], make_env(tmp_path))
        assert result.exit_code == 0, result.output

    def test_connect_no_path(self, tmp_path):
        result = invoke(["connect", "A", "B"], make_env(tmp_path))
        data = json.loads(result.output)
        assert data["paths"] == []


# ── entities ───────────────────────────────────────────────────────────────────

class TestCLIEntities:
    def test_entities_exits_0(self, tmp_path):
        result = invoke(["entities"], make_env(tmp_path))
        assert result.exit_code == 0

    def test_entities_outputs_json(self, tmp_path):
        result = invoke(["entities"], make_env(tmp_path))
        data = json.loads(result.output)
        assert "count" in data
        assert "entities" in data

    def test_entities_after_kg_index(self, tmp_path):
        env = make_env(tmp_path)
        save_res = invoke(["save", "test memory"], env)
        h = json.loads(save_res.output)["content_hash"]
        invoke(["kg-index", h, "--entities", '[{"name":"TestEntity","type":"TOPIC"}]'], env)
        result = invoke(["entities"], env)
        data = json.loads(result.output)
        assert data["count"] >= 1


# ── timeline ───────────────────────────────────────────────────────────────────

class TestCLITimeline:
    def test_timeline_exits_0(self, tmp_path):
        result = invoke(["timeline"], make_env(tmp_path))
        assert result.exit_code == 0

    def test_timeline_outputs_json(self, tmp_path):
        result = invoke(["timeline"], make_env(tmp_path))
        data = json.loads(result.output)
        assert "timeline" in data

    def test_timeline_after_saves(self, tmp_path):
        env = make_env(tmp_path)
        invoke(["save", "timeline entry one"], env)
        invoke(["save", "timeline entry two"], env)
        result = invoke(["timeline"], env)
        data = json.loads(result.output)
        # Service is re-created per-path due to env var isolation — exactly 2 entries
        assert data["count"] >= 2


# ── end-to-end workflow ────────────────────────────────────────────────────────

class TestE2EWorkflow:
    def test_full_2step_workflow(self, tmp_path):
        """Complete save → kg-index → search → recall workflow."""
        env = make_env(tmp_path)

        # Step 1: Save
        save_result = invoke(["save", "Hôm nay họp với Lan về dự án kioku-lite. Rất hứng khởi.", "--mood", "excited"], env)
        assert save_result.exit_code == 0
        save_data = json.loads(save_result.output)
        h = save_data["content_hash"]

        # Step 2: Agent extracts and indexes KG
        entities = json.dumps([
            {"name": "Lan", "type": "PERSON"},
            {"name": "dự án kioku-lite", "type": "TOPIC"},
        ])
        rels = json.dumps([
            {"source": "Lan", "target": "dự án kioku-lite", "rel_type": "INVOLVES", "weight": 0.8, "evidence": "họp về dự án kioku-lite"}
        ])
        kg_result = invoke(["kg-index", h, "--entities", entities, "--relationships", rels], env)
        assert kg_result.exit_code == 0
        assert json.loads(kg_result.output)["entities_added"] == 2

        # Step 3: Search finds it
        search_result = invoke(["search", "Lan", "--entities", "Lan"], env)
        assert search_result.exit_code == 0
        search_data = json.loads(search_result.output)
        assert search_data["count"] >= 1
        assert any("Lan" in r["content"] for r in search_data["results"])

        # Step 4: Recall shows connected entities
        recall_result = invoke(["recall", "Lan"], env)
        assert recall_result.exit_code == 0
        recall_data = json.loads(recall_result.output)
        node_names = [n["name"] for n in recall_data["nodes"]]
        assert "dự án kioku-lite" in node_names

    def test_alias_affects_search(self, tmp_path):
        """Alias registration should expand traversal to include canonical entity edges."""
        env = make_env(tmp_path)

        # Save and index with canonical name
        save_result = invoke(["save", "Phúc làm ở LINE Technology"], env)
        h = json.loads(save_result.output)["content_hash"]
        entities = json.dumps([{"name": "Nguyễn Trọng Phúc", "type": "PERSON"}, {"name": "LINE", "type": "ORGANIZATION"}])
        rels = json.dumps([{"source": "Nguyễn Trọng Phúc", "target": "LINE", "rel_type": "WORKS_AT", "weight": 0.9, "evidence": "làm ở LINE"}])
        invoke(["kg-index", h, "--entities", entities, "--relationships", rels], env)

        # Register alias
        invoke(["kg-alias", "Nguyễn Trọng Phúc", "--aliases", '["Phúc","tôi"]'], env)

        # Recall from alias should reach LINE
        recall_result = invoke(["recall", "Phúc"], env)
        recall_data = json.loads(recall_result.output)
        node_names = [n["name"] for n in recall_data["nodes"]]
        assert "LINE" in node_names


# ── kg-invalidate CLI ─────────────────────────────────────────────────────────

class TestCLIKgInvalidate:
    def test_kg_invalidate_exits_0(self, tmp_path):
        env = make_env(tmp_path)
        save_res = invoke(["save", "Phuc works at LINE"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "Phuc", "type": "PERSON"}, {"name": "LINE", "type": "ORG"}])
        rels = json.dumps([{"source": "Phuc", "target": "LINE", "rel_type": "WORKS_AT", "weight": 0.9}])
        invoke(["kg-index", h, "--entities", ents, "--relationships", rels], env)
        result = invoke(["kg-invalidate", "--source", "Phuc", "--target", "LINE"], env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["status"] == "invalidated"
        assert data["edges_updated"] == 1

    def test_kg_invalidate_no_match(self, tmp_path):
        env = make_env(tmp_path)
        result = invoke(["kg-invalidate", "--source", "X", "--target", "Y"], env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "no_match"

    def test_recall_include_historical(self, tmp_path):
        env = make_env(tmp_path)
        save_res = invoke(["save", "Phuc works at LINE"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "Phuc", "type": "PERSON"}, {"name": "LINE", "type": "ORG"}])
        rels = json.dumps([{"source": "Phuc", "target": "LINE", "rel_type": "WORKS_AT", "weight": 0.9}])
        invoke(["kg-index", h, "--entities", ents, "--relationships", rels], env)
        invoke(["kg-invalidate", "--source", "Phuc", "--target", "LINE", "--date", "2026-03-31"], env)
        # Default recall: no edges (invalidated)
        result = invoke(["recall", "Phuc"], env)
        data = json.loads(result.output)
        assert len(data["relationships"]) == 0
        # With --include-historical: edge is back
        result = invoke(["recall", "Phuc", "--include-historical"], env)
        data = json.loads(result.output)
        assert len(data["relationships"]) >= 1


# ── dedup-scan CLI ────────────────────────────────────────────────────────────

class TestCLIDedupScan:
    def test_dedup_scan_exits_0(self, tmp_path):
        env = make_env(tmp_path)
        # Add some entities first
        save_res = invoke(["save", "Phuc info"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "Phuc", "type": "PERSON"}])
        invoke(["kg-index", h, "--entities", ents], env)

        # Run dedup scan
        result = invoke(["dedup-scan"], env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "candidates" in data
        assert "auto_merged" in data
        assert "total_scanned" in data

    def test_dedup_scan_returns_json(self, tmp_path):
        env = make_env(tmp_path)
        result = invoke(["dedup-scan"], env)
        data = json.loads(result.output)
        assert isinstance(data["candidates"], list)
        assert isinstance(data["auto_merged"], list)

    def test_dedup_scan_empty_graph(self, tmp_path):
        env = make_env(tmp_path)
        result = invoke(["dedup-scan"], env)
        data = json.loads(result.output)
        assert data["total_scanned"] == 0

    def test_dedup_scan_auto_flag(self, tmp_path):
        env = make_env(tmp_path)
        # Add entities that might auto-merge
        save_res = invoke(["save", "Phuc info"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "Phuc", "type": "PERSON"}])
        invoke(["kg-index", h, "--entities", ents], env)

        # Run with --auto flag (should not crash)
        result = invoke(["dedup-scan", "--auto"], env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "auto_merged" in data


# ── merge CLI ──────────────────────────────────────────────────────────────────

class TestCLIMerge:
    def test_merge_exits_0(self, tmp_path):
        env = make_env(tmp_path)
        # Create two entities
        save_res = invoke(["save", "Phuc info"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "Phuc", "type": "PERSON"}, {"name": "Phúc", "type": "PERSON"}])
        invoke(["kg-index", h, "--entities", ents], env)

        # Merge them
        result = invoke(["merge", "Phuc", "Phúc"], env)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["status"] == "merged"

    def test_merge_returns_merged_status(self, tmp_path):
        env = make_env(tmp_path)
        save_res = invoke(["save", "Test merge"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "A", "type": "PERSON"}, {"name": "B", "type": "PERSON"}])
        invoke(["kg-index", h, "--entities", ents], env)

        result = invoke(["merge", "A", "B"], env)
        data = json.loads(result.output)
        assert data["status"] == "merged"
        assert data["source"] == "A"
        assert data["target"] == "B"

    def test_merge_nonexistent_entity(self, tmp_path):
        env = make_env(tmp_path)
        save_res = invoke(["save", "Single entity"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "OnlyOne", "type": "PERSON"}])
        invoke(["kg-index", h, "--entities", ents], env)

        # Try to merge with non-existent entity
        result = invoke(["merge", "OnlyOne", "NotExist"], env)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "skipped"

    def test_merge_removes_source_from_entities(self, tmp_path):
        env = make_env(tmp_path)
        save_res = invoke(["save", "Merge test"], env)
        h = json.loads(save_res.output)["content_hash"]
        ents = json.dumps([{"name": "X", "type": "PERSON"}, {"name": "Y", "type": "PERSON"}])
        invoke(["kg-index", h, "--entities", ents], env)

        # Check entities before merge
        entities_before = invoke(["entities", "--limit", "50"], env)
        data_before = json.loads(entities_before.output)
        names_before = [e["name"] for e in data_before["entities"]]
        assert "X" in names_before

        # Merge X into Y
        invoke(["merge", "X", "Y"], env)

        # Check entities after merge
        entities_after = invoke(["entities", "--limit", "50"], env)
        data_after = json.loads(entities_after.output)
        names_after = [e["name"] for e in data_after["entities"]]
        assert "X" not in names_after
        assert "Y" in names_after

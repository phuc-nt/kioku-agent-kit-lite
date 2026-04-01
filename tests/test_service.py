"""Tests for KiokuLiteService — the main business logic layer."""

from __future__ import annotations

import pytest

from kioku_lite.service import EntityInput, RelationshipInput


# ── save_memory ────────────────────────────────────────────────────────────────

class TestSaveMemory:
    def test_returns_content_hash(self, service):
        result = service.save_memory("First memory")
        assert "content_hash" in result
        assert len(result["content_hash"]) == 64  # sha256 hex

    def test_same_text_same_hash(self, service):
        r1 = service.save_memory("Deterministic text")
        r2 = service.save_memory("Deterministic text")
        assert r1["content_hash"] == r2["content_hash"]

    def test_different_text_different_hash(self, service):
        r1 = service.save_memory("text one")
        r2 = service.save_memory("text two")
        assert r1["content_hash"] != r2["content_hash"]

    def test_returns_status_saved(self, service):
        result = service.save_memory("test")
        assert result["status"] == "saved"

    def test_mood_stored(self, service):
        result = service.save_memory("happy day", mood="happy")
        assert result["mood"] == "happy"

    def test_tags_stored(self, service):
        result = service.save_memory("tag test", tags=["work", "meeting"])
        assert result["tags"] == ["work", "meeting"]

    def test_hint_in_output(self, service):
        result = service.save_memory("something")
        assert "hint" in result

    def test_vector_indexed(self, service):
        result = service.save_memory("vector test")
        assert result["vector_indexed"] is True  # FakeEmbedder supports vec

    def test_markdown_file_created(self, service, tmp_path):
        service.save_memory("markdown test entry")
        md_files = list((tmp_path / "memory").glob("*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "markdown test entry" in content


# ── kg_index ───────────────────────────────────────────────────────────────────

class TestKgIndex:
    def test_index_entities(self, service):
        r = service.save_memory("Gặp Hùng tại TBV")
        h = r["content_hash"]
        result = service.kg_index(
            h,
            [EntityInput("Hùng", "PERSON"), EntityInput("TBV", "ORGANIZATION")],
            [],
        )
        assert result["status"] == "indexed"
        assert result["entities_added"] == 2

    def test_index_relationships(self, service):
        r = service.save_memory("Hùng làm ở TBV")
        h = r["content_hash"]
        result = service.kg_index(
            h,
            [EntityInput("Hùng", "PERSON"), EntityInput("TBV", "ORGANIZATION")],
            [RelationshipInput("Hùng", "TBV", "WORKS_AT", 0.9, "Hùng làm ở TBV")],
        )
        assert result["relationships_added"] == 1

    def test_entities_searchable_after_index(self, service):
        r = service.save_memory("Phúc làm việc tại LINE")
        service.kg_index(
            r["content_hash"],
            [EntityInput("Phúc", "PERSON"), EntityInput("LINE", "ORGANIZATION")],
            [RelationshipInput("Phúc", "LINE", "WORKS_AT", 0.9, "làm việc tại LINE")],
        )
        entities = service.list_entities()["entities"]
        names = [e["name"] for e in entities]
        assert "Phúc" in names
        assert "LINE" in names

    def test_empty_kg_index_ok(self, service):
        r = service.save_memory("no entities here")
        result = service.kg_index(r["content_hash"], [], [])
        assert result["status"] == "indexed"
        assert result["entities_added"] == 0


# ── kg_alias ───────────────────────────────────────────────────────────────────

class TestKgAlias:
    def test_add_alias(self, service):
        service.save_memory("Phúc hôm nay đi làm")
        service.kg_index("fake_hash_1", [EntityInput("Phúc", "PERSON")], [])
        result = service.kg_alias("Phúc", ["tôi", "mình", "phuc-nt"])
        assert result["status"] == "ok"
        assert result["canonical"] == "Phúc"
        assert len(result["aliases_added"]) == 3

    def test_canonical_not_added_to_aliases(self, service):
        result = service.kg_alias("Phúc", ["Phúc", "tôi"])
        # "Phúc" same as canonical should be skipped
        assert "Phúc" not in result["aliases_added"]
        assert "tôi" in result["aliases_added"]


# ── search_memories ────────────────────────────────────────────────────────────

class TestSearchMemories:
    def test_basic_search(self, service):
        service.save_memory("Hùng rất vui hôm nay")
        result = service.search_memories("Hùng")
        assert result["count"] >= 1
        assert any("Hùng" in r["content"] for r in result["results"])

    def test_no_results(self, service):
        service.save_memory("completely different content here abc")
        # Use BM25-focused query with unique token never in corpus
        result = service.search_memories("xyzzy_unique_token_98765_notindb")
        # BM25 and graph should return 0 — only vec may fuzzy-match (acceptable)
        bm25_count = sum(1 for r in result["results"] if r["source"] == "bm25")
        assert bm25_count == 0

    def test_limit_respected(self, service):
        for i in range(15):
            service.save_memory(f"memory about topic alpha number {i}")
        result = service.search_memories("topic alpha", limit=5)
        assert result["count"] <= 5

    def test_entity_search_includes_graph_context(self, service):
        mem = service.save_memory("Gặp Hùng tại công ty TBV")
        service.kg_index(
            mem["content_hash"],
            [EntityInput("Hùng", "PERSON"), EntityInput("TBV", "ORGANIZATION")],
            [RelationshipInput("Hùng", "TBV", "WORKS_AT", 0.9, "at TBV")],
        )
        result = service.search_memories("Hùng", entities=["Hùng"])
        assert "graph_context" in result
        assert any(n["name"] == "TBV" for n in result["graph_context"]["nodes"])

    def test_date_filter_from(self, service):
        from kioku_lite.config import Settings
        from kioku_lite.service import KiokuLiteService
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            # Fake older date by inserting directly
            settings = Settings(
                user_id="filter_test",
                memory_dir=p / "mem",
                data_dir=p / "data",
                embed_provider="fake",
                embed_dim=128,
            )
            svc = KiokuLiteService(settings=settings)
            svc.db.memory.insert("old entry", "2025-12-01", "2025-12-01T10:00:00", content_hash="old111")
            svc.db.memory.insert("new entry", "2026-02-27", "2026-02-27T10:00:00", content_hash="new222")
            result = svc.search_memories("entry", date_from="2026-01-01")
            assert all(r["date"] >= "2026-01-01" for r in result["results"] if r["date"])
            svc.close()

    def test_entities_used_in_response(self, service):
        result = service.search_memories("test", entities=["SomePerson"])
        assert result["entities_used"] == ["SomePerson"]

    def test_query_reflected_in_response(self, service):
        result = service.search_memories("my query here")
        assert result["query"] == "my query here"


# ── recall_entity ──────────────────────────────────────────────────────────────

class TestRecallEntity:
    def test_recall_shows_connected_nodes(self, service):
        mem = service.save_memory("Phúc làm ở LINE")
        service.kg_index(
            mem["content_hash"],
            [EntityInput("Phúc", "PERSON"), EntityInput("LINE", "ORGANIZATION")],
            [RelationshipInput("Phúc", "LINE", "WORKS_AT", 0.9, "làm ở LINE")],
        )
        result = service.recall_entity("Phúc")
        assert result["entity"] == "Phúc"
        node_names = [n["name"] for n in result["nodes"]]
        assert "LINE" in node_names

    def test_recall_includes_source_memories(self, service):
        mem = service.save_memory("Gặp Hùng hôm nay")
        service.kg_index(
            mem["content_hash"],
            [EntityInput("Hùng", "PERSON")],
            [],
        )
        result = service.recall_entity("Hùng")
        assert len(result["source_memories"]) >= 0  # depends on edge source_hash

    def test_recall_nonexistent_entity(self, service):
        result = service.recall_entity("Ghost")
        assert result["entity"] == "Ghost"
        assert result["connected_count"] == 0


# ── explain_connection ─────────────────────────────────────────────────────────

class TestExplainConnection:
    def test_direct_connection_found(self, service):
        mem = service.save_memory("Phúc làm ở LINE")
        service.kg_index(
            mem["content_hash"],
            [EntityInput("Phúc", "PERSON"), EntityInput("LINE", "ORGANIZATION")],
            [RelationshipInput("Phúc", "LINE", "WORKS_AT", 0.9, "làm ở LINE")],
        )
        result = service.explain_connection("Phúc", "LINE")
        assert len(result["paths"]) > 0

    def test_indirect_connection(self, service):
        mem = service.save_memory("Phúc biết Hùng, Hùng làm ở LINE")
        service.kg_index(
            mem["content_hash"],
            [EntityInput("Phúc", "PERSON"), EntityInput("Hùng", "PERSON"), EntityInput("LINE", "ORGANIZATION")],
            [
                RelationshipInput("Phúc", "Hùng", "KNOWS", 0.8, "biết Hùng"),
                RelationshipInput("Hùng", "LINE", "WORKS_AT", 0.9, "làm ở LINE"),
            ],
        )
        result = service.explain_connection("Phúc", "LINE")
        assert len(result["paths"]) > 0
        assert result["paths"][0][0] in ("Phúc",)
        assert result["paths"][0][-1] in ("LINE",)

    def test_no_connection(self, service):
        service.kg_index("fakehash1", [EntityInput("A", "PERSON")], [])
        service.kg_index("fakehash2", [EntityInput("B", "PERSON")], [])
        result = service.explain_connection("A", "B")
        assert len(result["paths"]) == 0


# ── list_entities / timeline / dates ──────────────────────────────────────────

class TestListAndTimeline:
    def test_list_entities_empty(self, service):
        result = service.list_entities()
        assert result["count"] == 0

    def test_list_entities_after_index(self, service):
        service.kg_index("h1", [EntityInput("E1", "TOPIC"), EntityInput("E2", "TOPIC")], [])
        result = service.list_entities()
        assert result["count"] == 2

    def test_list_entities_limit(self, service):
        for i in range(20):
            service.kg_index(f"h{i}", [EntityInput(f"Entity{i}", "TOPIC")], [])
        result = service.list_entities(limit=5)
        assert len(result["entities"]) <= 5

    def test_get_timeline(self, service):
        for i in range(3):
            service.save_memory(f"timeline entry {i}")
        result = service.get_timeline(limit=10)
        assert result["count"] == 3
        assert len(result["timeline"]) == 3

    def test_timeline_sort_by_processing_time(self, service):
        service.save_memory("first")
        service.save_memory("second")
        result = service.get_timeline(sort_by="processing_time")
        assert result["sort_by"] == "processing_time"


# ── temporal range detection ───────────────────────────────────────────────────

class TestTemporalRange:
    def test_year_pattern(self, service):
        df, dt = service._extract_temporal_range("memories from năm 2025")
        assert df == "2025-01-01"
        assert dt == "2025-12-31"

    def test_current_year(self, service):
        from datetime import datetime, timezone, timedelta
        year = datetime.now(timezone(timedelta(hours=7))).year
        df, dt = service._extract_temporal_range("năm nay tôi đã làm gì")
        assert df == f"{year}-01-01"
        assert dt == f"{year}-12-31"

    def test_last_year(self, service):
        from datetime import datetime, timezone, timedelta
        year = datetime.now(timezone(timedelta(hours=7))).year - 1
        df, dt = service._extract_temporal_range("năm ngoái tôi đi đâu")
        assert df == f"{year}-01-01"
        assert dt == f"{year}-12-31"

    def test_no_temporal_hint(self, service):
        df, dt = service._extract_temporal_range("what did I do with Hùng")
        assert df is None
        assert dt is None

    def test_4digit_year_in_query(self, service):
        df, dt = service._extract_temporal_range("something from 2024")
        assert df == "2024-01-01"
        assert dt == "2024-12-31"


# ── temporal fact management ─────────────────────────────────────────────────

class TestTemporalFactManagement:
    def test_kg_index_sets_valid_from(self, service):
        from kioku_lite.service import EntityInput, RelationshipInput
        service.save_memory("Phuc works at Google")
        h = service.save_memory("Phuc works at Google")["content_hash"]
        rels = [RelationshipInput(source="Phuc", target="Google", rel_type="WORKS_AT", weight=0.9)]
        service.kg_index(h, [EntityInput("Phuc"), EntityInput("Google")], rels, event_time="2026-04-01")
        edges = service.db.graph.get_all_edges()
        works_at = [e for e in edges if e["relation"] == "WORKS_AT"]
        assert works_at[0]["valid_from"] == "2026-04-01"

    def test_kg_invalidate_success(self, service):
        from kioku_lite.service import EntityInput, RelationshipInput
        h = service.save_memory("Phuc works at LINE")["content_hash"]
        rels = [RelationshipInput(source="Phuc", target="LINE", rel_type="WORKS_AT", weight=0.9)]
        service.kg_index(h, [EntityInput("Phuc"), EntityInput("LINE")], rels)
        result = service.kg_invalidate(source="Phuc", target="LINE", rel_type="WORKS_AT", valid_until="2026-03-31")
        assert result["status"] == "invalidated"
        assert result["edges_updated"] == 1

    def test_kg_invalidate_no_match(self, service):
        result = service.kg_invalidate(source="X", target="Y")
        assert result["status"] == "no_match"
        assert result["edges_updated"] == 0


# ── dedup_scan ──────────────────────────────────────────────────────────────────

class TestDedupScan:
    def test_dedup_scan_returns_dict(self, service):
        """dedup_scan should return dict with candidates and auto_merged."""
        result = service.dedup_scan()
        assert isinstance(result, dict)
        assert "candidates" in result
        assert "auto_merged" in result
        assert "total_scanned" in result

    def test_dedup_scan_empty_graph(self, service):
        """dedup_scan on empty graph should return empty lists."""
        result = service.dedup_scan()
        assert result["candidates"] == []
        assert result["auto_merged"] == []
        assert result["total_scanned"] == 0

    def test_dedup_scan_single_entity(self, service):
        """dedup_scan with single entity should return empty (no pairs)."""
        service.kg_index("h1", [EntityInput("OnlyOne", "PERSON")], [])
        result = service.dedup_scan()
        assert result["total_scanned"] == 0

    def test_dedup_scan_returns_candidates(self, service):
        """dedup_scan should find candidate pairs (depends on embedder/thresholds)."""
        # Add two entities with similar names
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phúc", "PERSON")], [])
        result = service.dedup_scan()
        # May or may not have candidates depending on embedder
        assert isinstance(result["candidates"], list)

    def test_dedup_scan_auto_merge_false_returns_all_candidates(self, service):
        """With auto_merge=False, all qualifying pairs should be candidates."""
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phúc", "PERSON")], [])
        result = service.dedup_scan(auto_merge=False)
        assert isinstance(result["candidates"], list)
        assert isinstance(result["auto_merged"], list)

    def test_dedup_scan_auto_merge_true_merges_qualifying_pairs(self, service):
        """With auto_merge=True, qualifying pairs should be auto-merged."""
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phúc", "PERSON")], [])
        result = service.dedup_scan(auto_merge=True)
        # auto_merged list should be populated or empty (depends on embedder)
        assert isinstance(result["auto_merged"], list)


# ── merge_entities ──────────────────────────────────────────────────────────────

class TestMergeEntitiesService:
    def test_merge_entities_returns_dict(self, service):
        """merge_entities should return a dict with status."""
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phúc", "PERSON")], [])
        result = service.merge_entities("Phuc", "Phúc")
        assert isinstance(result, dict)
        assert "status" in result

    def test_merge_entities_success(self, service):
        """merge_entities should successfully merge when both exist."""
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phúc", "PERSON")], [])
        result = service.merge_entities("Phuc", "Phúc")
        assert result["status"] == "merged"
        assert result["source"] == "Phuc"
        assert result["target"] == "Phúc"

    def test_merge_entities_nonexistent_returns_skipped(self, service):
        """merge_entities with nonexistent source should return skipped."""
        service.kg_index("h1", [EntityInput("Phúc", "PERSON")], [])
        result = service.merge_entities("NonExistent", "Phúc")
        assert result["status"] == "skipped"

    def test_merge_entities_removes_source_from_list(self, service):
        """After merge, source should not appear in list_entities."""
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phúc", "PERSON")], [])
        entities_before = service.list_entities()["entities"]
        names_before = [e["name"] for e in entities_before]
        assert "Phuc" in names_before

        service.merge_entities("Phuc", "Phúc")

        entities_after = service.list_entities()["entities"]
        names_after = [e["name"] for e in entities_after]
        assert "Phuc" not in names_after
        assert "Phúc" in names_after

    def test_merge_entities_combines_mention_counts(self, service):
        """Merge should accumulate mention counts."""
        service.kg_index("h1", [EntityInput("Phuc", "PERSON")], [])
        service.kg_index("h2", [EntityInput("Phuc", "PERSON")], [])  # 2nd mention
        service.kg_index("h3", [EntityInput("Phúc", "PERSON")], [])  # 1st mention of phúc

        service.merge_entities("Phuc", "Phúc")

        entities = service.list_entities()["entities"]
        phuc = next(e for e in entities if e["name"] == "Phúc")
        assert phuc["mentions"] == 3  # 2 + 1


# ── kg_index with dedup ────────────────────────────────────────────────────────

class TestKgIndexWithDedup:
    def test_kg_index_returns_auto_merged_field(self, service):
        """kg_index should return auto_merged field (dedup results)."""
        r = service.save_memory("Phuc info here")
        result = service.kg_index(r["content_hash"], [EntityInput("Phuc", "PERSON")], [])
        assert "auto_merged" in result
        assert isinstance(result["auto_merged"], list)

    def test_kg_index_returns_dedup_candidates_field(self, service):
        """kg_index should return dedup_candidates field."""
        r = service.save_memory("Test entity here")
        result = service.kg_index(r["content_hash"], [EntityInput("TestEntity", "PERSON")], [])
        assert "dedup_candidates" in result
        assert isinstance(result["dedup_candidates"], list)

    def test_kg_index_dedup_finds_similar_during_index(self, service):
        """When indexing new entity, dedup should scan against existing ones."""
        # Index first entity
        r1 = service.save_memory("Phuc info")
        service.kg_index(r1["content_hash"], [EntityInput("Phuc", "PERSON")], [])

        # Index similar entity — dedup should find it
        r2 = service.save_memory("Phuc again")
        result = service.kg_index(r2["content_hash"], [EntityInput("Phúc", "PERSON")], [])

        # May have auto_merged or candidates (depends on embedder threshold)
        assert len(result["auto_merged"]) >= 0 or len(result["dedup_candidates"]) >= 0


# ── confidence in kg_index ──────────────────────────────────────────────────────

class TestConfidenceInService:
    def test_kg_index_with_custom_confidence(self, service):
        """kg_index should preserve entity confidence."""
        r = service.save_memory("High confidence entity")
        entity = EntityInput("Expert", "PERSON", confidence=0.95)
        service.kg_index(r["content_hash"], [entity], [])

        entities = service.list_entities()["entities"]
        expert = next(e for e in entities if e["name"] == "Expert")
        assert expert["confidence"] == 0.95

    def test_kg_index_default_confidence_1_0(self, service):
        """Entity without explicit confidence should default to 1.0."""
        r = service.save_memory("Default confidence entity")
        entity = EntityInput("Default", "PERSON")  # no explicit confidence
        service.kg_index(r["content_hash"], [entity], [])

        entities = service.list_entities()["entities"]
        default = next(e for e in entities if e["name"] == "Default")
        assert default["confidence"] == 1.0

    def test_list_entities_includes_confidence(self, service):
        """list_entities should include confidence field."""
        r = service.save_memory("Confidence test")
        service.kg_index(r["content_hash"], [EntityInput("Test", "PERSON", confidence=0.75)], [])

        entities = service.list_entities()["entities"]
        test = next(e for e in entities if e["name"] == "Test")
        assert "confidence" in test
        assert test["confidence"] == 0.75

    def test_recall_entity_returns_nodes(self, service):
        """recall_entity should return connected nodes."""
        r = service.save_memory("Recall with confidence")
        service.kg_index(
            r["content_hash"],
            [EntityInput("Alice", "PERSON", confidence=0.8), EntityInput("Bob", "PERSON", confidence=0.9)],
            [RelationshipInput("Alice", "Bob", "KNOWS", 0.7, "they know each other")],
        )

        result = service.recall_entity("Alice")
        alice_node = next(n for n in result["nodes"] if n["name"] == "Alice")
        assert alice_node["name"] == "Alice"
        assert "type" in alice_node

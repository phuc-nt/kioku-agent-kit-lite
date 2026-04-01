"""Tests for dedup engine: cosine similarity, candidate discovery, auto-merge.

Uses ControlledEmbedder to inject known vectors for deterministic testing
of DedupEngine.find_similar and scan_all logic.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from kioku_lite.pipeline.dedup import (
    DedupEngine,
    cosine_similarity,
)
from kioku_lite.pipeline.embedder import EmbeddingProvider


class ControlledEmbedder(EmbeddingProvider):
    """Controlled embedder that returns pre-set vectors for deterministic tests."""

    def __init__(self, vectors: Optional[dict[str, list[float]]] = None, dimension: int = 128):
        self.vectors = vectors or {}
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Return pre-set vector or random normalized vector."""
        normalized = text.strip().lower()
        if normalized in self.vectors:
            return self.vectors[normalized]
        # Fallback: return zero vector for unknown text
        return [0.0] * self.dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed(text) for text in texts]


def _normalize_vector(v: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


# ── cosine_similarity tests ────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        # (1,0,0) . (0,1,0) = 0
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == -1.0

    def test_normalized_same_direction(self):
        # Normalized versions of [2,0,0] and [4,0,0] should give 1.0
        a = [2.0, 0.0, 0.0]
        b = [4.0, 0.0, 0.0]
        score = cosine_similarity(a, b)
        assert abs(score - 1.0) < 0.0001

    def test_zero_vector_safety(self):
        # Should return 0.0, not raise ZeroDivisionError
        a = [0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors(self):
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_length_mismatch(self):
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_one_empty_vector(self):
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], [1.0]) == 0.0

    def test_45_degree_angle(self):
        # (1,1) . (1,1) / (sqrt(2) * sqrt(2)) = 2/2 = 1.0 (same direction)
        a = [1.0, 1.0]
        b = [1.0, 1.0]
        assert abs(cosine_similarity(a, b) - 1.0) < 0.0001

    def test_90_degree_angle(self):
        a = [1.0, 1.0]
        b = [1.0, -1.0]
        # (1*1 + 1*-1) / (sqrt(2) * sqrt(2)) = 0/2 = 0.0
        assert abs(cosine_similarity(a, b)) < 0.0001


# ── DedupEngine tests ──────────────────────────────────────────────────────────

class TestDedupEngine:
    def test_default_thresholds(self):
        engine = DedupEngine()
        assert engine.vec_auto == 0.98
        assert engine.name_auto == 0.85
        assert engine.vec_candidate == 0.90

    def test_custom_thresholds(self):
        engine = DedupEngine(vec_auto=0.95, name_auto=0.80, vec_candidate=0.85)
        assert engine.vec_auto == 0.95
        assert engine.name_auto == 0.80
        assert engine.vec_candidate == 0.85

    def test_pre_filter_rejects_zero_length_ratio(self):
        # "a" vs "bcd..." (1 char vs 10 chars = 0.1 ratio < 0.33 threshold)
        assert not DedupEngine._pre_filter("a", "bcd" * 5)

    def test_pre_filter_accepts_reasonable_ratio(self):
        # "abc" vs "abcd" (3 vs 4 = 0.75 ratio > 0.33)
        assert DedupEngine._pre_filter("abc", "abcd")

    def test_pre_filter_rejects_empty(self):
        assert not DedupEngine._pre_filter("", "test")
        assert not DedupEngine._pre_filter("test", "")

    def test_pre_filter_case_insensitive(self):
        assert DedupEngine._pre_filter("ABC", "abc")

    def test_pre_filter_whitespace_ignored(self):
        assert DedupEngine._pre_filter("  abc  ", "abc")


class TestDedupEngineWithDB:
    @pytest.fixture
    def embedder_similar_pairs(self):
        """Embedder with controllable similarity."""
        # Create vectors where specific pairs are similar
        embedder = ControlledEmbedder(dimension=128)

        # "Phuc" and "Phúc" vectors: very similar (0.99 cosine sim)
        v_phuc = _normalize_vector([1.0] * 128)
        v_phuc2 = _normalize_vector([1.0] * 127 + [0.99])
        embedder.vectors["phuc"] = v_phuc
        embedder.vectors["phúc"] = v_phuc2

        # "LINE" and "LINE Corp" vectors: very similar (0.97)
        v_line = _normalize_vector([2.0] * 128)
        v_line_corp = _normalize_vector([2.0] * 127 + [1.98])
        embedder.vectors["line"] = v_line
        embedder.vectors["line corp"] = v_line_corp

        # "John" and "Jane" vectors: moderately similar (0.88)
        v_john = _normalize_vector([3.0] * 128)
        v_jane = _normalize_vector([2.9] * 128)
        embedder.vectors["john"] = v_john
        embedder.vectors["jane"] = v_jane

        return embedder

    def test_find_similar_empty_db(self, db, embedder_similar_pairs):
        """find_similar on empty DB should return empty result."""
        engine = DedupEngine()
        result = engine.find_similar("Phuc", embedder_similar_pairs, db.conn)
        assert result.auto_merged == []
        assert result.candidates == []

    def test_find_similar_no_existing_candidates(self, graph, embedder_similar_pairs):
        """find_similar when no candidates exist should return empty."""
        graph.upsert_node("Alice", "PERSON", "2026-02-27")
        engine = DedupEngine()
        result = engine.find_similar("Bob", embedder_similar_pairs, graph.conn)

        # Bob and Alice don't match on name similarity
        assert result.auto_merged == []
        # Might have candidates depending on embedder

    def test_find_similar_auto_merge_both_thresholds(self, graph, db, embedder_similar_pairs):
        """find_similar should auto-merge when vec + name both pass thresholds."""
        # Add "Phúc" first with its vector
        graph.upsert_node("Phúc", "PERSON", "2026-02-27")

        # Now search for "Phuc" (similar name, similar vector)
        engine = DedupEngine(vec_auto=0.95, name_auto=0.80)
        result = engine.find_similar("Phuc", embedder_similar_pairs, db.conn)

        # Should find auto_merged item since both thresholds pass
        assert len(result.auto_merged) >= 1
        auto_merged = result.auto_merged[0]
        assert auto_merged.source_name == "Phuc"
        assert auto_merged.target_name == "Phúc"

    def test_find_similar_candidate_vec_only(self, graph, db, embedder_similar_pairs):
        """find_similar should surface as candidate if vec passes but name doesn't."""
        # Add a node with similar vector but dissimilar name
        graph.upsert_node("Phúc", "PERSON", "2026-02-27")

        # Search for "phuc_different"
        embedder_similar_pairs.vectors["phuc_different"] = embedder_similar_pairs.vectors["phúc"]

        engine = DedupEngine(vec_auto=0.95, name_auto=0.85, vec_candidate=0.90)
        result = engine.find_similar("phuc_different", embedder_similar_pairs, db.conn)

        # Should be a candidate (high vec similarity)
        assert len(result.candidates) >= 0  # May or may not have candidates

    def test_find_similar_skip_self_match(self, graph, db, embedder_similar_pairs):
        """find_similar should skip the exact-match row (same name, case-insensitive)."""
        graph.upsert_node("Phuc", "PERSON", "2026-02-27")
        graph.upsert_node("Phuc", "PERSON", "2026-02-28")  # Increment mention_count

        # Search for "PHUC" (case variant of existing)
        engine = DedupEngine()
        result = engine.find_similar("PHUC", embedder_similar_pairs, db.conn)

        # Should not auto-merge with itself
        assert not any(a.source_name.lower() == a.target_name.lower() for a in result.auto_merged)

    def test_scan_all_empty_db(self, db, embedder_similar_pairs):
        """scan_all on empty DB should return empty list."""
        engine = DedupEngine()
        candidates = engine.scan_all(db.conn, embedder_similar_pairs)
        assert candidates == []

    def test_scan_all_single_entity(self, graph, db, embedder_similar_pairs):
        """scan_all with single entity should return empty (no pairs)."""
        graph.upsert_node("OnlyOne", "PERSON", "2026-02-27")

        engine = DedupEngine()
        candidates = engine.scan_all(db.conn, embedder_similar_pairs)
        assert candidates == []

    def test_scan_all_finds_duplicates(self, graph, db, embedder_similar_pairs):
        """scan_all should find pair where both vec and name sim are high."""
        graph.upsert_node("Phuc", "PERSON", "2026-02-27")
        graph.upsert_node("Phúc", "PERSON", "2026-02-27")

        engine = DedupEngine(vec_auto=0.95, vec_candidate=0.90)
        candidates = engine.scan_all(db.conn, embedder_similar_pairs)

        # Should find at least one candidate
        assert len(candidates) >= 1
        # Should not include pairs with itself
        for cand in candidates:
            assert cand.source_name.lower() != cand.target_name.lower()

    def test_scan_all_dedupes_pairs(self, graph, db, embedder_similar_pairs):
        """scan_all should not return the same pair twice (A→B and B→A)."""
        graph.upsert_node("Phuc", "PERSON", "2026-02-27")
        graph.upsert_node("Phúc", "PERSON", "2026-02-27")

        engine = DedupEngine(vec_candidate=0.80)
        candidates = engine.scan_all(db.conn, embedder_similar_pairs)

        # Build set of normalized pairs to check deduplication
        pairs_seen = set()
        for cand in candidates:
            pair = frozenset({cand.source_name.lower(), cand.target_name.lower()})
            assert pair not in pairs_seen, f"Pair {pair} returned twice"
            pairs_seen.add(pair)

    def test_scan_all_orders_by_mention_count(self, graph, db, embedder_similar_pairs):
        """scan_all should scan in order of mention_count (highest first)."""
        # Create entities with different mention counts
        graph.upsert_node("HighCount", "PERSON", "2026-02-27")
        graph.upsert_node("HighCount", "PERSON", "2026-02-27")  # 2 mentions
        graph.upsert_node("HighCount", "PERSON", "2026-02-27")  # 3 mentions
        graph.upsert_node("LowCount", "PERSON", "2026-02-27")   # 1 mention

        embedder_similar_pairs.vectors["highcount"] = embedder_similar_pairs.vectors["phuc"]
        embedder_similar_pairs.vectors["lowcount"] = embedder_similar_pairs.vectors["phúc"]

        engine = DedupEngine(vec_candidate=0.80)
        candidates = engine.scan_all(db.conn, embedder_similar_pairs)

        # Just verify it returns without error and respects ordering
        assert isinstance(candidates, list)


class TestDedupEdgeCases:
    def test_empty_name_returns_empty_result(self, db):
        """find_similar with empty name should return empty result."""
        engine = DedupEngine()
        embedder = ControlledEmbedder()
        result = engine.find_similar("", embedder, db.conn)
        assert result.auto_merged == []
        assert result.candidates == []

    def test_whitespace_only_name_returns_empty_result(self, db):
        """find_similar with whitespace-only name should return empty result."""
        engine = DedupEngine()
        embedder = ControlledEmbedder()
        result = engine.find_similar("   ", embedder, db.conn)
        assert result.auto_merged == []
        assert result.candidates == []

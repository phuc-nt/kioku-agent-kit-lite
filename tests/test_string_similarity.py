"""Tests for Jaro-Winkler string similarity (pure Python, no external deps)."""

from __future__ import annotations

from kioku_lite.utils.string_similarity import (
    jaro_similarity,
    jaro_winkler_similarity,
)


class TestJaroSimilarity:
    def test_identical_strings(self):
        assert jaro_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        score = jaro_similarity("abc", "xyz")
        assert 0.0 <= score < 0.3

    def test_similar_strings(self):
        # Classic example: martha/marhta should score ~0.944
        score = jaro_similarity("martha", "marhta")
        assert 0.94 < score < 0.96

    def test_empty_both_strings(self):
        assert jaro_similarity("", "") == 1.0

    def test_empty_first_string(self):
        assert jaro_similarity("", "abc") == 0.0

    def test_empty_second_string(self):
        assert jaro_similarity("abc", "") == 0.0

    def test_single_char_identical(self):
        assert jaro_similarity("a", "a") == 1.0

    def test_single_char_different(self):
        assert jaro_similarity("a", "b") == 0.0

    def test_vietnamese_diacritics(self):
        # Phúc vs Phúc (identical including diacritics)
        score = jaro_similarity("Phúc", "Phúc")
        assert score == 1.0

    def test_case_insensitive(self):
        score1 = jaro_similarity("Phuc", "phuc")
        score2 = jaro_similarity("PHUC", "phuc")
        assert score1 == 1.0
        assert score2 == 1.0

    def test_whitespace_normalized(self):
        # Leading/trailing whitespace should be stripped
        score = jaro_similarity("  abc  ", "abc")
        assert score == 1.0

    def test_partial_match(self):
        # "sat" and "cat" share 'a' and 't'
        score = jaro_similarity("sat", "cat")
        assert 0.5 < score < 0.9


class TestJaroWinklerSimilarity:
    def test_identical_strings(self):
        assert jaro_winkler_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        score = jaro_winkler_similarity("abc", "xyz")
        assert 0.0 <= score < 0.3

    def test_similar_strings_martha_marhta(self):
        # With prefix bonus, should be ~0.961
        score = jaro_winkler_similarity("martha", "marhta")
        assert 0.95 < score < 0.97

    def test_empty_strings(self):
        assert jaro_winkler_similarity("", "") == 1.0

    def test_empty_vs_nonempty(self):
        assert jaro_winkler_similarity("", "abc") == 0.0
        assert jaro_winkler_similarity("abc", "") == 0.0

    def test_single_char(self):
        assert jaro_winkler_similarity("a", "a") == 1.0

    def test_case_insensitive(self):
        assert jaro_winkler_similarity("PHUC", "phuc") == 1.0

    def test_vietnamese_names(self):
        # "Phúc" vs "Phuc" should be high but not identical
        score = jaro_winkler_similarity("Phúc", "Phuc")
        # Both have 'P', 'h' in common, but different in diacritics
        # This tests that the algorithm handles Unicode well
        assert 0.7 < score < 1.0

    def test_prefix_bonus_applied(self):
        # Strings with common prefix should score higher than jaro alone
        jaro_score = jaro_similarity("prefix_one", "prefix_two")
        winkler_score = jaro_winkler_similarity("prefix_one", "prefix_two")
        # Jaro-Winkler adds bonus for matching prefix (up to 4 chars)
        assert winkler_score > jaro_score

    def test_prefix_exactly_4_chars(self):
        # Prefix bonus only uses first 4 chars
        # "abcd" and "abcdefgh" share "abcd" -> should get max prefix bonus
        score = jaro_winkler_similarity("abcd", "abcdefgh")
        assert score > 0.8

    def test_whitespace_stripped(self):
        score = jaro_winkler_similarity("  test  ", "test")
        assert score == 1.0

    def test_prefix_weight_parameter(self):
        # With default prefix_weight=0.1, common prefix boosts score
        default_score = jaro_winkler_similarity("test_a", "test_b")
        # No direct way to test custom weight without modifying function,
        # but we verify the default works
        assert 0.5 < default_score < 1.0


class TestEdgeCases:
    def test_very_long_strings(self):
        long_a = "a" * 100
        long_b = "a" * 100
        assert jaro_winkler_similarity(long_a, long_b) == 1.0

    def test_very_long_different(self):
        long_a = "a" * 100
        long_b = "b" * 100
        assert jaro_winkler_similarity(long_a, long_b) == 0.0

    def test_unicode_mixed(self):
        # Vietnamese + English
        score = jaro_winkler_similarity("Phúc Nguyễn", "Phuc Nguyen")
        # Should be reasonably high despite diacritic differences
        assert 0.6 < score < 1.0

    def test_numbers_in_string(self):
        score = jaro_winkler_similarity("user123", "user124")
        assert 0.5 < score < 1.0

    def test_special_characters(self):
        score = jaro_winkler_similarity("test@email.com", "test@email.com")
        assert score == 1.0

    def test_special_chars_different(self):
        score = jaro_winkler_similarity("test@email.com", "test#email.com")
        # Only @ vs # differ
        assert 0.8 < score < 1.0

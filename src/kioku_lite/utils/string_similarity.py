"""Pure-Python Jaro-Winkler string similarity.

No external dependencies. Handles Unicode (Vietnamese diacritics, etc.).
"""

from __future__ import annotations


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity between two strings. Returns float in [0.0, 1.0]."""
    s1 = s1.strip().lower()
    s2 = s2.strip().lower()

    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    match_dist = max(match_dist, 0)

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Count matching characters
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Jaro-Winkler similarity. Boosts score for common prefix (up to 4 chars).

    Returns float in [0.0, 1.0]. Normalizes inputs via strip().lower().
    """
    s1_norm = s1.strip().lower()
    s2_norm = s2.strip().lower()

    jaro = jaro_similarity(s1_norm, s2_norm)

    # Common prefix length (max 4)
    prefix_len = 0
    for c1, c2 in zip(s1_norm[:4], s2_norm[:4]):
        if c1 == c2:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1.0 - jaro)

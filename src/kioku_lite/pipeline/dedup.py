"""Dedup engine: find and merge near-duplicate entities in the knowledge graph.

Uses cosine similarity on embeddings + Jaro-Winkler name similarity
to classify entity pairs as auto-merge, candidate, or skip.

No external dependencies.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

from kioku_lite.pipeline.embedder import EmbeddingProvider
from kioku_lite.utils.string_similarity import jaro_winkler_similarity


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors. Returns 0.0 for zero vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class DedupCandidate:
    """A pair of entities that may be duplicates (below auto-merge threshold)."""
    source_name: str
    target_name: str
    vector_sim: float
    name_sim: float


@dataclass
class MergeAction:
    """An entity pair that qualifies for auto-merge."""
    source_name: str
    target_name: str
    vector_sim: float
    name_sim: float


@dataclass
class DedupResult:
    """Result of a dedup scan for a single entity."""
    auto_merged: list[MergeAction] = field(default_factory=list)
    candidates: list[DedupCandidate] = field(default_factory=list)


class DedupEngine:
    """Find and classify near-duplicate entities using embedding + name similarity.

    Thresholds:
      vec_auto + name_auto  -> auto-merge (both must pass)
      vec_candidate         -> surface as candidate for manual review
    """

    def __init__(
        self,
        vec_auto: float = 0.98,
        name_auto: float = 0.85,
        vec_candidate: float = 0.90,
    ) -> None:
        self.vec_auto = vec_auto
        self.name_auto = name_auto
        self.vec_candidate = vec_candidate

    # ── cheap pre-filter ───────────────────────────────────────────────────────

    @staticmethod
    def _pre_filter(name: str, candidate: str) -> bool:
        """Quick length-ratio + first-char check before expensive embed call."""
        n, c = name.strip().lower(), candidate.strip().lower()
        if not n or not c:
            return False
        # Length ratio: skip if one is >3x longer than the other
        min_len, max_len = min(len(n), len(c)), max(len(n), len(c))
        if max_len > 0 and min_len / max_len < 0.33:
            return False
        return True

    # ── find_similar ──────────────────────────────────────────────────────────

    def find_similar(
        self,
        name: str,
        embedder: EmbeddingProvider,
        conn: sqlite3.Connection,
    ) -> DedupResult:
        """Find entities similar to `name` among existing kg_nodes.

        Skips the exact-match row (same name, case-insensitive).
        Returns DedupResult with auto_merged and candidates lists.
        """
        result = DedupResult()
        if not name or not name.strip():
            return result

        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM kg_nodes WHERE name != ? COLLATE NOCASE",
            (name,),
        )
        existing_names = [row[0] for row in cur.fetchall()]
        if not existing_names:
            return result

        # Pre-filter before expensive embed
        candidates_to_check = [n for n in existing_names if self._pre_filter(name, n)]
        if not candidates_to_check:
            return result

        query_vec = embedder.embed(name)
        candidate_vecs = embedder.embed_batch(candidates_to_check)

        for cname, cvec in zip(candidates_to_check, candidate_vecs):
            vec_sim = cosine_similarity(query_vec, cvec)

            if vec_sim >= self.vec_auto:
                name_sim = jaro_winkler_similarity(name, cname)
                if name_sim >= self.name_auto:
                    result.auto_merged.append(
                        MergeAction(source_name=name, target_name=cname,
                                    vector_sim=vec_sim, name_sim=name_sim)
                    )
                    continue
                # High vector sim but name doesn't quite match -> candidate
                result.candidates.append(
                    DedupCandidate(source_name=name, target_name=cname,
                                   vector_sim=vec_sim, name_sim=name_sim)
                )
            elif vec_sim >= self.vec_candidate:
                name_sim = jaro_winkler_similarity(name, cname)
                result.candidates.append(
                    DedupCandidate(source_name=name, target_name=cname,
                                   vector_sim=vec_sim, name_sim=name_sim)
                )

        return result

    # ── scan_all ──────────────────────────────────────────────────────────────

    def scan_all(
        self,
        conn: sqlite3.Connection,
        embedder: EmbeddingProvider,
    ) -> list[DedupCandidate]:
        """Pairwise scan of all kg_nodes. Returns pairs above candidate threshold.

        O(N^2) — acceptable for personal KGs (<10k nodes).
        Embeddings are cached in memory during the scan.
        """
        cur = conn.cursor()
        cur.execute("SELECT name FROM kg_nodes ORDER BY mention_count DESC")
        names = [row[0] for row in cur.fetchall()]

        if len(names) < 2:
            return []

        # Embed all names once (cached in dict)
        vecs: dict[str, list[float]] = {}
        for name in names:
            vecs[name] = embedder.embed(name)

        found: list[DedupCandidate] = []
        seen: set[frozenset] = set()

        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                pair = frozenset({name_a.lower(), name_b.lower()})
                if pair in seen:
                    continue
                seen.add(pair)

                if not self._pre_filter(name_a, name_b):
                    continue

                vec_sim = cosine_similarity(vecs[name_a], vecs[name_b])
                if vec_sim >= self.vec_candidate:
                    name_sim = jaro_winkler_similarity(name_a, name_b)
                    found.append(
                        DedupCandidate(source_name=name_a, target_name=name_b,
                                       vector_sim=vec_sim, name_sim=name_sim)
                    )

        return found

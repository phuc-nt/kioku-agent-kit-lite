"""Consolidation helpers — pure orchestration logic for kioku-lite consolidate command.

Extracted from service.py to keep files under 200 lines (dev rule).
Called by KiokuLiteService.consolidate().
"""

from __future__ import annotations

from datetime import date, timedelta


def build_stale_report(
    memories: list[dict],
    cutoff_date: date,
) -> dict:
    """Build stale_memories section of consolidation report."""
    return {
        "older_than": cutoff_date.isoformat(),
        "count": len(memories),
        "memories": [
            {
                "text": m.get("text", ""),
                "date": m.get("date", ""),
                "content_hash": m.get("content_hash", ""),
            }
            for m in memories
        ],
    }


def build_decay_report(decayed: list[dict], half_life_days: int) -> dict:
    """Build decay section of consolidation report."""
    return {
        "half_life_days": half_life_days,
        "edges_decayed": len(decayed),
        "edges": decayed,
    }


def stale_cutoff(older_than_days: int) -> date:
    """Return the cutoff date for stale memories (today - older_than_days)."""
    return date.today() - timedelta(days=older_than_days)

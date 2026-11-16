"""Retrieval / context metrics shared by every experiment (definitions in data/benchmark/README.md).

- An evidence item matches a chunk when source_id and page are equal and squash(evidence) is a
  substring of squash(chunk text) (whitespace-insensitive).
- Single-evidence question: `gold` items are alternatives; any match is a hit.
- Multi-hop question (`evidence_groups`): every group must be matched (any alternative inside a group).
Abstain questions never enter these denominators.
"""
from __future__ import annotations

from typing import Any, Iterable

from .retrieval import squash


def groups(q: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return q["evidence_groups"] if "evidence_groups" in q else [q["gold"]]


def item_in(chunk: dict[str, Any], g: dict[str, Any]) -> bool:
    return chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and squash(g["evidence"]) in squash(chunk["text"])


def groups_covered(q: dict[str, Any], chunks: Iterable[dict[str, Any]]) -> list[bool]:
    chunks = list(chunks)
    return [any(item_in(c, g) for c in chunks for g in grp) for grp in groups(q)]


def full_hit(q: dict[str, Any], chunks: Iterable[dict[str, Any]]) -> bool:
    return all(groups_covered(q, chunks))


def first_hit_rank(q: dict[str, Any], ranked: list[dict[str, Any]]) -> int | None:
    """Rank at which all groups are first covered (prefix of the ranking)."""
    for k in range(1, len(ranked) + 1):
        if full_hit(q, ranked[:k]):
            return k
    return None


def ratio(n: int, d: int) -> dict[str, Any]:
    return {"n": n, "N": d, "rate": round(n / d, 4) if d else None, "text": f"{n}/{d}" if d else "0/0 (n/a)"}

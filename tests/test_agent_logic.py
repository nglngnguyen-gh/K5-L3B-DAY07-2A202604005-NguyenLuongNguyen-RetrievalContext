"""Offline tests of the agent's deterministic parts (citation validation, context packing). No API calls."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import GeminiAgent, cited_ids  # noqa: E402


def row(cid, text, rank=1):
    return {"id": cid, "text": text, "page": 1, "rank": rank, "score": 0.9, "metadata": {"sections": ["IFC"]}}


class CitationTests(unittest.TestCase):
    def test_cited_ids_accepts_grouped_brackets(self):
        self.assertEqual(cited_ids("x [a:1] y [b:2, c:3; d:4]"), {"a:1", "b:2", "c:3", "d:4"})

    def test_finish_validates_against_context(self):
        ctx = {"a": row("a", "t")}
        ok = GeminiAgent._finish(None, {"answer": "Yes [a]", "citations": ["a"], "abstain": False}, ctx, [])
        self.assertEqual(ok["status"], "answered")
        bad = GeminiAgent._finish(None, {"answer": "Yes [z]", "citations": ["z"], "abstain": False}, ctx, [])
        self.assertEqual(bad["status"], "invalid_citations")
        nocite = GeminiAgent._finish(None, {"answer": "Yes", "citations": [], "abstain": False}, ctx, [])
        self.assertEqual(nocite["status"], "invalid_citations")
        ab = GeminiAgent._finish(None, {"answer": "Not stated", "citations": [], "abstain": True}, ctx, [])
        self.assertEqual(ab["status"], "abstain")


class PackTests(unittest.TestCase):
    def test_dedup_and_budget(self):
        ctx, budget = {}, [10]
        p, added, withheld = GeminiAgent._pack(None, [row("a", "12345"), row("b", "123456789")], ctx, budget)
        self.assertEqual(added, ["a"])
        self.assertEqual(withheld, ["b"])
        p2, added2, _ = GeminiAgent._pack(None, [row("a", "12345")], ctx, budget)
        self.assertEqual(added2, [])
        self.assertEqual(p2[0]["text"], "(already shown above)")


if __name__ == "__main__":
    unittest.main()

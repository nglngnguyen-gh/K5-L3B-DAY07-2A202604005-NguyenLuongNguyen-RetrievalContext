"""Offline tests for src.retrieval / src.metrics (no DB, no API: dense is replaced by a stub list)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import first_hit_rank, full_hit, groups_covered  # noqa: E402
from src.retrieval import BM25Retriever, HybridRRFRetriever, RetrievalError, tokenize  # noqa: E402


def chunk(cid, page, text, sections=("IFC",)):
    return {"id": cid, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page,
            "metadata": {"sections": list(sections), "printed_pages": [], "heading": None, "chunk_index": 0}}


CHUNKS = [
    chunk("c1", 1, "IBRD Commitments 50,177 40,885 in millions of dollars", ("World Bank (IBRD/IDA)",)),
    chunk("c2", 2, "IFC allocable income 1,743 for fiscal 2026"),
    chunk("c3", 3, "The European Investment Bank pledged over\n€1 billion to support Mission 300", ("One World Bank Group",)),
]


class StubDense:
    def __init__(self, order):
        self.order = order

    def search(self, query, top_k=5, filters=None):
        by = {c["id"]: c for c in CHUNKS}
        rows = [by[i] for i in self.order if not filters or filters.get("section") in by[i]["metadata"]["sections"]]
        return [dict(by[c["id"]], score=1 - 0.1 * r, rank=r + 1, method="dense") for r, c in enumerate(rows[:top_k])]


class TokenizeTests(unittest.TestCase):
    def test_numbers_keep_and_strip_separators(self):
        toks = tokenize("IBRD Commitments 50,177 and $13.9 billion")
        self.assertIn("50,177", toks)
        self.assertIn("50177", toks)
        self.assertIn("13.9", toks)
        self.assertNotIn("and", toks)


class BM25Tests(unittest.TestCase):
    def test_ranks_exact_number_first_and_filters(self):
        bm = BM25Retriever(CHUNKS)
        self.assertEqual(bm.search("IBRD commitments 50,177")[0]["id"], "c1")
        self.assertEqual([r["id"] for r in bm.search("allocable income", filters={"section": "World Bank (IBRD/IDA)"})], [])
        with self.assertRaises(RetrievalError):
            bm.search("x", filters={"year": 2026})


class HybridTests(unittest.TestCase):
    def test_rrf_uses_ranks_of_both_lists(self):
        hy = HybridRRFRetriever(StubDense(["c3", "c2", "c1"]), BM25Retriever(CHUNKS), k=60, candidates=3)
        res = hy.search("IBRD commitments 50,177", top_k=3)
        top = {r["id"]: r for r in res}
        self.assertEqual(top["c1"]["component_ranks"], {"dense": 3, "bm25": 1})
        self.assertAlmostEqual(top["c1"]["score"], round(1 / 63 + 1 / 61, 6))
        self.assertEqual([r["rank"] for r in res], [1, 2, 3])


class MetricTests(unittest.TestCase):
    def test_whitespace_insensitive_and_multi_hop(self):
        q1 = {"gold": [{"source_id": "doc", "page": 3, "evidence": "pledged over €1 billion"}]}
        self.assertTrue(full_hit(q1, [CHUNKS[2]]))
        self.assertFalse(full_hit(q1, [dict(CHUNKS[2], page=4)]))
        qm = {"evidence_groups": [[{"source_id": "doc", "page": 1, "evidence": "50,177"}],
                                  [{"source_id": "doc", "page": 3, "evidence": "Mission 300"}]]}
        self.assertEqual(groups_covered(qm, [CHUNKS[0]]), [True, False])
        self.assertEqual(first_hit_rank(qm, [CHUNKS[0], CHUNKS[1], CHUNKS[2]]), 3)


if __name__ == "__main__":
    unittest.main()

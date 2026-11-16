"""Retrieval methods for Day 7 on the Day 6 corpus.

Methods (all return the same result dict, ranked, with score/rank/method and full source metadata):
- dense  : Gemini `gemini-embedding-001` query embedding (RETRIEVAL_QUERY, 768d, L2-normalized, identical to Day 6)
           + exact cosine search in the Day 6 PostgreSQL/pgvector store; section filter applied in SQL before ranking.
- bm25   : Okapi BM25 over the same chunks (in memory), number-aware tokenizer; section filter applied before ranking.
- hybrid : Reciprocal Rank Fusion of dense and bm25 candidate lists (rank-based, so no score normalization needed).

Errors (API, DB, bad filter) raise RetrievalError; there is no fallback to fake vectors or fake results.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
DAY6_DATA = ROOT / "starter" / "data" / "day6"
SOURCE_ID = "wbg-ar-2026"
DAY6_STRATEGY_FOR_INDEX = {"A": "A", "B": "B"}  # Day 6 C == B vectors + section filter
ALLOWED_FILTERS = {"section", "page"}


class RetrievalError(RuntimeError):
    """Raised when a retrieval backend fails. Callers log it; they must not substitute results."""


def squash(s: str) -> str:
    return " ".join(s.split())


def load_chunks(index: str) -> list[dict[str, Any]]:
    path = DAY6_DATA / f"chunks_{index}.json"
    if not path.exists():
        raise RetrievalError(f"{path} missing: run `python -m src.export_day6 --day6 <Day 6 repo>` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = dict(filters or {})
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise RetrievalError(f"unsupported filter keys: {sorted(bad)} (allowed: {sorted(ALLOWED_FILTERS)})")
    if "page" in filters:
        filters["page"] = int(filters["page"])
    return filters


def _passes(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    if "section" in filters and filters["section"] not in chunk["metadata"].get("sections", []):
        return False
    if "page" in filters and chunk["page"] != filters["page"]:
        return False
    return True


def _result(chunk: dict[str, Any], score: float, rank: int, method: str, **extra: Any) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "score": round(float(score), 6), "rank": rank, "method": method,
            "metadata": {k: chunk["metadata"].get(k) for k in ("sections", "printed_pages", "heading", "chunk_index")},
            **extra}


# --------------------------------------------------------------------------- embeddings
class QueryEmbedder(Protocol):
    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]: ...


class GeminiQueryEmbedder:
    """Same model/task/dimension/normalization as Day 6 `GeminiEmbedder.embed_query`.

    Query vectors are cached in the Day 6 `embedding_cache` table (keyed by model, dims, task, text hash),
    so a repeated query costs no API call. Usage counters feed the cost report.
    """

    task_type = "RETRIEVAL_QUERY"

    def __init__(self, conn, model_name: str = "gemini-embedding-001", dimensions: int = 768, max_retries: int = 5):
        from google import genai  # imported lazily so BM25-only runs need no API package/key

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RetrievalError("GEMINI_API_KEY is not set (.env)")
        self.client = genai.Client(api_key=key)
        self.conn = conn
        self.model_name, self.dimensions, self.max_retries = model_name, dimensions, max_retries
        self.api_calls = 0
        self.cache_hits = 0
        self.chars_embedded = 0

    def _cached(self, h: str) -> list[float] | None:
        row = self.conn.execute(
            "SELECT embedding::text FROM embedding_cache WHERE model=%s AND dims=%s AND task_type=%s AND text_hash=%s",
            (self.model_name, self.dimensions, self.task_type, h)).fetchone()
        return json.loads(row[0]) if row else None

    def embed_query(self, text: str) -> list[float]:
        from google.genai import errors, types

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._cached(h)
        if cached is not None:
            self.cache_hits += 1
            return cached
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.embed_content(
                    model=self.model_name, contents=[text],
                    config=types.EmbedContentConfig(task_type=self.task_type, output_dimensionality=self.dimensions))
                break
            except errors.APIError as exc:
                retriable = getattr(exc, "code", None) in (429, 500, 503)
                if not retriable or attempt == self.max_retries - 1:
                    raise RetrievalError(f"Gemini query embedding failed: {exc}") from exc
                time.sleep(min(30, 2 ** attempt * 2))
        self.api_calls += 1
        self.chars_embedded += len(text)
        v = list(resp.embeddings[0].values)
        if len(v) != self.dimensions:
            raise RetrievalError(f"embedding has {len(v)} dims, expected {self.dimensions}")
        n = math.sqrt(sum(x * x for x in v))
        if n == 0:
            raise RetrievalError("zero-norm query embedding")
        v = [x / n for x in v]
        self.conn.execute(
            "INSERT INTO embedding_cache (model, dims, task_type, text_hash, embedding) VALUES (%s,%s,%s,%s,%s::vector) "
            "ON CONFLICT DO NOTHING", (self.model_name, self.dimensions, self.task_type, h, _vec(v)))
        return v


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


# --------------------------------------------------------------------------- dense
class DenseRetriever:
    """Exact cosine search over the Day 6 pgvector table `chunks` (no ANN index), like Day 6."""

    method = "dense"

    def __init__(self, conn, embedder: QueryEmbedder, index: str = "B"):
        if index not in DAY6_STRATEGY_FOR_INDEX:
            raise RetrievalError(f"unknown index {index}")
        self.conn, self.embedder, self.index = conn, embedder, index
        self.strategy = DAY6_STRATEGY_FOR_INDEX[index]
        self.by_id = {c["id"]: c for c in load_chunks(index)}
        dims = conn.execute("SELECT DISTINCT vector_dims(embedding), embedding_model FROM chunks WHERE strategy=%s",
                            (self.strategy,)).fetchall()
        if dims != [(embedder.dimensions, embedder.model_name)]:
            raise RetrievalError(f"index {index} vectors {dims} incompatible with query embedder "
                                 f"({embedder.model_name}, {embedder.dimensions})")

    def search(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        q = _vec(self.embedder.embed_query(query))
        where, params = ["strategy = %s"], [self.strategy]
        if "section" in filters:
            where.append("%s = ANY(sections)")
            params.append(filters["section"])
        if "page" in filters:
            where.append("page = %s")
            params.append(filters["page"])
        sql = (f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score FROM chunks WHERE {' AND '.join(where)} "
               f"ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        try:
            rows = self.conn.execute(sql, [q, *params, q, top_k]).fetchall()
        except Exception as exc:  # psycopg errors
            raise RetrievalError(f"pgvector search failed: {exc}") from exc
        out = []
        for rank, (cid, score) in enumerate(rows, 1):
            if cid not in self.by_id:
                raise RetrievalError(f"chunk {cid} in store but not in export; re-run src.export_day6")
            out.append(_result(self.by_id[cid], score, rank, self.method))
        return out


# --------------------------------------------------------------------------- BM25
_STOP = set("""a an and are as at be by did do does for from had has have how in into is it its of on or s
that the their them this those to was were what when where which who whom why will with""".split())
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase NFKC tokens. Numbers keep inner separators ('50,177', '13.9') and also emit a
    separator-free form ('50177'), so '50,177' matches '50177'. Hyphenated words emit parts too."""
    text = unicodedata.normalize("NFKC", text).casefold()
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok[0].isdigit():
            out.append(tok)
            bare = re.sub(r"[.,]", "", tok)
            if bare != tok and "," in tok:
                out.append(bare)
        elif tok not in _STOP:
            out.append(tok)
            if "-" in tok:
                out.extend(p for p in tok.split("-") if p and p not in _STOP)
    return out


class BM25Retriever:
    method = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = _check_filters(filters)
        q = [t for t in dict.fromkeys(tokenize(query)) if t in self.idf]
        scored = []
        for i, c in enumerate(self.chunks):
            if not _passes(c, filters):
                continue
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((s, c["id"], c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_result(c, s, r, self.method) for r, (s, _, c) in enumerate(scored[:top_k], 1)]


# --------------------------------------------------------------------------- hybrid
class HybridRRFRetriever:
    """score(d) = sum_m 1 / (k + rank_m(d)) over dense and bm25 lists of `candidates` items each.
    RRF uses ranks only, so cosine and BM25 scores are never added directly."""

    method = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60, candidates: int = 20,
                 weights: tuple[float, float] = (1.0, 1.0)):
        self.dense, self.bm25, self.k, self.candidates, self.weights = dense, bm25, k, candidates, weights

    def search(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        lists = {"dense": self.dense.search(query, self.candidates, filters),
                 "bm25": self.bm25.search(query, self.candidates, filters)}
        fused: dict[str, dict[str, Any]] = {}
        for (name, results), w in zip(lists.items(), self.weights):
            for r in results:
                e = fused.setdefault(r["id"], {"chunk": r, "score": 0.0, "ranks": {}})
                e["score"] += w / (self.k + r["rank"])
                e["ranks"][name] = r["rank"]
        order = sorted(fused.values(), key=lambda e: (-e["score"], e["chunk"]["id"]))[:top_k]
        out = []
        for rank, e in enumerate(order, 1):
            r = dict(e["chunk"])
            r.update(score=round(e["score"], 6), rank=rank, method=self.method,
                     component_ranks={"dense": e["ranks"].get("dense"), "bm25": e["ranks"].get("bm25")})
            out.append(r)
        return out


# --------------------------------------------------------------------------- factory
@dataclass
class RetrievalStack:
    """Opens the DB/embedder once and exposes every method over one index."""

    index: str = "B"
    conn: Any = None
    embedder: GeminiQueryEmbedder | None = None
    methods: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, index: str = "B", rrf_k: int = 60, rrf_candidates: int = 20) -> "RetrievalStack":
        import psycopg
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise RetrievalError("DATABASE_URL is not set (.env)")
        try:
            conn = psycopg.connect(url, autocommit=True)
        except Exception as exc:
            raise RetrievalError(f"cannot connect to Day 6 vector store: {exc}") from exc
        emb = GeminiQueryEmbedder(conn, os.environ.get("EMBEDDING_MODEL") or "gemini-embedding-001",
                                  int(os.environ.get("EMBEDDING_DIMENSIONS") or 768))
        dense = DenseRetriever(conn, emb, index)
        bm25 = BM25Retriever(load_chunks(index))
        hybrid = HybridRRFRetriever(dense, bm25, k=rrf_k, candidates=rrf_candidates)
        return cls(index=index, conn=conn, embedder=emb, methods={"dense": dense, "bm25": bm25, "hybrid": hybrid})

    def search(self, method: str, query: str, top_k: int = 5, filters: dict[str, Any] | None = None):
        if method not in self.methods:
            raise RetrievalError(f"unknown method {method}; choose from {sorted(self.methods)}")
        return self.methods[method].search(query, top_k, filters)

    def usage(self) -> dict[str, int]:
        e = self.embedder
        return {"embed_api_calls": e.api_calls, "embed_cache_hits": e.cache_hits, "embed_chars": e.chars_embedded}

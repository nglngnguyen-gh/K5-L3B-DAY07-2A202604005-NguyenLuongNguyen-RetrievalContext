"""Export Day 6 chunks (PostgreSQL + pgvector) and page texts to the Day 7 JSON contract.

Run from the repo root:
    .venv/Scripts/python.exe -m src.export_day6 --day6 "<path to Day 6 repo>"

Writes (git-ignored, the source PDF is CC BY-NC-ND so extracted text is not redistributed):
    starter/data/day6/chunks_<strategy>.json   one JSON array per Day 6 strategy (A, B)
    starter/data/day6/pages.json               normalized page texts from Day 6 ingestion
    starter/data/day6/export_meta.json         counts, embedding model, dimensions, content hash
The export never touches embeddings: dense search keeps querying the Day 6 vector store.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "starter" / "data" / "day6"
STRATEGIES = ("A", "B")  # C reuses B's chunks/vectors and only adds a section filter at query time


def export_chunks(conn: psycopg.Connection, strategy: str) -> list[dict]:
    rows = conn.execute(
        """SELECT chunk_id, document_id, source, page, printed_pages, sections, heading, chunk_index,
                  text, content_hash, embedding_model, vector_dims(embedding) AS dims
           FROM chunks WHERE strategy = %s ORDER BY page, chunk_index, chunk_id""",
        (strategy,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"no chunks for strategy {strategy} in the Day 6 store")
    out = []
    for (cid, doc, source, page, printed, sections, heading, idx, text, h, model, dims) in rows:
        out.append({
            "id": cid,
            "text": text,
            "source_id": doc,
            "source": source,
            "page": page,
            "metadata": {
                "strategy": strategy,
                "printed_pages": list(printed or []),
                "sections": list(sections or []),
                "heading": heading,
                "chunk_index": idx,
                "content_hash": h,
                "embedding_model": model,
                "embedding_dims": dims,
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day6", required=True, type=Path, help="Day 6 repo root")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (.env)")

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategies": {}}
    with psycopg.connect(url) as conn:
        meta["pgvector_version"] = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        for s in STRATEGIES:
            chunks = export_chunks(conn, s)
            ids = [c["id"] for c in chunks]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"duplicate chunk ids in strategy {s}")
            blob = json.dumps(chunks, ensure_ascii=False, indent=1)
            (OUT / f"chunks_{s}.json").write_text(blob + "\n", encoding="utf-8")
            meta["strategies"][s] = {
                "chunks": len(chunks),
                "pages": len({c["page"] for c in chunks}),
                "embedding_models": sorted({c["metadata"]["embedding_model"] for c in chunks}),
                "embedding_dims": sorted({c["metadata"]["embedding_dims"] for c in chunks}),
                "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            }

    pages_src = args.day6 / "starter" / "data" / "processed" / "pages.jsonl"
    pages = [json.loads(line) for line in pages_src.read_text(encoding="utf-8").splitlines() if line.strip()]
    (OUT / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    meta["pages"] = {"total": len(pages), "counted": sum(1 for p in pages if p["metadata"].get("counted"))}
    (OUT / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

"""Retrieval-only benchmark: every method x index x filter setting on one split.

    .venv/Scripts/python.exe -m src.eval_retrieval --split dev
    .venv/Scripts/python.exe -m src.eval_retrieval --split dev --methods dense bm25 --indexes B --filters on

Only the question text (and, when `--filters on`, the user-provided `filter`) is sent to the retriever;
gold/expected answers are used only afterwards to score. `--split test` is refused until
starter/results/selection.json (the locked configuration) exists.

Outputs (starter/results/retrieval/<split>/): summary.json, traces.jsonl (top-k ids, pages, scores,
component ranks, hit flags, 120-char previews), and a printed table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .metrics import first_hit_rank, full_hit, groups, groups_covered, ratio
from .retrieval import ROOT, RetrievalError, RetrievalStack

BENCH = ROOT / "data" / "benchmark" / "questions.json"
LOCK = ROOT / "starter" / "results" / "selection.json"
OUT = ROOT / "starter" / "results" / "retrieval"
TRACE_K = 10


def run(split: str, methods: list[str], indexes: list[str], filter_modes: list[str]) -> dict:
    qs = [q for q in json.loads(BENCH.read_text(encoding="utf-8")) if q["split"] == split]
    out_dir = OUT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, traces = {"split": split, "n_answerable": len(qs), "configs": {}}, []
    for index in indexes:
        stack = RetrievalStack.open(index)
        for method in methods:
            for fm in filter_modes:
                name = f"{method}-{index}" + ("+filter" if fm == "on" else "")
                lat, rows, errors = [], [], []
                before = stack.usage()
                for q in qs:
                    filt = q.get("filter") if fm == "on" else None
                    t0 = time.perf_counter()
                    try:
                        res = stack.search(method, q["question"], TRACE_K, filt)
                    except RetrievalError as exc:
                        errors.append({"id": q["id"], "error": str(exc)})
                        traces.append({"config": name, "id": q["id"], "error": str(exc)})
                        continue
                    lat.append((time.perf_counter() - t0) * 1000)
                    rank = first_hit_rank(q, res)
                    rows.append({"q": q, "res": res, "rank": rank})
                    traces.append({
                        "config": name, "id": q["id"], "type": q["type"], "query": q["question"], "filter": filt,
                        "first_full_hit_rank": rank,
                        "groups_covered_top3": groups_covered(q, res[:3]),
                        "top": [{"rank": r["rank"], "id": r["id"], "page": r["page"], "score": r["score"],
                                 "component_ranks": r.get("component_ranks"), "sections": r["metadata"]["sections"],
                                 "preview": " ".join(r["text"].split())[:120]} for r in res],
                    })
                single = [r for r in rows if len(groups(r["q"])) == 1]
                multi = [r for r in rows if len(groups(r["q"])) > 1]
                n_ans = len(qs)
                used = stack.usage()
                summary["configs"][name] = {
                    "method": method, "index": index, "filter": fm,
                    "hit_at_1_single": ratio(sum(full_hit(r["q"], r["res"][:1]) for r in single), len(single)),
                    "evidence_hit_at_3": ratio(sum(full_hit(r["q"], r["res"][:3]) for r in rows), n_ans),
                    "evidence_hit_at_5": ratio(sum(full_hit(r["q"], r["res"][:5]) for r in rows), n_ans),
                    "evidence_hit_at_10": ratio(sum(full_hit(r["q"], r["res"][:10]) for r in rows), n_ans),
                    "mrr_at_10": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n_ans, 4) if n_ans else None,
                    "multi_hop_groups_top3": ratio(sum(sum(groups_covered(r["q"], r["res"][:3])) for r in multi),
                                                   sum(len(groups(r["q"])) for r in multi)),
                    "multi_hop_full_top10": ratio(sum(full_hit(r["q"], r["res"][:10]) for r in multi), len(multi)),
                    "misses_at_3": [r["q"]["id"] for r in rows if not full_hit(r["q"], r["res"][:3])],
                    "errors": errors,
                    "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
                    "latency_ms_p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 1) if lat else None,
                    "embed_api_calls": used["embed_api_calls"] - before["embed_api_calls"],
                    "embed_cache_hits": used["embed_cache_hits"] - before["embed_cache_hits"],
                    "embed_chars": used["embed_chars"] - before["embed_chars"],
                }
        stack.conn.close()
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (out_dir / "traces.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return summary


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    ap.add_argument("--indexes", nargs="+", default=["A", "B"])
    ap.add_argument("--filters", nargs="+", choices=["off", "on"], default=["off", "on"])
    args = ap.parse_args()
    if args.split == "test" and not LOCK.exists():
        print(f"refusing to run test: lock the configuration first ({LOCK.relative_to(ROOT)})")
        return 2
    s = run(args.split, args.methods, args.indexes, args.filters)
    hdr = f"{'config':<18}{'Hit@1(single)':>15}{'EvHit@3':>10}{'EvHit@5':>10}{'EvHit@10':>10}{'MRR':>7}{'MH grp@3':>10}{'ms':>7}{'API':>5}  misses@3"
    print(hdr)
    for name, c in s["configs"].items():
        print(f"{name:<18}{c['hit_at_1_single']['text']:>15}{c['evidence_hit_at_3']['text']:>10}"
              f"{c['evidence_hit_at_5']['text']:>10}{c['evidence_hit_at_10']['text']:>10}{c['mrr_at_10']:>7}"
              f"{c['multi_hop_groups_top3']['text']:>10}{c['latency_ms_median']:>7}{c['embed_api_calls']:>5}  "
              f"{','.join(c['misses_at_3'])}{'  ERR' + str(len(c['errors'])) if c['errors'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

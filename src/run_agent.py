"""Run the Gemini agent with one config on one split and save full traces.

    .venv/Scripts/python.exe -m src.run_agent --config configs/dense-B-k3.json --split dev
    .venv/Scripts/python.exe -m src.run_agent --config configs/dense-B-k3.json --split dev --ids d01 d21 u-d02

Only `question` and the user-provided `filter` are passed to the agent. Gold evidence is loaded *after* each
run, only to compute automatic context metrics. `--split test` is refused until starter/results/selection.json exists.

Outputs in starter/results/agent/<config name>/<split>/:
  runs.jsonl    full record per question (events, tool calls, context texts, answer, citations, usage, cost, errors)
  traces.json   starter/agent_metrics.py format (query_id, status, events with event="search", citations)
  auto.json     automatic metrics: tool use, searches, Context Evidence Rate, abstain counts, latency, tokens, cost
Manual labels (answer grade, citation support, abstention) are added later in reviews.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path

from .agent import AgentConfig, GeminiAgent
from .metrics import full_hit, groups, groups_covered, ratio
from .retrieval import ROOT, RetrievalStack

BENCH = ROOT / "data" / "benchmark"
LOCK = ROOT / "starter" / "results" / "selection.json"
OUT = ROOT / "starter" / "results" / "agent"


def load_cases(split: str, ids: list[str] | None) -> list[dict]:
    qs = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))
    ab = json.loads((BENCH / "abstain.json").read_text(encoding="utf-8"))
    cases = [{**q, "answerable": True} for q in qs] + [{**a, "answerable": False} for a in ab]
    cases = [c for c in cases if c["split"] == split and (not ids or c["id"] in ids)]
    if ids and len(cases) != len(set(ids)):
        raise SystemExit(f"unknown ids for split {split}: {sorted(set(ids) - {c['id'] for c in cases})}")
    return cases


def agent_input(case: dict) -> tuple[str, dict | None]:
    """The ONLY fields the agent sees."""
    return case["question"], case.get("filter")


def auto_metrics(cases: list[dict], runs: dict[str, dict]) -> dict:
    ok = [c for c in cases if runs[c["id"]]["status"] != "error"]
    ans = [c for c in ok if c["answerable"]]
    single = [c for c in ans if len(groups(c)) == 1]
    multi = [c for c in ans if len(groups(c)) > 1]
    una = [c for c in ok if not c["answerable"]]
    ctx = lambda c: runs[c["id"]]["context"]  # noqa: E731
    first_search = lambda c: [r for r in runs[c["id"]]["first_search_results"]]  # noqa: E731
    lat = [runs[c["id"]]["latency_s"] for c in ok]
    searches = Counter(runs[c["id"]]["n_searches"] for c in ok)
    tot = lambda k: sum(runs[c["id"]]["usage"][k] for c in ok)  # noqa: E731
    return {
        "cases": len(cases), "errors": [c["id"] for c in cases if runs[c["id"]]["status"] == "error"],
        "tool_use": ratio(sum(runs[c["id"]]["n_searches"] > 0 for c in ok), len(ok)),
        "searches_distribution": dict(sorted(searches.items())),
        "searches_mean": round(statistics.mean(runs[c["id"]]["n_searches"] for c in ok), 2) if ok else None,
        "first_search_hit_at_1_single": ratio(sum(full_hit(c, first_search(c)[:1]) for c in single), len(single)),
        "first_search_evidence_hit_at_3": ratio(sum(full_hit(c, first_search(c)[:3]) for c in ans), len(ans)),
        "context_evidence_rate": ratio(sum(full_hit(c, ctx(c)) for c in ans), len(ans)),
        "context_evidence_multi_hop_groups": ratio(sum(sum(groups_covered(c, ctx(c))) for c in multi),
                                                   sum(len(groups(c)) for c in multi)),
        "context_misses": [c["id"] for c in ans if not full_hit(c, ctx(c))],
        "answered_answerable": ratio(sum(runs[c["id"]]["status"] == "answered" for c in ans), len(ans)),
        "false_abstention": ratio(sum(runs[c["id"]]["status"] == "abstain" for c in ans), len(ans)),
        "abstained_unanswerable": ratio(sum(runs[c["id"]]["status"] == "abstain" for c in una), len(una)),
        "citations_in_context": ratio(sum(all(x in runs[c["id"]]["context_ids"] for x in runs[c["id"]]["citations"]) for c in ok), len(ok)),
        "context_chars_mean": round(statistics.mean(runs[c["id"]]["context_chars"] for c in ok)) if ok else None,
        "latency_s_median": round(statistics.median(lat), 2) if lat else None,
        "latency_s_total": round(sum(lat), 1),
        "model_calls": tot("model_calls"), "prompt_tokens": tot("prompt_tokens"),
        "output_tokens": tot("output_tokens"), "thinking_tokens": tot("thinking_tokens"),
        "cost_usd_total": round(sum(runs[c["id"]]["cost_usd"] for c in ok), 5),
        "note": "Automatic metrics only. Answer correctness, citation support and abstention correctness need manual labels.",
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. smoke)")
    args = ap.parse_args()
    if args.split == "test" and not LOCK.exists():
        print(f"refusing to run test: lock the configuration first ({LOCK.relative_to(ROOT)})")
        return 2
    cfg = AgentConfig.load(args.config)
    cases = load_cases(args.split, args.ids)
    stack = RetrievalStack.open(cfg.index)
    agent = GeminiAgent(cfg, stack)
    out_dir = OUT / cfg.name / (args.split + (f"-{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    with (out_dir / "runs.jsonl").open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            question, filt = agent_input(case)
            try:
                r = agent.run(question, filt)
            except Exception as exc:  # model/DB failure: record, never substitute
                r = {"status": "error", "answer": None, "citations": [], "events": [], "context": [], "context_ids": [],
                     "context_chars": 0, "n_searches": 0, "usage": {"model_calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                     "thinking_tokens": 0}, "cost_usd": 0.0, "latency_s": 0.0, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()[-1500:]}
            first = next((e for e in r["events"] if e["event"] == "search"), None)
            r["first_search_results"] = [stack_chunk(agent, x["id"]) for x in first["results"]] if first else []
            record = {"query_id": case["id"], "split": case["split"], "answerable": case["answerable"],
                      "question": question, "filter_given": filt, "config": cfg.__dict__, **r}
            runs[case["id"]] = record
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(cases)}] {case['id']:<6} {r['status']:<9} searches={r['n_searches']} "
                  f"ctx={len(r['context_ids'])} cit={r['citations']} {r['latency_s']}s ${r['cost_usd']}"
                  + (f"  ERROR {r.get('error')}" if r["status"] == "error" else ""))
    traces = [{"query_id": cid, "status": r["status"], "citations": r["citations"],
               "events": [{"event": e["event"], "tool": e.get("tool"), "query": e.get("query")}
                          for e in r["events"] if e["event"] in ("search", "neighbors")]} for cid, r in runs.items()]
    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = {"config": cfg.__dict__, "split": args.split, "ids": args.ids, **auto_metrics(cases, runs)}
    (out_dir / "auto.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["text"] if isinstance(v, dict) and "text" in v else v) for k, v in summary.items()
                      if k not in ("config", "note")}, ensure_ascii=False, indent=1))
    return 0


def stack_chunk(agent: GeminiAgent, cid: str) -> dict:
    return agent.by_id[cid]


if __name__ == "__main__":
    raise SystemExit(main())

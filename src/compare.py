"""Build the comparison table for a split from saved results (no API calls).

    .venv/Scripts/python.exe -m src.compare --split dev --configs dense-B-k3 bm25-B-k3 hybrid-B-k3

Columns
- Retrieval on the ORIGINAL question (src.eval_retrieval, same method/index/filter as the agent config):
  Hit@1 (single-evidence questions), Evidence Hit@3 (all answerable; multi-hop needs every group).
- Agent's first search (query REWRITTEN by the model): Hit@1, Evidence Hit@3.
- Context Evidence Rate: full evidence inside the final context sent to the model (all searches).
- Answer quality from manual labels (labels/<split>/<config>.reviews.json via starter/agent_metrics.py).
- Searches per question, median latency, tokens and cost (USD, list price in the config).
Writes results/<split>_comparison.md and prints it.
"""
from __future__ import annotations

import argparse
import json
import sys

from .retrieval import ROOT

RES = ROOT / "starter" / "results"


def fmt(r: dict) -> str:
    return f"{r['n']}/{r['N']} ({r['rate']:.0%})" if r.get("N") else "0/0 (n/a)"


def frac(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0 (n/a)"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    retr = json.loads((RES / "retrieval" / args.split / "summary.json").read_text(encoding="utf-8"))["configs"]
    rows = []
    for name in args.configs:
        d = RES / "agent" / name / args.split
        auto = json.loads((d / "auto.json").read_text(encoding="utf-8"))
        cfg = auto["config"]
        key = f"{cfg['method']}-{cfg['index']}" + ("+filter" if cfg["use_filter"] else "")
        rq = retr.get(key)
        man_path = d / "agent_metrics.json"
        man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else None
        g = man["grounded_answer"] if man else None
        rows.append({
            "config": name,
            "setup": f"{cfg['method']}, index {cfg['index']}, top-{cfg['top_k']}, filter {'on' if cfg['use_filter'] else 'off'}, "
                     f"budget {cfg['context_budget_chars']} chars, ≤{cfg['max_searches']} searches"
                     + (", neighbors" if cfg.get("neighbors_tool") else "") + (f", thinking {cfg['thinking_budget']}" if cfg.get("thinking_budget") else ""),
            "orig_h1": fmt(rq["hit_at_1_single"]) if rq else "—",
            "orig_h3": fmt(rq["evidence_hit_at_3"]) if rq else "—",
            "rw_h1": fmt(auto["first_search_hit_at_1_single"]),
            "rw_h3": fmt(auto["first_search_evidence_hit_at_3"]),
            "cer": fmt(auto["context_evidence_rate"]),
            "grade": f"{g['0']}/{g['1']}/{g['2']} · {g['mean']:.2f}" if g else "—",
            "cit": frac(man["citation_validity"]["n"], man["citation_validity"]["N"]) if man else "—",
            "abst": frac(man["abstention"]["n"], man["abstention"]["N"]) if man else "—",
            "false_abst": frac(man["false_abstention"]["n"], man["false_abstention"]["N"]) if man else "—",
            "searches": f"{auto['searches_mean']} {auto['searches_distribution']}",
            "lat": auto["latency_s_median"],
            "tok": f"{auto['prompt_tokens'] + auto['output_tokens'] + auto['thinking_tokens']:,}",
            "cost": f"${auto['cost_usd_total']:.4f}",
            "errors": len(auto["errors"]),
        })
    lines = [f"# So sánh cấu hình — split `{args.split}`", "",
             "| Cấu hình | Thiết lập | Hit@1 query gốc | EvHit@3 query gốc | Hit@1 query agent | EvHit@3 query agent | Context Evidence Rate | Grounded 0/1/2 · TB | Citation hợp lệ | Abstention đúng | Từ chối nhầm | Lượt search TB {phân bố} | Trễ trung vị (s) | Token | Chi phí | Lỗi |",
             "|" + "---|" * 16]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in ("config", "setup", "orig_h1", "orig_h3", "rw_h1", "rw_h3", "cer", "grade",
                                                         "cit", "abst", "false_abst", "searches", "lat", "tok", "cost", "errors")) + " |")
    lines += ["", "Hit@1 tính trên câu một evidence; EvHit@3 và Context Evidence Rate tính trên mọi câu có đáp án (multi-hop cần đủ mọi nhóm). "
              "Grounded, citation và abstention lấy từ nhãn thủ công (`labels/`), tổng hợp bằng `starter/agent_metrics.py`. "
              "Chi phí tính theo token × đơn giá trong config (Gemini 2.5 Flash $0.30/1M input, $2.50/1M output), không phải hóa đơn."]
    text = "\n".join(lines) + "\n"
    out = ROOT / (args.out or f"results/{args.split}_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

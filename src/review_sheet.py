"""Print a review sheet (markdown) for manual labelling of one agent run.

    .venv/Scripts/python.exe -m src.review_sheet --run starter/results/agent/dense-B-k3/dev > sheet.md

For each question: question, filter, expected answer, gold evidence (or absence reason), the agent's
answer/status, every search query, and the FULL text of each cited chunk, so the reviewer can check whether
each citation really supports the claim it is attached to. The sheet is a reading aid only; labels are
written by the reviewer in reviews.json (see data/benchmark/README.md for the scales).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import full_hit, groups, groups_covered
from .retrieval import ROOT


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    bench = ROOT / "data" / "benchmark"
    cases = {q["id"]: q for q in json.loads((bench / "questions.json").read_text(encoding="utf-8"))}
    cases.update({a["id"]: a for a in json.loads((bench / "abstain.json").read_text(encoding="utf-8"))})
    for line in (args.run / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if args.ids and r["query_id"] not in args.ids:
            continue
        c = cases[r["query_id"]]
        ctx = {x["id"]: x for x in r["context"]}
        print(f"## {r['query_id']} ({'answerable' if r['answerable'] else 'ABSTAIN expected'}) — status `{r['status']}`"
              f" ({r.get('stop_reason')}) · searches {r['n_searches']} · {r['latency_s']} s · ${r['cost_usd']}")
        print(f"**Q:** {c['question']}" + (f"  \n**Filter:** {r['filter_given']}" if r.get("filter_given") else ""))
        if r["answerable"]:
            print(f"**Expected:** {c['expected_answer']}")
            for gi, grp in enumerate(groups(c)):
                for g in grp:
                    print(f"- gold g{gi} p{g['page']}: \"{g['evidence']}\"")
            print(f"**Context has full evidence:** {full_hit(c, r['context'])} (groups {groups_covered(c, r['context'])})")
        else:
            print(f"**Absence reason:** {c['absence_reason']}")
        print("**Searches:** " + " | ".join(f"`{e['query']}` → p{[x['page'] for x in e['results']]}"
                                           for e in r["events"] if e["event"] == "search"))
        print(f"**Answer:** {r['answer']}")
        if r.get("rejected_answer"):
            print(f"**Rejected answer (invalid citations):** {r['rejected_answer']}")
        if r.get("error"):
            print(f"**ERROR:** {r['error']}")
        for cid in r["citations"]:
            t = ctx.get(cid)
            print(f"> **[{cid}]** p{t['page'] if t else '?'}: {' '.join(t['text'].split()) if t else '(not in context!)'}")
        print()


if __name__ == "__main__":
    main()

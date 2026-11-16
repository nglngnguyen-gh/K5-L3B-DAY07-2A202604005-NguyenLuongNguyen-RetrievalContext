"""Verify the Day 7 benchmark against the Day 6 corpus BEFORE any retrieval/agent run.

Run from the repo root (after src.export_day6):
    .venv/Scripts/python.exe -m src.verify_benchmark

Checks
- split sizes (dev >= 15 answerable + 2 abstain, test >= 5 + 2), unique ids, required fields;
- every gold evidence is on its gold page (whitespace-insensitive) and the page is a counted page;
- whether each evidence fits fully inside at least one chunk of index A and of index B
  (needed for the strict hit definition), and which other pages repeat it;
- filter value is in the gold page's sections;
- dev/test gold-page overlap (reported, not an error);
- abstain questions: every page/snippet that matches their absence_check keywords, for manual review.
Writes data/benchmark/verification.json (with SHA-256 of the benchmark files) and data/benchmark/cases.json.
Never sends questions or gold to a model or retriever.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
DAY6 = ROOT / "starter" / "data" / "day6"


def squash(s: str) -> str:
    """Whitespace-insensitive form used for every evidence match (same rule as Day 6)."""
    return " ".join(s.split())


def evidence_items(q: dict) -> list[tuple[int, dict]]:
    """(group index, gold item). Single-evidence questions are one group of alternatives."""
    if "evidence_groups" in q:
        return [(gi, g) for gi, grp in enumerate(q["evidence_groups"]) for g in grp]
    return [(0, g) for g in q["gold"]]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    qs = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))
    ab = json.loads((BENCH / "abstain.json").read_text(encoding="utf-8"))
    pages = {(p["document_id"], p["page"]): p for p in json.loads((DAY6 / "pages.json").read_text(encoding="utf-8"))}
    idx = {s: json.loads((DAY6 / f"chunks_{s}.json").read_text(encoding="utf-8")) for s in ("A", "B")}
    sq_pages = {k: squash(v["text"]) for k, v in pages.items()}
    sq_chunks = {s: [(c["source_id"], c["page"], c["id"], squash(c["text"])) for c in cs] for s, cs in idx.items()}

    errors: list[str] = []
    warnings: list[str] = []
    ids = [q["id"] for q in qs] + [a["id"] for a in ab]
    for i, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate id {i}")

    report = []
    for q in qs:
        for k in ("id", "split", "type", "question", "expected_answer"):
            if not q.get(k):
                errors.append(f"{q.get('id')}: missing {k}")
        if "gold" not in q and "evidence_groups" not in q:
            errors.append(f"{q['id']}: needs gold or evidence_groups")
        checks = []
        for gi, g in evidence_items(q):
            key = (g["source_id"], g["page"])
            ev = squash(g["evidence"])
            page = pages.get(key)
            if page is None:
                errors.append(f"{q['id']}: page {key} not in corpus")
                continue
            on_page = ev in sq_pages[key]
            counted = bool(page["metadata"].get("counted"))
            also_on = [p for (d, p), t in sq_pages.items() if d == g["source_id"] and p != g["page"] and ev in t]
            in_chunk = {s: [cid for (d, p, cid, t) in rows if (d, p) == key and ev in t] for s, rows in sq_chunks.items()}
            if not on_page:
                errors.append(f"{q['id']}: evidence not found on page {g['page']}: {g['evidence'][:60]}...")
            if not counted:
                errors.append(f"{q['id']}: page {g['page']} is not a counted page")
            for s in ("A", "B"):
                if on_page and not in_chunk[s]:
                    warnings.append(f"{q['id']} (group {gi}, p{g['page']}): evidence split across chunks in index {s}")
            filt = q.get("filter")
            filter_ok = None
            if filt:
                filter_ok = filt["section"] in page["metadata"].get("sections", [])
                if not filter_ok:
                    warnings.append(f"{q['id']}: filter {filt} not in sections of p{g['page']} (alternative evidence)")
            checks.append({"group": gi, "page": g["page"], "evidence_on_page": on_page, "page_counted": counted,
                           "evidence_also_on_pages": also_on, "chunks_containing_A": in_chunk["A"],
                           "chunks_containing_B": in_chunk["B"], "filter_value_on_page": filter_ok})
        if q.get("filter") and not any(c["filter_value_on_page"] for c in checks):
            errors.append(f"{q['id']}: filter value not on any gold page")
        report.append({"id": q["id"], "split": q["split"], "type": q["type"],
                       "n_groups": len(q.get("evidence_groups", [None])), "checks": checks})

    abstain_report = []
    for a in ab:
        for k in ("id", "split", "question", "absence_reason", "absence_check"):
            if not a.get(k):
                errors.append(f"{a.get('id')}: missing {k}")
        if a.get("answerable") is not False or a.get("expected_behavior") != "search_and_abstain":
            errors.append(f"{a['id']}: must be answerable=false / search_and_abstain")
        matches = {}
        for kw in a["absence_check"]:
            pat = re.compile(r"\s+".join(map(re.escape, kw.split())), re.I)
            hits = []
            for (d, p), t in sq_pages.items():
                if not pages[(d, p)]["metadata"].get("counted"):
                    continue
                for m in pat.finditer(t):
                    hits.append({"page": p, "snippet": t[max(0, m.start() - 70): m.end() + 70]})
            matches[kw] = {"pages": sorted({h["page"] for h in hits}), "snippets": hits[:8]}
        abstain_report.append({"id": a["id"], "split": a["split"], "keyword_matches": matches})

    split = Counter(q["split"] for q in qs)
    split_ab = Counter(a["split"] for a in ab)
    if split["dev"] < 15 or split["test"] < 5:
        errors.append(f"answerable split too small: {dict(split)}")
    if split_ab["dev"] < 2 or split_ab["test"] < 2:
        errors.append(f"abstain split too small: {dict(split_ab)}")

    def gold_pages(sp: str) -> set[int]:
        return {g["page"] for q in qs if q["split"] == sp for _, g in evidence_items(q)}
    overlap = sorted(gold_pages("dev") & gold_pages("test"))
    q_text = Counter(q["question"].strip().lower() for q in qs + ab)
    for t, n in q_text.items():
        if n > 1:
            errors.append(f"duplicate question text: {t[:60]}")

    files = {name: hashlib.sha256((BENCH / name).read_bytes()).hexdigest() for name in ("questions.json", "abstain.json")}
    out = {
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": files,
        "counts": {"answerable": dict(split), "abstain": dict(split_ab),
                   "types": {sp: dict(Counter(q["type"] for q in qs if q["split"] == sp)) for sp in ("dev", "test")}},
        "dev_test_gold_page_overlap": overlap,
        "errors": errors,
        "warnings": warnings,
        "questions": report,
        "abstain": abstain_report,
    }
    (BENCH / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases = [{"id": q["id"], "split": q["split"], "answerable": True, "expected_behavior": "search_and_answer"} for q in qs]
    cases += [{"id": a["id"], "split": a["split"], "answerable": False, "expected_behavior": "search_and_abstain"} for a in ab]
    (BENCH / "cases.json").write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")

    print(json.dumps(out["counts"], indent=1))
    print("dev/test gold-page overlap:", overlap)
    print(f"errors ({len(errors)}):", *errors, sep="\n  ")
    print(f"warnings ({len(warnings)}):", *warnings, sep="\n  ")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

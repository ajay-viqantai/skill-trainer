"""Check experience extraction against Qwen's reading of each resume.

    python scripts/08_run_experience.py            (run the parser first)
    python scripts/09_check_experience_llm.py --docs 99

Qwen lists each resume's jobs with start/end dates. The script compares its
total experience and job count with the parser's, and lists the resumes
where they disagree most. Qwen can be wrong too: open the worst ones and
judge which side is right before changing anything.
Safe to stop and re-run: finished resumes are cached.
"""
import argparse
import json
import re
from datetime import date

import requests
from tqdm import tqdm

from _common import append_jsonl, read_jsonl
import config

CACHE = config.EXPERIENCE_DIR / "llm.jsonl"

PROMPT = """Read this resume and list every job in the work experience.
Include full-time jobs, internships and freelance/contract roles.
Do NOT include education, courses, certifications or personal/academic projects.

For each job give:
- title, company (as written; "" if missing)
- start: "YYYY-MM" (use "YYYY-07" if only a year is given)
- end: "YYYY-MM", or "present" if it is the current job (use "YYYY-06" if only a year)
- internship: true or false

Resume:
\"\"\"
{text}
\"\"\"

Return JSON only:
{{"jobs": [{{"title": "", "company": "", "start": "2019-01", "end": "present", "internship": false}}]}}"""


def to_idx(s, today):
    s = str(s).strip().lower()
    if s in {"present", "current", "now", ""}:
        return today.year * 12 + today.month - 1
    m = re.match(r"^((?:19|20)\d{2})-(\d{1,2})$", s)
    if not m or not 1 <= int(m.group(2)) <= 12:
        return None
    return int(m.group(1)) * 12 + int(m.group(2)) - 1

def verified(jobs, text):
    """Drop Qwen jobs whose start year never appears in the resume (invented dates)."""
    keep = []
    for j in jobs:
        year = str(j.get("start", ""))[:4]
        if year.isdigit() and year in text:
            keep.append(j)
    return keep

def total_years(jobs, today):
    iv = []
    for j in jobs:
        a, b = to_idx(j.get("start"), today), to_idx(j.get("end"), today)
        if a is not None and b is not None and b >= a:
            iv.append((a, b))
    merged = []
    for s, e in sorted(iv):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return round(sum(e - s + 1 for s, e in merged) / 12, 1), len(iv)


def ask(text, model):
    resp = requests.post(config.OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(text=text[:7000])}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    resp.raise_for_status()
    jobs = json.loads(resp.json()["message"]["content"]).get("jobs", [])
    return [j for j in jobs if isinstance(j, dict)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=99)
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    args = ap.parse_args()
    today = date.today()

    parsed = {r["doc_id"]: r for r in read_jsonl(config.EXPERIENCE_DIR / "results.jsonl")}
    if not parsed:
        raise SystemExit("Run scripts/08_run_experience.py first.")
    done = {r["doc_id"] for r in read_jsonl(CACHE)}
    ids = sorted(parsed)[: args.docs]
    todo = [d for d in ids if d not in done]
    print(f"{len(ids)} resumes | {len(ids) - len(todo)} already checked | {len(todo)} to ask Qwen")

    for doc_id in tqdm(todo, desc="Qwen reading jobs"):
        text = (config.TEXT_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
        try:
            jobs = ask(text, args.model)
        except Exception as e:
            print(f"\n{doc_id} failed ({e}); it will be retried on the next run.")
            continue
        append_jsonl(CACHE, [{"doc_id": doc_id, "jobs": jobs}])

    llm = {r["doc_id"]: r["jobs"] for r in read_jsonl(CACHE)}
    rows, dropped = [], 0
    for d in ids:
        if d not in llm:
            continue
        text = (config.TEXT_DIR / f"{d}.txt").read_text(encoding="utf-8")
        jobs = verified(llm[d], text)
        dropped += len(llm[d]) - len(jobs)
        llm_total, llm_count = total_years(jobs, today)
        p = parsed[d]
        rows.append((d, p["total_years"], llm_total, p["job_count"], llm_count))

    if not rows:
        return
    n = len(rows)
    diffs = [abs(r[1] - r[2]) for r in rows]
    within_half = sum(x <= 0.5 for x in diffs)
    within_one = sum(x <= 1.0 for x in diffs)
    same_count = sum(r[3] == r[4] for r in rows)
    print(f"\nIgnored {dropped} Qwen jobs whose start year is not in the resume (invented dates).")
    print(f"Compared {n} resumes (parser vs Qwen):")
    print(f"  total experience within 0.5 yr: {within_half}/{n} ({within_half / n:.0%})")
    print(f"  total experience within 1 yr:   {within_one}/{n} ({within_one / n:.0%})")
    print(f"  average difference:             {sum(diffs) / n:.2f} yrs")
    print(f"  same number of jobs:            {same_count}/{n} ({same_count / n:.0%})")
    print("\nBiggest disagreements (check these resumes):")
    for d, pt, lt, pc, lc in sorted(rows, key=lambda r: -abs(r[1] - r[2]))[:12]:
        print(f"  {d}  parser {pt:>4} yrs / {pc} jobs   qwen {lt:>4} yrs / {lc} jobs")


if __name__ == "__main__":
    main()
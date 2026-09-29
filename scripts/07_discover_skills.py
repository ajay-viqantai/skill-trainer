"""Find technologies that are missing from the skills library.

    python scripts/07_discover_skills.py extract --docs 99
      Qwen lists every technology in each resume. Anything the library
      doesn't already know is collected in data/discovery/new_skills.csv,
      most common first.

    Mark rows with: winpty python scripts/review_new_skills.py
    (or edit the 'add' column yourself: 1 = add, 2 = add with exact-case
    matching for names that are also ordinary words, blank/0 = skip).

    python scripts/07_discover_skills.py apply
      Adds the chosen rows to library/skills_library.json (new skills, or
      new aliases when the canonical name already exists) and bumps the
      library version. Then re-run 02_find_mentions.py.

Safe to stop and re-run: finished documents are cached.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict

import requests
from tqdm import tqdm

from _common import append_jsonl, read_jsonl
import config
from skillx import CATEGORIES, SkillLibrary

RAW_FILE = config.DISCOVERY_DIR / "raw.jsonl"
CSV_FILE = config.DISCOVERY_DIR / "new_skills.csv"
CHUNK_CHARS = 3500

PROMPT = """List every technology named in this resume text: programming languages,
frameworks, libraries, databases, cloud services, DevOps tools, data/BI tools,
testing tools and developer software.

Rules:
- Copy each name exactly as written in the text.
- Only concrete named technologies. Skip general concepts (REST APIs, microservices,
  OOP, CI/CD, agile), job titles, companies, universities, and spoken languages.
- category must be one of: {categories}

Text:
\"\"\"
{text}
\"\"\"

Return JSON only:
{{"items": [{{"name": "NestJS", "category": "framework"}}]}}"""


def chunks(text):
    lines, buf = text.split("\n"), ""
    for ln in lines:
        if len(buf) + len(ln) > CHUNK_CHARS and buf:
            yield buf
            buf = ""
        buf += ln + "\n"
    if buf.strip():
        yield buf


def ask(text, model):
    resp = requests.post(config.OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            categories=", ".join(CATEGORIES), text=text)}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    resp.raise_for_status()
    items = json.loads(resp.json()["message"]["content"]).get("items", [])
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = re.sub(r"\s+", " ", str(it.get("name", ""))).strip(" .,;:-•")
        cat = it.get("category") if it.get("category") in CATEGORIES else "other_tool"
        if 1 < len(name) <= 40 and name.lower() in text.lower():  # must really be in the text
            out.append({"name": name, "category": cat})
    return out


def known(lib, name):
    """True if the library already matches this whole name."""
    return any(h[0] == 0 and h[1] == len(name) for h in lib.find(name))


def extract(args):
    lib = SkillLibrary(config.LIBRARY_FILE)
    config.DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    done = {r["doc_id"] for r in read_jsonl(RAW_FILE)}
    files = sorted(config.TEXT_DIR.glob("*.txt"))[: args.docs]
    todo = [f for f in files if f.stem not in done]
    print(f"{len(files)} docs selected | {len(files) - len(todo)} already done | {len(todo)} to process")

    for path in tqdm(todo, desc="Discovering"):
        text = path.read_text(encoding="utf-8")
        items = []
        try:
            for part in chunks(text):
                items += ask(part, args.model)
        except Exception as e:
            print(f"\n{path.stem} failed ({e}); it will be retried on the next run.")
            continue
        append_jsonl(RAW_FILE, [{"doc_id": path.stem, "items": items}])

    # Aggregate across documents
    docs_per_key, variants, cats = Counter(), defaultdict(Counter), defaultdict(Counter)
    for r in read_jsonl(RAW_FILE):
        seen = set()
        for it in r["items"]:
            name = it["name"]
            if known(lib, name):
                continue
            key = re.sub(r"[\s.\-_]", "", name.lower())
            variants[key][name] += 1
            cats[key][it["category"]] += 1
            if key not in seen:
                docs_per_key[key] += 1
                seen.add(key)

    rows = []
    for key, n in docs_per_key.most_common():
        if n < args.min_docs:
            continue
        top = variants[key].most_common()
        rows.append({
            "add": "",
            "canonical_name": top[0][0],
            "category": cats[key].most_common(1)[0][0],
            "doc_count": n,
            "variants": " | ".join(v for v, _ in top),
        })

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["add", "canonical_name", "category", "doc_count", "variants"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} unknown technologies found in {args.min_docs}+ docs -> {CSV_FILE}")
    for r in rows[:25]:
        print(f"  {r['doc_count']:>4}  {r['canonical_name']:<22} {r['category']}")


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower().replace("+", "plus").replace("#", "sharp")).strip("_")
    return s or "skill"


def apply(_args):
    if not CSV_FILE.exists():
        raise SystemExit("Run 'extract' first.")
    with open(config.LIBRARY_FILE, encoding="utf-8") as f:
        data = json.load(f)
    by_name = {s["name"].lower(): s for s in data["skills"]}
    ids = {s["id"] for s in data["skills"]}
    added = aliased = 0

    with open(CSV_FILE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mark = row["add"].strip().lower()
            if mark not in {"1", "2", "y", "yes"}:
                continue
            exact_case = mark == "2"  # also an ordinary word: match exact case only
            name = row["canonical_name"].strip()
            cat = row["category"].strip()
            if cat not in CATEGORIES:
                print(f"Skipping {name}: unknown category '{cat}'")
                continue
            forms = [v.strip() for v in row["variants"].split("|") if v.strip()]
            existing = by_name.get(name.lower())
            if existing:  # canonical already exists: just add new spellings
                for v in forms:
                    if v.lower() != existing["name"].lower() and v not in existing["aliases"]:
                        existing["aliases"].append(v)
                        aliased += 1
                continue
            sid = slug(name)
            while sid in ids:
                sid += "_2"
            ids.add(sid)
            skill = {"id": sid, "name": name, "category": cat,
                     "aliases": sorted({v for v in forms if v.lower() != name.lower()}),
                     "ambiguous": exact_case, "case_sensitive": exact_case}
            data["skills"].append(skill)
            by_name[name.lower()] = skill
            added += 1

    if not added and not aliased:
        raise SystemExit("Nothing marked to add (the 'add' column has no 1 or 2). "
                         "Run: winpty python scripts/review_new_skills.py")

    old = str(data.get("version", "0"))
    data["version"] = str(int(old) + 1) if old.isdigit() else old + "+1"
    with open(config.LIBRARY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    SkillLibrary(config.LIBRARY_FILE)  # validates the result
    print(f"Added {added} skills and {aliased} aliases. Library v{old} -> v{data['version']} "
          f"({len(data['skills'])} skills).")
    print("Next: python scripts/02_find_mentions.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["extract", "apply"])
    ap.add_argument("--docs", type=int, default=99)
    ap.add_argument("--min-docs", type=int, default=2, help="ignore names seen in fewer docs")
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    args = ap.parse_args()
    extract(args) if args.action == "extract" else apply(args)


if __name__ == "__main__":
    main()
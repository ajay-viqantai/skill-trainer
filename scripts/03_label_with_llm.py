"""Step 3: auto-label mentions with a self-hosted LLM (Ollama + Qwen).

    ollama pull qwen2.5:7b-instruct
    python scripts/03_label_with_llm.py --docs 1500

For each mention the LLM answers: is this term used as the technical skill
(1) or as an ordinary word / something else (0)? Safe to stop and re-run:
already-labeled mentions are skipped.
"""
import argparse
import json
import random
from collections import defaultdict

import requests
from tqdm import tqdm

from _common import append_jsonl, read_jsonl
import config

PROMPT = """You are checking technology mentions found in resumes and job descriptions.
In each item, the term inside [[ ]] matched a known technology.

Answer true when the term refers to that technology. This is the usual case, including:
- items in skills lists, tech stacks, tool lists, or comma/pipe/slash-separated lists
- phrases like "using X", "built with X", "experience in X", "deployed to X", "X scripts", "X dashboards", "the X upgrade"
- lowercase or unusual spellings: reactjs, nextjs, mongo, golang, Git-hub
- the technology inside a longer name: "Angular Material", "Asynchronous Django server", "NGX Bootstrap"

Answer false only when the term is clearly NOT that technology:
- an ordinary English word: "ready to go live", "express ideas clearly", "excel in teamwork", "a swift response"
- part of a company, award, person or place name: "Oracle Financial Services" as an employer, "Spark Innovation award"
- a single letter or fragment of spaced-out text, like the R in "D E V E L O P E R"

If you are unsure, answer true.

Items:
{items}

Return JSON only, one entry per item, in the same order:
{{"labels": [{{"n": 1, "is_skill": true}}, {{"n": 2, "is_skill": false}}]}}"""


def format_item(n, m):
    left = m["left"].replace("\n", " ")[-70:]
    right = m["right"].replace("\n", " ")[:70]
    return f'{n}. skill="{m["skill_name"]}" ({m["category"]}) | context: "...{left}[[{m["surface"]}]]{right}..."'


def ask(chunk, model, url):
    items = "\n".join(format_item(i + 1, m) for i, m in enumerate(chunk))
    resp = requests.post(url, json={
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(items=items)}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    raw = json.loads(content).get("labels", [])
    # Accept both [{"n": 1, "is_skill": true}, ...] and {"1": true, ...}
    if isinstance(raw, dict):
        answers = {str(k): v for k, v in raw.items()}
    else:
        answers = {str(x.get("n")): x.get("is_skill") for x in raw if isinstance(x, dict)}
    out = []
    for i, m in enumerate(chunk):
        v = answers.get(str(i + 1))
        if isinstance(v, bool):
            out.append({"id": m["id"], "label": int(v), "source": "llm", "model": model})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1500, help="how many documents to label")
    ap.add_argument("--batch", type=int, default=10, help="mentions per LLM call")
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mentions = read_jsonl(config.MENTIONS_FILE)
    if not mentions:
        raise SystemExit("No mentions found. Run 02_find_mentions.py first.")
    done = {r["id"] for r in read_jsonl(config.LLM_LABELS_FILE)}

    by_doc = defaultdict(list)
    for m in mentions:
        by_doc[m["doc_id"]].append(m)
    doc_ids = sorted(by_doc)
    random.Random(args.seed).shuffle(doc_ids)
    doc_ids = doc_ids[: args.docs]

    todo = [m for d in doc_ids for m in by_doc[d] if m["id"] not in done]
    print(f"{len(doc_ids)} docs selected | {len(done)} already labeled | {len(todo)} to label")

    missed = 0
    for i in tqdm(range(0, len(todo), args.batch), desc="Labeling"):
        chunk = todo[i:i + args.batch]
        try:
            rows = ask(chunk, args.model, config.OLLAMA_URL)
        except Exception as e:
            print(f"\nBatch failed ({e}); it will be retried on the next run.")
            continue
        missed += len(chunk) - len(rows)
        append_jsonl(config.LLM_LABELS_FILE, rows)

    total = read_jsonl(config.LLM_LABELS_FILE)
    pos = sum(r["label"] for r in total)
    print(f"\nLabels saved: {len(total)} ({pos} skill / {len(total) - pos} not skill)")
    if missed:
        print(f"{missed} mentions got no answer; re-run to retry them.")


if __name__ == "__main__":
    main()
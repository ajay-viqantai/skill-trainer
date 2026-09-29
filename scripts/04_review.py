"""Step 4: hand-check a sample of the LLM labels.

    python scripts/04_review.py sample --n 300
      -> open data/review/review_sample.csv, fill the human_label column
         with 1 (real skill) or 0 (not a skill), leave blank to skip
    python scripts/04_review.py merge
    python scripts/04_review.py check   (after re-labeling: compare the new
                                         LLM labels with your saved answers)

Merged human labels override LLM labels during training, and the held-out
human labels become your most trustworthy test set.
"""
import argparse
import csv
import random

from _common import read_jsonl, write_jsonl
import config

SAMPLE_FILE = config.REVIEW_DIR / "review_sample.csv"


def sample(n, seed):
    mentions = {m["id"]: m for m in read_jsonl(config.MENTIONS_FILE)}
    labels = read_jsonl(config.LLM_LABELS_FILE)
    if not labels:
        raise SystemExit("No LLM labels yet. Run 03_label_with_llm.py first.")
    rng = random.Random(seed)

    # Oversample the interesting cases: ambiguous skills and LLM 'not skill'
    hard = [l for l in labels if l["label"] == 0
            or mentions[l["id"]]["features"]["is_ambiguous"]]
    easy = [l for l in labels if l not in hard]
    k_hard = min(len(hard), n // 2)
    picked = rng.sample(hard, k_hard) + rng.sample(easy, min(len(easy), n - k_hard))
    rng.shuffle(picked)

    config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "skill", "category", "context", "llm_label", "human_label"])
        for l in picked:
            m = mentions[l["id"]]
            ctx = (m["left"][-60:] + "[[" + m["surface"] + "]]" + m["right"][:60]).replace("\n", " ")
            w.writerow([l["id"], m["skill_name"], m["category"], ctx, l["label"], ""])
    print(f"Wrote {len(picked)} rows -> {SAMPLE_FILE}")


def merge():
    if not SAMPLE_FILE.exists():
        raise SystemExit("No review file. Run: python scripts/04_review.py sample")
    existing = {r["id"]: r for r in read_jsonl(config.HUMAN_LABELS_FILE)}
    agree = total = 0
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row["human_label"].strip().lower()
            if v not in {"0", "1", "y", "n", "yes", "no"}:
                continue
            label = int(v in {"1", "y", "yes"})
            existing[row["id"]] = {"id": row["id"], "label": label, "source": "human"}
            total += 1
            agree += int(label == int(row["llm_label"]))
    write_jsonl(config.HUMAN_LABELS_FILE, existing.values())
    print(f"Merged {total} reviewed rows ({len(existing)} human labels total)")
    if total:
        print(f"LLM agreed with you on {agree / total:.1%} of reviewed rows")
        print("Under ~90%? Improve the prompt or library before training on LLM labels.")


def check():
    """Compare the current LLM labels with your human labels."""
    human = {r["id"]: r["label"] for r in read_jsonl(config.HUMAN_LABELS_FILE)}
    llm = {r["id"]: r["label"] for r in read_jsonl(config.LLM_LABELS_FILE)}
    mentions = {m["id"]: m for m in read_jsonl(config.MENTIONS_FILE)}
    ids = [i for i in human if i in llm and i in mentions]
    if not ids:
        raise SystemExit("No overlap between human and LLM labels. Label first, then check.")
    wrong = [i for i in ids if human[i] != llm[i]]
    print(f"Compared {len(ids)} mentions: LLM agrees with you on {1 - len(wrong) / len(ids):.1%}")
    dropped = len([i for i in human if i not in mentions])
    if dropped:
        print(f"({dropped} reviewed mentions no longer exist after library changes; ignored)")
    for i in wrong:
        m = mentions[i]
        ctx = (m["left"][-50:] + "[[" + m["surface"] + "]]" + m["right"][:50]).replace("\n", " ")
        print(f"  you={human[i]} llm={llm[i]}  {m['skill_name']:<14} {ctx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["sample", "merge", "check"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    {"sample": lambda: sample(args.n, args.seed), "merge": merge, "check": check}[args.action]()


if __name__ == "__main__":
    main()
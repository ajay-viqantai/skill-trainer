"""Step 2: run the skills library over all text and save every mention.

    python scripts/02_find_mentions.py

Writes data/mentions/mentions.jsonl (one mention per line, with features)
and prints the most common skills plus documents with zero matches.
"""
from collections import Counter

from tqdm import tqdm

from _common import write_jsonl
import config
from skillx import SkillLibrary, build_mentions


def main():
    lib = SkillLibrary(config.LIBRARY_FILE)
    files = sorted(config.TEXT_DIR.glob("*.txt"))
    print(f"Library v{lib.version}: {len(lib.skills)} skills | {len(files)} documents")

    rows, per_skill, empty_docs = [], Counter(), []
    for path in tqdm(files, desc="Finding mentions"):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        mentions = build_mentions(text, lib, config.CONTEXT_CHARS)
        if not mentions:
            empty_docs.append(doc_id)
        for m in mentions:
            m["id"] = f"{doc_id}:{m['start']}"
            m["doc_id"] = doc_id
            rows.append(m)
            per_skill[m["skill_name"]] += 1

    write_jsonl(config.MENTIONS_FILE, rows)
    print(f"\nSaved {len(rows)} mentions -> {config.MENTIONS_FILE}")
    print("\nTop 25 skills:")
    for name, n in per_skill.most_common(25):
        print(f"  {name:<20} {n}")
    print(f"\nDocuments with zero mentions: {len(empty_docs)}")
    print("Tip: open a few of these to find skills missing from the library.")


if __name__ == "__main__":
    main()

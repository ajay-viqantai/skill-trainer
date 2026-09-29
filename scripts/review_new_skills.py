"""Decide which discovered technologies to add to the library, one at a time.

    winpty python scripts/review_new_skills.py

Keys:
  1 = add
  2 = add, but it's also an ordinary word (Gin, Jest, Karma): only match exact case
  0 = don't add (concepts, duplicates, noise)
  c = change category     n = change name     b = back     q = save and quit

Saves after every answer. When done: python scripts/07_discover_skills.py apply
"""
import csv
import os

import _common  # noqa: F401
import config
from skillx import CATEGORIES

FILE = config.DISCOVERY_DIR / "new_skills.csv"
BOLD, CYAN, DIM, RESET = "\033[1m", "\033[96m", "\033[2m", "\033[0m"


def load():
    with open(FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save(fields, rows):
    with open(FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def show(i, rows):
    r = rows[i]
    answered = sum(1 for x in rows if x["add"].strip())
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{DIM}Row {i + 1}/{len(rows)}  |  answered {answered}{RESET}\n")
    print(f"{BOLD}{CYAN}{r['canonical_name']}{RESET}   {DIM}category:{RESET} {r['category']}")
    print(f"{DIM}found in {r['doc_count']} resumes  |  spellings: {r['variants']}{RESET}\n")
    print("  1 = add    2 = add (ordinary word, exact case only)    0 = skip")
    print("  c = change category    n = change name    b = back    q = quit")


def pick_category():
    for n, c in enumerate(CATEGORIES, 1):
        print(f"   {n}. {c}")
    choice = input("category number > ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
        return CATEGORIES[int(choice) - 1]
    return None


def main():
    fields, rows = load()
    i = 0
    while i < len(rows):
        if rows[i]["add"].strip():
            i += 1
            continue
        show(i, rows)
        key = input("> ").strip().lower()
        if key in {"0", "1", "2"}:
            rows[i]["add"] = key
            save(fields, rows)
            i += 1
        elif key == "c":
            cat = pick_category()
            if cat:
                rows[i]["category"] = cat
                save(fields, rows)
        elif key == "n":
            name = input("new name > ").strip()
            if name:
                rows[i]["canonical_name"] = name
                save(fields, rows)
        elif key == "b":
            j = i - 1
            while j >= 0 and not rows[j]["add"].strip():
                j -= 1
            if j >= 0:
                rows[j]["add"] = ""
                save(fields, rows)
                i = j
        elif key == "q":
            break

    chosen = sum(1 for r in rows if r["add"].strip() in {"1", "2"})
    print(f"\nSaved. {chosen} marked to add.")
    print("Next: python scripts/07_discover_skills.py apply")


if __name__ == "__main__":
    main()
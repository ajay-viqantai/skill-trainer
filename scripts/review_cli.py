"""Review mentions one at a time in the terminal.

    python scripts/review_cli.py

Keys:  1 = real skill   0 = not a skill   s = skip   b = back   q = save and quit
Saves after every answer, so you can quit and continue later.
Qwen's answer is hidden so it doesn't influence you.
"""
import csv
import os

import _common  # noqa: F401
import config

FILE = config.REVIEW_DIR / "review_sample.csv"
BOLD, YELLOW, DIM, RESET = "\033[1m", "\033[93m", "\033[2m", "\033[0m"


def load():
    with open(FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save(fields, rows):
    with open(FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def show(i, total, done, row):
    os.system("cls" if os.name == "nt" else "clear")
    ctx = row["context"].replace("[[", YELLOW + BOLD + "[[").replace("]]", "]]" + RESET)
    print(f"{DIM}Row {i + 1}/{total}  |  answered {done}{RESET}\n")
    print(f"{BOLD}Skill:{RESET} {row['skill']}  {DIM}({row['category']}){RESET}\n")
    print(f"...{ctx}...\n")
    print(f"{DIM}Is the highlighted word used as this skill here?{RESET}")
    print("  1 = yes, real skill    0 = no, not a skill    s = skip    b = back    q = quit")


def main():
    fields, rows = load()
    i = 0
    while i < len(rows):
        row = rows[i]
        if row["human_label"].strip():  # already answered, move on
            i += 1
            continue
        done = sum(1 for r in rows if r["human_label"].strip())
        show(i, len(rows), done, row)
        key = input("> ").strip().lower()
        if key in {"1", "0"}:
            row["human_label"] = key
            save(fields, rows)
            i += 1
        elif key == "s":
            i += 1
        elif key == "b":
            # step back to the previous row and clear its answer to redo it
            j = i - 1
            while j >= 0 and not rows[j]["human_label"].strip():
                j -= 1
            if j >= 0:
                rows[j]["human_label"] = ""
                save(fields, rows)
                i = j
        elif key == "q":
            break

    done = sum(1 for r in rows if r["human_label"].strip())
    print(f"\nSaved. {done}/{len(rows)} answered.")
    print("When finished, run: python scripts/04_review.py merge")


if __name__ == "__main__":
    main()
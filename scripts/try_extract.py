"""Quick manual test of an exported model.

    python scripts/try_extract.py --artifact artifacts/v1 --file data/text/<id>.txt
    python scripts/try_extract.py --artifact artifacts/v1 --text "Worked in Go and Python"
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--file")
    ap.add_argument("--text")
    ap.add_argument("--category", help="e.g. programming_language")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.artifact).resolve()))  # use the exported skillx
    from skillx.extractor import SkillExtractor

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    ex = SkillExtractor(args.artifact)
    skills = ex.by_category(text, args.category) if args.category else ex.extract(text)
    for s in skills:
        print(f"{s['skill']:<16}{s['category']:<22}{s['confidence']:<7}x{s['mentions']}  | {s['evidence']}")
    if not skills:
        print("No skills found.")


if __name__ == "__main__":
    main()

"""Show why experience extraction missed jobs in specific resumes.

    python scripts/debug_experience.py 5c32af4f9509 db06eefbb02e > debug.txt

Emails and phone numbers are masked, so the output is safe to share.
"""
import re
import sys
from datetime import date

from _common import read_jsonl
import config
from skillx.experience import _find_ranges
from skillx.sections import _header_section, line_sections

DATEISH = re.compile(r"(19|20)\d{2}|present|current|till", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")


def mask(s):
    return PHONE.sub("[phone]", EMAIL.sub("[email]", s))


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # emojis in resumes
    today = date.today()
    llm = {r["doc_id"]: r["jobs"] for r in read_jsonl(config.EXPERIENCE_DIR / "llm.jsonl")}
    for doc_id in sys.argv[1:]:
        lines = (config.TEXT_DIR / f"{doc_id}.txt").read_text(encoding="utf-8").split("\n")
        sections, _ = line_sections(lines)
        print("=" * 90)
        print(doc_id)
        heads = [f"{i}:{lines[i].strip()[:30]}->{_header_section(lines[i])}"
                 for i in range(len(lines)) if _header_section(lines[i])]
        print("Section headings found:", heads or "NONE")
        print("Lines with dates   [section]   RANGE = parser understood it")
        for i, line in enumerate(lines):
            if DATEISH.search(line) and not EMAIL.search(line):
                ok = "RANGE" if _find_ranges(line, today) else "-----"
                print(f"  {i:>3} [{sections[i][:10]:<10}] {ok} | {mask(line.strip())[:90]}")
        print("Qwen found:")
        for j in llm.get(doc_id, []):
            print(f"    {j.get('start')} -> {j.get('end')}   {str(j.get('title', ''))[:35]}")


if __name__ == "__main__":
    main()
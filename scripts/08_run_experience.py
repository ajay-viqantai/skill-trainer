"""Run experience extraction over all resumes and summarize.

    python scripts/08_run_experience.py

Writes data/experience/results.jsonl and prints:
  - how many resumes had jobs found / none found
  - resumes whose own "X years of experience" claim differs from the
    computed total (a quick accuracy check without any LLM)
"""
from tqdm import tqdm

from _common import write_jsonl
import config
from skillx import SkillLibrary
from skillx.experience import extract_experience


def main():
    lib = SkillLibrary(config.LIBRARY_FILE)
    files = sorted(config.TEXT_DIR.glob("*.txt"))
    rows = []
    for path in tqdm(files, desc="Experience"):
        exp = extract_experience(path.read_text(encoding="utf-8"), lib)
        rows.append({"doc_id": path.stem, **exp})
    config.EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.EXPERIENCE_DIR / "results.jsonl", rows)

    with_jobs = [r for r in rows if r["job_count"]]
    none = [r["doc_id"] for r in rows if not r["job_count"]]
    print(f"\n{len(rows)} resumes | jobs found in {len(with_jobs)} | none found in {len(none)}")
    if with_jobs:
        totals = sorted(r["total_years"] for r in with_jobs)
        print(f"Total experience: median {totals[len(totals) // 2]} yrs, "
              f"range {totals[0]}–{totals[-1]} yrs")

    claimed = [r for r in with_jobs if r["claimed_total_years"]]
    if claimed:
        close = [r for r in claimed if abs(r["claimed_total_years"] - r["total_years"]) <= 1.0]
        print(f"\nSelf-check: {len(claimed)} resumes state their own total experience.")
        print(f"Computed total within 1 year of their claim: {len(close)}/{len(claimed)} "
              f"({len(close) / len(claimed):.0%})")
        off = sorted(claimed, key=lambda r: -abs(r["claimed_total_years"] - r["total_years"]))[:10]
        print("Biggest differences (open these to see why):")
        for r in off:
            print(f"  {r['doc_id']}  claimed {r['claimed_total_years']:>4}  computed {r['total_years']:>4}  jobs {r['job_count']}")

    if none:
        print(f"\nNo jobs found (freshers, or a date format we miss): {', '.join(none[:15])}"
              + (" ..." if len(none) > 15 else ""))
    print(f"\nDetails: {config.EXPERIENCE_DIR / 'results.jsonl'}")


if __name__ == "__main__":
    main()
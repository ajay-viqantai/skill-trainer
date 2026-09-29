"""Paste any resume or JD text below, then run:

    python check_text.py

It prints every skill found, grouped by category, with years of experience,
and the jobs found. Uses the newest exported version in artifacts/.
"""
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PASTE YOUR RAW TEXT BETWEEN THE TRIPLE QUOTES
# ---------------------------------------------------------------------------
RAW_TEXT = """
Senior Software Engineer with 6+ years of experience in Python and Go.

Technical Skills
Languages: Python, Golang, JavaScript, SQL
Frameworks: React.js, FastAPI, NestJS
Databases: Postgres, MongoDB, Redis
Cloud & DevOps: AWS (EC2, S3), Docker, Kubernetes, GitHub Actions

Experience
Senior Engineer, Acme Technologies
Jan 2022 - Present
- Built REST APIs with FastAPI and Go, deployed on AWS using Docker.
- Always ready to go the extra mile for the team.
Software Engineer | Beta Pvt Ltd | Jun 2019 - Dec 2021
- Developed dashboards in React.js with a Python backend and PostgreSQL.

Education
B.Tech Computer Science, 2015 - 2019
"""

# Optional: show only one category, e.g. "programming_language".
# Leave as None to see everything.
ONLY_CATEGORY = None
# ---------------------------------------------------------------------------


CATEGORY_LABELS = {
    "programming_language": "Programming languages",
    "framework": "Frameworks",
    "library": "Libraries",
    "database": "Databases",
    "cloud": "Cloud",
    "devops_tool": "DevOps tools",
    "data_tool": "Data tools",
    "other_tool": "Other tools",
}


def _years_text(s):
    if s.get("years") is None and s.get("explicit_years") is None:
        return "listed only"
    parts = []
    if s.get("years") is not None:
        parts.append(f"{s['years']} yrs")
        if s.get("last_used"):
            parts.append(f"last {s['last_used']}")
    if s.get("explicit_years") is not None:
        parts.append(f"says {s['explicit_years']:g}")
    return ", ".join(parts)


def newest_artifact():
    root = Path(__file__).resolve().parent / "artifacts"
    versions = [p for p in root.glob("v*") if (p / "meta.json").exists()]
    if not versions:
        raise SystemExit("No exported version found. Run: python scripts/06_export.py --version 1 --library-only")

    def key(p):
        v = p.name[1:]
        return (0, int(v)) if v.isdigit() else (1, v)

    return sorted(versions, key=key)[-1]


def main():
    artifact = newest_artifact()
    sys.path.insert(0, str(artifact))
    from skillx.extractor import SkillExtractor

    extractor = SkillExtractor(artifact)
    result = extractor.analyze(RAW_TEXT) if hasattr(extractor, "analyze") else {
        "skills": extractor.extract(RAW_TEXT), "experience": None}
    skills = result["skills"]
    if ONLY_CATEGORY:
        skills = [s for s in skills if s["category"] == ONLY_CATEGORY]

    print(f"Using {artifact.name} ({extractor.mode} mode, {len(extractor.library.skills)} skills in library)\n")
    if not skills:
        print("No skills found.")
        return

    by_cat = {}
    for s in skills:
        by_cat.setdefault(s["category"], []).append(s)

    for cat in CATEGORY_LABELS:
        if cat not in by_cat:
            continue
        items = sorted(by_cat[cat], key=lambda s: (-s["mentions"], s["skill"]))
        print(f"{CATEGORY_LABELS[cat]} ({len(items)})")
        for s in items:
            print(f"  {s['skill']:<18} {_years_text(s):<24} | {s['evidence'][:70]}")
        print()

    print(f"Total: {len(skills)} skills")

    exp = result["experience"]
    if exp is None:
        print("\n(This export has no experience module. Export a new version to see it.)")
        return
    print(f"\nEXPERIENCE: about {exp['total_years']} years across {exp['job_count']} job(s)"
          + (f"  |  resume claims {exp['claimed_total_years']} years" if exp["claimed_total_years"] else ""))
    for j in exp["jobs"]:
        end = "present" if j["is_current"] else j["end"]
        tags = " [internship]" if j["is_internship"] else ""
        tags += " [year-only dates]" if j["approximate"] else ""
        print(f"  {j['start']} -> {end:<8} {round(j['months'] / 12, 1):>4} yrs  {j['header'][:60]}{tags}")
        if j["skills"]:
            print(f"      skills: {', '.join(j['skills'])}")


if __name__ == "__main__":
    main()
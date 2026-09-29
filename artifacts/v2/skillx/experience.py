"""Experience extraction: jobs with dates, total years, and years per skill.

    from skillx.experience import extract_experience
    exp = extract_experience(resume_text, library)

How it works (no ML, no LLM):
  1. Find date ranges ("Jan 2019 - Present", "03/2019 - 06/2021", "Feb' 23 - Apr' 25")
     in the experience section (or outside education/projects if there is none).
  2. Each date range starts a job block that runs until the next job.
  3. Skills found inside a job block get that job's dates.
  4. Overlapping periods are merged, never added twice.
  5. Explicit claims like "5+ years of experience in Python" are read separately.

All numbers are estimates: every job keeps its header text as evidence.
"""
import re
from datetime import date

from .sections import _header_section, line_sections

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_APOS = "'’`‘"
# One date: "Jan 2020", "January, 2020", "Feb' 23", "03/2019", "2019"
_DATE = (rf"(?:{_MON}\.?[\s,{_APOS}\-]*(?:(?:19|20)\d{{2}}|[{_APOS}]?\s?\d{{2}})"
         r"|(?:0?[1-9]|1[0-2])\s*[/.\-]\s*(?:19|20)\d{2}"
         r"|(?:19|20)\d{2})")
_PRESENT = (r"(?:present|current(?:ly)?|till\s+(?:date|now)|to\s+date|until\s+now|"
            r"ongoing|now|today|running)")
_SEP = r"\s*(?:-|–|—|to|till|until|~|→)\s*"
RANGE = re.compile(rf"(?<![\w/])(?P<a>{_DATE}){_SEP}(?P<b>{_DATE}|{_PRESENT})(?![\w/])",
                   re.IGNORECASE)

_MON_YEAR = re.compile(rf"^(?P<m>{_MON})\.?[\s,{_APOS}\-]*[{_APOS}]?\s?(?P<y>\d{{4}}|\d{{2}})$",
                       re.IGNORECASE)
_NUM = re.compile(r"^(?P<m>0?[1-9]|1[0-2])\s*[/.\-]\s*(?P<y>(?:19|20)\d{2})$")
_YEAR = re.compile(r"^(?P<y>(?:19|20)\d{2})$")
_PRESENT_RE = re.compile(rf"^{_PRESENT}$", re.IGNORECASE)

EDU_WORDS = re.compile(
    r"\b(b\.?\s?tech|m\.?\s?tech|b\.?e\b|bachelor|master|university|college|school|"
    r"degree|cgpa|gpa|percentage|hsc|ssc|10th|12th|graduat|diploma|mca|bca|b\.?sc|"
    r"m\.?sc|mba|phd|class\s+x|board)", re.IGNORECASE)
INTERN_WORDS = re.compile(r"\b(intern|internship|trainee\s+intern)\b", re.IGNORECASE)
BULLET = re.compile(r"^\s*[•\-\*▪●◦·➢►✓]")

# "5+ years of experience in Python, Go" / "Python (3 years)"
YEARS = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)


def _parse(s, is_end, today):
    """Date text -> (year, month, approximate) or None. 'present' -> today."""
    s = s.strip().strip(".,")
    if _PRESENT_RE.match(s):
        return today.year, today.month, False
    m = _MON_YEAR.match(s)
    if m:
        y = int(m.group("y"))
        if y < 100:
            y += 2000 if y <= today.year % 100 + 1 else 1900
        return y, MONTHS[m.group("m")[:3].lower()], False
    m = _NUM.match(s)
    if m:
        return int(m.group("y")), int(m.group("m")), False
    m = _YEAR.match(s)
    if m:  # year only: assume mid-year so "2019 - 2021" is about 2 years
        return int(m.group("y")), 6 if is_end else 7, True
    return None


def _idx(y, m):
    return y * 12 + (m - 1)


def _fmt(i):
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def _merge(intervals):
    """Union of [start, end] month indexes."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _months(intervals):
    return sum(e - s + 1 for s, e in _merge(intervals))


def _years(months):
    return round(months / 12, 1)


def _find_ranges(line, today):
    found = []
    for m in RANGE.finditer(line):
        a = _parse(m.group("a"), False, today)
        b = _parse(m.group("b"), True, today)
        if not a or not b:
            continue
        start, end = _idx(a[0], a[1]), _idx(b[0], b[1])
        now = _idx(today.year, today.month)
        if not (1980 <= a[0] <= today.year) or start > now:
            continue
        end = min(end, now)
        if end < start:
            if a[2] and b[2] and a[0] == b[0]:  # "2021 - 2021"
                end = start + 5
            else:
                continue
        if end - start > 45 * 12:
            continue
        current = bool(_PRESENT_RE.match(m.group("b").strip().strip(".,")))
        found.append({"start": start, "end": end, "current": current,
                      "approximate": a[2] or b[2], "span": m.span()})
    return found


def _is_header_line(line):
    t = line.strip()
    return (0 < len(t) <= 90 and not BULLET.match(t) and not t.endswith(".")
            and len(t.split()) <= 12 and _header_section(t) is None)


def extract_experience(text, library, today=None, include_internships=True):
    """Jobs, total experience and years per skill from resume text."""
    today = today or date.today()
    lines = text.split("\n")
    sections, _ = line_sections(lines)
    has_exp_section = "experience" in sections

    def allowed(i):
        if has_exp_section:
            return sections[i] == "experience"
        return sections[i] in {"other", "summary"} and not EDU_WORDS.search(lines[i])

    # 1. Anchors: first date range on each allowed line
    anchors = []
    for i, line in enumerate(lines):
        if not allowed(i):
            continue
        ranges = _find_ranges(line, today)
        if ranges:
            anchors.append((i, ranges[0]))

    # 2. Job blocks: header may start up to 2 lines above the date line
    starts = []
    for k, (i, _) in enumerate(anchors):
        floor = anchors[k - 1][0] + 1 if k else 0
        s = i
        lookback = 2 if has_exp_section else 0  # without sections, lines above are unrelated
        while s - 1 >= floor and i - (s - 1) <= lookback and allowed(s - 1) and _is_header_line(lines[s - 1]):
            s -= 1
        starts.append(s)

    jobs = []
    for k, (i, r) in enumerate(anchors):
        end_line = starts[k + 1] if k + 1 < len(anchors) else len(lines)
        block = []
        for j in range(starts[k], end_line):
            if j > i and (not allowed(j) or _header_section(lines[j])):
                break  # left the experience section
            block.append(lines[j])
        a, b = r["span"]
        header_parts = [ln.strip() for ln in lines[starts[k]:i]] + \
                       [(lines[i][:a] + " " + lines[i][b:]).strip(" |,-–—()")]
        header = " | ".join(p for p in header_parts if p)[:160]
        block_text = "\n".join(block)
        skill_ids = sorted({h[3] for h in library.find(block_text)})
        jobs.append({
            "header": header,
            "start": _fmt(r["start"]),
            "end": None if r["current"] else _fmt(r["end"]),
            "is_current": r["current"],
            "approximate": r["approximate"],
            "months": r["end"] - r["start"] + 1,
            "is_internship": bool(INTERN_WORDS.search(header)),
            "skill_ids": skill_ids,
            "_interval": (r["start"], r["end"]),
        })

    counted = [j for j in jobs if include_internships or not j["is_internship"]]
    total_months = _months([j["_interval"] for j in counted])

    # 3. Years per skill from job blocks
    per_skill = {}
    for j in counted:
        for sid in j["skill_ids"]:
            per_skill.setdefault(sid, []).append(j)

    skills = {}
    for sid, js in per_skill.items():
        last = max(j["_interval"][1] for j in js)
        skills[sid] = {
            "years": _years(_months([j["_interval"] for j in js])),
            "roles": len(js),
            "last_used": "present" if any(j["is_current"] for j in js) else _fmt(last),
            "explicit_years": None,
        }

    # 4. Explicit claims
    claimed_total = None
    for m in YEARS.finditer(text):
        yrs = float(m.group(1))
        if not 0 < yrs <= 40:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        after = re.split(r"[.;\n]", text[m.end():m.end() + 120])[0]
        before = text[line_start:m.start()]
        near = (text[max(line_start, m.start() - 40):m.start()] + after[:40]).lower()

        targets = {h[3] for h in library.find(after)} if re.match(
            r"\s*(of\s+)?(hands[-\s]on\s+|professional\s+|relevant\s+|industry\s+|work\s+)?"
            r"(experience|exp)?\s*(in|with|on|using|of|:)", after, re.IGNORECASE) else set()
        # "Python (3 years)" / "Python - 3 yrs"
        for h in library.find(before):
            if re.fullmatch(r"\s*[:(\-–]\s*", before[h[1]:]):
                targets.add(h[3])

        if targets:
            for sid in targets:
                entry = skills.setdefault(sid, {"years": None, "roles": 0,
                                                "last_used": None, "explicit_years": None})
                entry["explicit_years"] = max(entry["explicit_years"] or 0, yrs)
        elif "experience" in near or "exp" in near.split():
            claimed_total = max(claimed_total or 0, yrs)

    for j in jobs:
        j.pop("_interval")

    return {
        "total_years": _years(total_months),
        "claimed_total_years": claimed_total,
        "job_count": len(jobs),
        "jobs": jobs,
        "skills": skills,  # skill_id -> years / roles / last_used / explicit_years
    }

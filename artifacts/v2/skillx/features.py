"""Mention finding + feature building.

This file is shared by training and the main app, so both compute
features in exactly the same way. Change it -> retrain the model.
"""
import bisect
import re

from .library import CATEGORIES
from .sections import line_sections

TECH_WORDS = {
    "developer", "developed", "develop", "development", "programming", "programmer",
    "language", "languages", "framework", "frameworks", "library", "libraries",
    "experience", "experienced", "proficient", "proficiency", "knowledge",
    "skills", "skill", "using", "used", "built", "build", "coding", "code",
    "engineer", "stack", "backend", "frontend", "api", "apis", "database",
    "scripting", "scripts", "cloud", "deployed", "technologies", "tools",
    "hands-on", "expertise", "familiar", "worked", "working",
}
STOP_BEFORE = {"to", "the", "a", "an", "will", "can", "could", "would", "let",
               "lets", "must", "should", "we", "i", "they", "you", "not",
               "may", "might", "please", "always", "never", "that", "this"}
STOP_AFTER = {"to", "the", "a", "an", "for", "of", "through", "ahead", "back",
              "up", "out", "on", "into", "beyond", "away"}
SEPARATORS = set(",|/•;·()[]")
BULLET_START = re.compile(r"^\s*[•\-\*▪●◦·]\s*")
VERSION_AFTER = re.compile(r"^\s*v?\d")
WORD = re.compile(r"[A-Za-z][A-Za-z+#.\-]*")

FEATURE_NAMES = [
    "category_code", "is_ambiguous", "case_sensitive", "canonical_form",
    "is_lower", "is_upper", "is_title", "mention_len",
    "in_skills", "in_experience", "in_projects", "in_summary", "in_education",
    "inline_skills_line", "line_known_skills", "line_tokens", "line_separators",
    "adjacent_separator", "tech_word_near", "stop_before", "stop_after",
    "version_after", "sentence_start", "bullet_line",
]


def _prev_char(text, i):
    i -= 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return text[i] if i >= 0 else "\n"


def _next_char(text, i):
    while i < len(text) and text[i] in " \t":
        i += 1
    return text[i] if i < len(text) else "\n"


def build_mentions(text, library, context_chars=80):
    """Find every skill mention in text and compute its features.

    Returns a list of dicts: start, end, surface, skill_id, skill_name,
    category, left, right (context), features (dict of FEATURE_NAMES).
    """
    lines = text.split("\n")
    line_starts, pos = [], 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln) + 1
    sections, inline = line_sections(lines)

    hits = library.find(text)
    per_line = {}
    for h in hits:
        li = bisect.bisect_right(line_starts, h[0]) - 1
        per_line[li] = per_line.get(li, 0) + 1

    out = []
    for start, end, surface, sid in hits:
        skill = library.skills[sid]
        li = bisect.bisect_right(line_starts, start) - 1
        line = lines[li]
        sec = sections[li]

        before_words = [w.lower() for w in WORD.findall(text[max(0, start - 60):start])]
        after_words = [w.lower() for w in WORD.findall(text[end:end + 60])]
        window = set(before_words[-5:]) | set(after_words[:5])
        pc, nc = _prev_char(text, start), _next_char(text, end)

        f = {
            "category_code": CATEGORIES.index(skill.category),
            "is_ambiguous": int(skill.ambiguous),
            "case_sensitive": int(skill.case_sensitive),
            "canonical_form": int(library.is_canonical_form(sid, surface)),
            "is_lower": int(surface.islower()),
            "is_upper": int(surface.isupper() and len(surface) > 1),
            "is_title": int(surface[:1].isupper() and not surface.isupper()),
            "mention_len": len(surface),
            "in_skills": int(sec == "skills"),
            "in_experience": int(sec == "experience"),
            "in_projects": int(sec == "projects"),
            "in_summary": int(sec == "summary"),
            "in_education": int(sec == "education"),
            "inline_skills_line": int(inline[li]),
            "line_known_skills": per_line.get(li, 0),
            "line_tokens": len(line.split()),
            "line_separators": sum(ch in SEPARATORS for ch in line),
            "adjacent_separator": int(pc in SEPARATORS or nc in SEPARATORS),
            "tech_word_near": int(bool(window & TECH_WORDS)),
            "stop_before": int(bool(before_words) and before_words[-1] in STOP_BEFORE),
            "stop_after": int(bool(after_words) and after_words[0] in STOP_AFTER),
            "version_after": int(bool(VERSION_AFTER.match(text[end:end + 6]))),
            "sentence_start": int(pc in ".!?\n"),
            "bullet_line": int(bool(BULLET_START.match(line))),
        }
        out.append({
            "start": start,
            "end": end,
            "surface": surface,
            "skill_id": sid,
            "skill_name": skill.name,
            "category": skill.category,
            "left": text[max(0, start - context_chars):start],
            "right": text[end:end + context_chars],
            "features": f,
        })
    return out
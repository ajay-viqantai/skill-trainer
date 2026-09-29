"""Skills library: canonical skills, categories, aliases, and a fast matcher."""
import json
import re
from dataclasses import dataclass, field

CATEGORIES = [
    "programming_language",
    "framework",
    "library",
    "database",
    "cloud",
    "devops_tool",
    "data_tool",
    "other_tool",
]


@dataclass
class Skill:
    id: str
    name: str
    category: str
    aliases: list = field(default_factory=list)
    ambiguous: bool = False       # also an ordinary English word (Go, Spark, Swift)
    case_sensitive: bool = False  # only match exact case (C, R, Go)


def _compile(forms, flags):
    forms = sorted(set(forms), key=len, reverse=True)  # longest first wins
    if not forms:
        return None
    alt = "|".join(re.escape(f) for f in forms)
    # Not glued to letters/digits or . @ _ on the left (so "JS" doesn't
    # match inside "Moment.js"); not followed by letters, digits, + or #
    # (so "C" doesn't match inside "C++" or "C#").
    return re.compile(rf"(?<![A-Za-z0-9.@_])(?:{alt})(?![A-Za-z0-9+#])", flags)


_URLISH = re.compile(r"(https?://|www\.|@|\.(com|io|in|org|net|dev|me)\b)", re.IGNORECASE)


def _in_spaced_letters(text, start, end):
    """True for a single letter inside spaced-out text like 'D E V E L O P E R'."""
    if end - start != 1:
        return False
    def single_letter_at(i):
        return (0 <= i < len(text) and text[i].isalpha()
                and (i == 0 or not text[i - 1].isalnum())
                and (i + 1 >= len(text) or not text[i + 1].isalnum()))
    return ((start >= 2 and text[start - 1] == " " and single_letter_at(start - 2))
            or (end + 1 < len(text) and text[end] == " " and single_letter_at(end + 1)))

def _inside_url_or_email(text, start, end):
    """True if the match sits inside a URL, email or path like github.com/user."""
    left = start
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace():
        right += 1
    token = text[left:right]
    return bool(_URLISH.search(token)) and len(token) > (end - start) + 3


class SkillLibrary:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.version = str(data.get("version", "0"))
        self.skills = {}
        self._ci = {}  # lowercased form -> skill id
        self._cs = {}  # exact form -> skill id

        for raw in data["skills"]:
            s = Skill(**raw)
            if s.category not in CATEGORIES:
                raise ValueError(f"Unknown category '{s.category}' for {s.id}")
            if s.id in self.skills:
                raise ValueError(f"Duplicate skill id: {s.id}")
            self.skills[s.id] = s
            for form in [s.name, *s.aliases]:
                if s.case_sensitive:
                    self._cs[form] = s.id
                else:
                    self._ci[form.lower()] = s.id

        self._re_ci = _compile(self._ci.keys(), re.IGNORECASE)
        self._re_cs = _compile(self._cs.keys(), 0)

    def find(self, text):
        """Return non-overlapping (start, end, surface, skill_id), in order."""
        hits = []
        if self._re_ci:
            for m in self._re_ci.finditer(text):
                hits.append((m.start(), m.end(), m.group(), self._ci[m.group().lower()]))
        if self._re_cs:
            for m in self._re_cs.finditer(text):
                hits.append((m.start(), m.end(), m.group(), self._cs[m.group()]))

        hits = [h for h in hits
                if not _inside_url_or_email(text, h[0], h[1])
                and not _in_spaced_letters(text, h[0], h[1])]

        # Prefer longer matches when two overlap ("React Native" over "React")
        hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        result, last_end = [], -1
        for h in hits:
            if h[0] >= last_end:
                result.append(h)
                last_end = h[1]
        return result

    def is_canonical_form(self, skill_id, surface):
        s = self.skills[skill_id]
        return surface in (s.name, *s.aliases)
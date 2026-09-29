"""Rough resume section detection (skills, experience, projects, ...)."""
import re

SECTION_KEYS = {
    "skills": ["skills", "technical skills", "key skills", "core skills",
               "technologies", "tech stack", "tools", "competencies",
               "core competencies", "technical expertise", "it skills"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment", "employment history", "work history",
                   "career history", "experience summary", "relevant experience",
                   "internship", "internships", "internship experience",
                   "work experience & internships", "professional background"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "education": ["education", "academic", "qualifications", "academic qualifications"],
    "summary": ["summary", "profile", "objective", "career objective",
                "professional summary", "about me"],
    "certifications": ["certifications", "certificates", "courses", "training"],
}

_INLINE_SKILLS = re.compile(
    r"^\s*(technical\s+|key\s+|core\s+)?(skills?|technologies|tech stack|tools)\s*[:\-–]",
    re.IGNORECASE,
)


_SPACED = re.compile(r"^(?:[A-Za-z&] ){2,}[A-Za-z&]$")


def _header_section(line):
    stripped = line.strip().strip(":").strip()
    # "E X P E R I E N C E" -> "EXPERIENCE" (words split by 2+ spaces stay apart)
    if _SPACED.match(stripped.replace("  ", " ")):
        stripped = re.sub(r"(?<=\S) (?=\S)", "", stripped)
        stripped = re.sub(r"\s{2,}", " ", stripped)
    if not stripped or len(stripped) > 40:
        return None
    low = re.sub(r"[^a-z &/]", "", stripped.lower()).strip()
    if len(low.split()) > 4:
        return None
    for section, keys in SECTION_KEYS.items():
        if low in keys:
            return section
    return None


def line_sections(lines):
    """Section name for every line, plus a flag for inline 'Skills: ...' lines."""
    current = "other"
    sections, inline = [], []
    for line in lines:
        is_inline = bool(_INLINE_SKILLS.match(line))
        header = None if is_inline else _header_section(line)
        if header:
            current = header
        sections.append("skills" if is_inline else current)
        inline.append(is_inline)
    return sections, inline

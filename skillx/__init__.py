"""skillx: skill extraction shared by training and the main app.

Only this package plus the files in artifacts/<version>/ get copied into
the main app. Training scripts live in scripts/ and never ship.
"""
from .library import SkillLibrary, CATEGORIES
from .features import build_mentions, FEATURE_NAMES

__all__ = ["SkillLibrary", "CATEGORIES", "build_mentions", "FEATURE_NAMES"]

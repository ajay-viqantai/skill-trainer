"""Runtime extractor for the main app.

    from skillx.extractor import SkillExtractor
    extractor = SkillExtractor("app/ml/skills_v1")   # load once at startup
    skills = extractor.extract(resume_text)
    languages = extractor.by_category(resume_text, "programming_language")
    result = extractor.analyze(resume_text)   # skills with years + experience

Works in two modes, chosen by what's in the artifact folder:
  - library only (no model.txt): every library match counts as a skill
  - library + model: the trained model filters out false matches
"""
import json
from pathlib import Path

from .experience import extract_experience
from .features import build_mentions
from .library import SkillLibrary


def _evidence(m):
    """The line the mention sits on (trimmed), for showing 'why' in the UI."""
    left = m["left"].rsplit("\n", 1)[-1][-60:]
    right = m["right"].split("\n", 1)[0][:60]
    return (left + m["surface"] + right).strip()


class SkillExtractor:
    def __init__(self, artifact_dir):
        d = Path(artifact_dir)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.library = SkillLibrary(d / "skills_library.json")
        self.version = self.meta["version"]
        self.model = None
        if (d / "model.txt").exists():
            import lightgbm as lgb  # only needed when a model ships

            self.model = lgb.Booster(model_file=str(d / "model.txt"))
            self.threshold = float(self.meta["threshold"])
            self.feature_names = self.meta["feature_names"]

    @property
    def mode(self):
        return "model" if self.model is not None else "library"

    def _scores(self, mentions):
        if self.model is None:
            return [1.0] * len(mentions)
        import numpy as np

        X = np.array(
            [[m["features"][f] for f in self.feature_names] for m in mentions],
            dtype=float,
        )
        return list(self.model.predict(X))

    def extract(self, text):
        """Skills found in text, one entry per skill, best evidence first."""
        if not text or not text.strip():
            return []
        mentions = build_mentions(text, self.library)
        if not mentions:
            return []
        threshold = self.threshold if self.model is not None else 0.5

        skills = {}
        for m, p in zip(mentions, self._scores(mentions)):
            if p < threshold:
                continue
            s = skills.get(m["skill_id"])
            if s is None:
                skills[m["skill_id"]] = {
                    "skill_id": m["skill_id"],
                    "skill": m["skill_name"],
                    "category": m["category"],
                    "confidence": float(p),
                    "mentions": 1,
                    "evidence": _evidence(m),
                }
            else:
                s["mentions"] += 1
                if p > s["confidence"]:
                    s["confidence"] = float(p)
                    s["evidence"] = _evidence(m)

        for s in skills.values():
            s["confidence"] = round(s["confidence"], 3)
        return sorted(skills.values(), key=lambda s: (-s["confidence"], -s["mentions"], s["skill"]))

    def by_category(self, text, category):
        return [s for s in self.extract(text) if s["category"] == category]

    def analyze(self, text, today=None, include_internships=True):
        """Skills with years of experience, plus jobs and total experience.

        Each skill gets: years (from job dates, None if it only appears in a
        skills list), explicit_years (from "5 years of Python"), roles and
        last_used ("present" or "YYYY-MM").
        """
        skills = self.extract(text)
        exp = extract_experience(text, self.library, today=today,
                                 include_internships=include_internships)
        for s in skills:
            info = exp["skills"].get(s["skill_id"], {})
            s["years"] = info.get("years")
            s["explicit_years"] = info.get("explicit_years")
            s["roles"] = info.get("roles", 0)
            s["last_used"] = info.get("last_used")
        found = {s["skill_id"] for s in skills}
        for j in exp["jobs"]:  # keep job skills consistent with the model's decisions
            j["skills"] = [self.library.skills[i].name for i in j.pop("skill_ids") if i in found]
        return {
            "skills": skills,
            "experience": {k: v for k, v in exp.items() if k != "skills"},
        }
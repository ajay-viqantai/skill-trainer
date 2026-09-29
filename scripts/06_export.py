"""Step 6: package skill extraction for the main app.

    python scripts/06_export.py --version 1 --library-only
      Ships just the skills library (no model). Needs no ML packages.

    python scripts/06_export.py --version 2
      Ships the library plus the trained model from models/
      (main app then also needs: pip install lightgbm numpy).

Creates artifacts/v<version>/ with meta.json, skills_library.json,
optionally model.txt, and the skillx/ runtime package.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone

import _common  # noqa: F401
import config
from skillx import SkillLibrary

RUNTIME_FILES = ["__init__.py", "library.py", "sections.py", "features.py",
                 "experience.py", "extractor.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. 1, 2, 2026-10")
    ap.add_argument("--library-only", action="store_true", help="export without the model")
    args = ap.parse_args()

    lib = SkillLibrary(config.LIBRARY_FILE)  # validates the library
    model, meta_path = config.MODELS_DIR / "model.txt", config.MODELS_DIR / "meta.json"

    if args.library_only:
        meta = {"mode": "library"}
    else:
        if not model.exists():
            raise SystemExit("No trained model. Run 05_train.py, or export with --library-only.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("library_version") != lib.version:
            raise SystemExit(
                f"Model was trained on library v{meta.get('library_version')}, "
                f"but the library is now v{lib.version}. Re-run 02 and 05 first, "
                "or export with --library-only.")
        meta["mode"] = "model"

    out = config.ARTIFACTS_DIR / f"v{args.version}"
    if out.exists():
        raise SystemExit(f"{out} already exists. Pick a new version.")
    (out / "skillx").mkdir(parents=True)

    meta.update({
        "version": str(args.version),
        "library_version": lib.version,
        "skill_count": len(lib.skills),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.copy(config.LIBRARY_FILE, out / "skills_library.json")
    if not args.library_only:
        shutil.copy(model, out / "model.txt")
    for name in RUNTIME_FILES:
        shutil.copy(config.ROOT / "skillx" / name, out / "skillx" / name)

    size_kb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024
    print(f"Exported {out} ({meta['mode']} mode, {len(lib.skills)} skills, {size_kb:.0f} KB)")
    if args.library_only:
        print("Main app needs no extra packages.")
    else:
        print("Main app needs: pip install lightgbm numpy")


if __name__ == "__main__":
    main()
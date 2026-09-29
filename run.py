"""One menu for the whole project.

    winpty python run.py        (Git Bash on Windows)
    python run.py               (PowerShell / VS Code terminal)

Pick a number; it runs the right scripts in the right order, stops if a step
fails, and checks Ollama before any step that needs Qwen.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
ALL = "99999"  # "--docs 99999" = every resume; cached ones are skipped
TRAIN_LOG_DIR = config.MODELS_DIR / "logs"
LATEST_TRAIN_LOG = config.MODELS_DIR / "train_log.txt"

BOLD, GREEN, YELLOW, RED, DIM, RESET = "\033[1m", "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"


# ----------------------------------------------------------------------------- helpers
def say(msg, color=""):
    print(f"{color}{msg}{RESET}")


def run(script, *args):
    """Run a script; return True if it succeeded."""
    cmd = [PY, str(ROOT / script), *args]
    say(f"\n▶ {script} {' '.join(args)}", BOLD)
    code = subprocess.call(cmd, cwd=ROOT, env=ENV)
    if code != 0:
        say(f"✗ {script} failed (exit code {code}). Stopping here.", RED)
        return False
    return True


def steps(*items):
    """Run (script, args...) items in order, stopping at the first failure."""
    for item in items:
        if not run(*item):
            return False
    return True


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default if default is not None else "")


def confirm(prompt, default=True):
    d = "Y/n" if default else "y/N"
    value = input(f"{prompt} ({d}): ").strip().lower()
    return default if not value else value.startswith("y")


def ollama_ready():
    """True if Ollama answers and the configured model is pulled."""
    try:
        import requests

        tags = requests.get("http://localhost:11434/api/tags", timeout=3).json()
    except Exception:
        say("✗ Ollama is not running. Open the Ollama app, then try again.", RED)
        return False
    names = {m.get("name", "") for m in tags.get("models", [])}
    if config.OLLAMA_MODEL not in names:
        say(f"✗ Model {config.OLLAMA_MODEL} not found. Run: ollama pull {config.OLLAMA_MODEL}", RED)
        return False
    return True


def count_lines(path):
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def text_count():
    return len(list(config.TEXT_DIR.glob("*.txt"))) if config.TEXT_DIR.exists() else 0


def raw_count():
    if not config.RAW_DIR.exists():
        return 0
    return sum(1 for p in config.RAW_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".docx"})


def library_info():
    data = json.loads(config.LIBRARY_FILE.read_text(encoding="utf-8"))
    return data.get("version", "?"), len(data["skills"])


def versions():
    out = []
    if config.ARTIFACTS_DIR.exists():
        for p in config.ARTIFACTS_DIR.glob("v*"):
            if p.name[1:].isdigit() and (p / "meta.json").exists():
                out.append(int(p.name[1:]))
    return sorted(out)


def next_version():
    v = versions()
    return str(v[-1] + 1 if v else 1)


def pending(cache, key="doc_id"):
    """How many extracted resumes are not in a Qwen cache yet."""
    done = set()
    if Path(cache).exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line).get(key))
    return len({p.stem for p in config.TEXT_DIR.glob("*.txt")} - done)


def train_logs():
    """Saved training logs, newest first."""
    if not TRAIN_LOG_DIR.exists():
        return []
    return sorted(TRAIN_LOG_DIR.glob("train_*.txt"), reverse=True)


# ----------------------------------------------------------------------------- actions
def status():
    say("\n── Status ─────────────────────────────────────────", BOLD)
    say(f"Resumes in data/raw:       {raw_count()}")
    say(f"Extracted to text:         {text_count()}")
    if config.MANIFEST.exists():
        with open(config.MANIFEST, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        bad = sum(r["status"] != "ok" for r in rows)
        if bad:
            say(f"  ({bad} scanned/broken files, see data/manifest.csv)", YELLOW)
    v, n = library_info()
    say(f"Skills library:            v{v}, {n} skills")
    say(f"Skill mentions found:      {count_lines(config.MENTIONS_FILE)}")
    say(f"Discovery waiting (Qwen):  {pending(config.DISCOVERY_DIR / 'raw.jsonl')} resumes")
    say(f"Experience check (Qwen):   {pending(config.EXPERIENCE_DIR / 'llm.jsonl')} resumes not checked")

    meta_path = config.MODELS_DIR / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = m.get("metrics", {})
        lgb_f1 = metrics.get("lightgbm", {}).get("f1")
        lib_f1 = metrics.get("library_mode (backend now)", {}).get("f1")
        say(f"Last model training:       {m.get('trained_at', '?')} (library v{m.get('library_version', '?')})")
        if lgb_f1 is not None and lib_f1 is not None:
            say(f"  LightGBM F1 {lgb_f1} vs backend library mode {lib_f1}")
        logs = train_logs()
        if logs:
            say(f"  Latest log: {LATEST_TRAIN_LOG}  ({len(logs)} saved in {TRAIN_LOG_DIR})", DIM)
        if str(m.get("library_version")) != str(v):
            say("  Model is older than the library: retrain (option 9) before exporting with it.", YELLOW)
    else:
        say("Last model training:       none (library-only mode)", DIM)

    vs = versions()
    if vs:
        meta = json.loads((config.ARTIFACTS_DIR / f"v{vs[-1]}" / "meta.json").read_text(encoding="utf-8"))
        say(f"Latest export:             v{vs[-1]} ({meta.get('mode', '?')} mode, "
            f"{meta.get('skill_count', '?')} skills, library v{meta.get('library_version', '?')})")
        if str(meta.get("library_version")) != str(v):
            say(f"  Library changed since v{vs[-1]}: export a new version (option 8).", YELLOW)
    else:
        say("Latest export:             none yet", YELLOW)
    try:
        import requests

        requests.get("http://localhost:11434/api/tags", timeout=2)
        say("Ollama:                    running", GREEN)
    except Exception:
        say("Ollama:                    not running (needed for Qwen steps)", YELLOW)


def extract_and_scan():
    say(f"{raw_count()} resumes in data/raw.")
    return steps(("scripts/01_extract_text.py",), ("scripts/02_find_mentions.py",))


def discover_skills():
    n = pending(config.DISCOVERY_DIR / "raw.jsonl")
    say(f"{n} resumes not yet read by Qwen for new skills (about 10–20 s each).")
    if n and not confirm("Start?"):
        return False
    if n and not ollama_ready():
        return False
    if not run("scripts/07_discover_skills.py", "extract", "--docs", ALL):
        return False
    say("\nNow choose which technologies to add.", BOLD)
    if not run("scripts/review_new_skills.py"):
        return False
    csv_file = config.DISCOVERY_DIR / "new_skills.csv"
    marked = 0
    if csv_file.exists():
        with open(csv_file, encoding="utf-8-sig") as f:
            marked = sum(r["add"].strip() in {"1", "2"} for r in csv.DictReader(f))
    if not marked:
        say("Nothing marked to add: library unchanged.", DIM)
        return True
    if not confirm(f"Add {marked} marked skills to the library now?"):
        return True
    return steps(("scripts/07_discover_skills.py", "apply"), ("scripts/02_find_mentions.py",))


def experience():
    if not run("scripts/08_run_experience.py"):
        return False
    if not count_lines(config.EXPERIENCE_DIR / "results.jsonl"):
        say("No resumes to check yet: run option 3 first.", YELLOW)
        return True
    n = pending(config.EXPERIENCE_DIR / "llm.jsonl")
    if n:
        say(f"\n{n} resumes not yet checked by Qwen (about 6 s each).")
        if not confirm("Check them with Qwen now?"):
            return True
        if not ollama_ready():
            return False
    return run("scripts/09_check_experience_llm.py", "--docs", ALL)


def debug_experience():
    ids = ask("Resume IDs, separated by spaces (from the 'biggest disagreements' list)")
    if not ids:
        return False
    out = ROOT / "debug_exp.txt"
    with open(out, "w", encoding="utf-8") as f:
        code = subprocess.call([PY, str(ROOT / "scripts/debug_experience.py"), *ids.split()],
                               cwd=ROOT, env=ENV, stdout=f)
    if code != 0:
        say("✗ Debug failed: check the IDs.", RED)
        return False
    print(out.read_text(encoding="utf-8"))
    say(f"(also saved to {out.name})", DIM)
    return True


def export():
    v = ask("Version number", next_version())
    if v.isdigit() and int(v) in versions():
        say(f"v{v} already exists. Pick a new number.", RED)
        return False
    with_model = (config.MODELS_DIR / "model.txt").exists() and confirm(
        "A trained model exists. Include it? (No = library-only, recommended)", default=False)
    args = ["--version", v] + ([] if with_model else ["--library-only"])
    return steps(("scripts/06_export.py", *args), ("check_text.py",))


def full_update():
    say("Full update for new resumes:", BOLD)
    say("  text → skills → discover new skills (Qwen) → review → experience (+ Qwen check) → export")
    if not confirm("Continue?"):
        return False
    return (extract_and_scan() and discover_skills() and experience() and export())


def train_model():
    say("Optional model (library-only mode is what currently ships).", BOLD)
    say("  label mentions with Qwen → review a sample → train → compare with library mode")
    if not confirm("Continue?"):
        return False
    if not ollama_ready():
        return False
    if not steps(("scripts/02_find_mentions.py",),
                 ("scripts/03_label_with_llm.py", "--docs", ask("How many resumes to label", ALL))):
        return False
    if confirm("Create a new review sample? (No = keep your existing answers)", default=False):
        if not run("scripts/04_review.py", "sample", "--n", ask("Rows", "150")):
            return False
        run("scripts/review_cli.py")
        if not run("scripts/04_review.py", "merge"):
            return False
    ok = steps(("scripts/04_review.py", "check"), ("scripts/05_train.py",))
    logs = train_logs()
    if ok and logs:
        say(f"\nTraining log saved: {logs[0]}", GREEN)
        say(f"Latest copy:        {LATEST_TRAIN_LOG}", DIM)
        if len(logs) > 1:
            say(f"Earlier runs:       {len(logs) - 1} more in {TRAIN_LOG_DIR}", DIM)
    return ok


MENU = [
    ("Status (what's there, what's pending)", status),
    ("Full update for new resumes (runs 3 → 4 → 5 → 8 in order)", full_update),
    ("Extract text + find skills", extract_and_scan),
    ("Discover new skills with Qwen, review, add to library", discover_skills),
    ("Experience: run + check with Qwen", experience),
    ("Debug experience for specific resumes", debug_experience),
    ("Test: paste text in check_text.py, then run it", lambda: run("check_text.py")),
    ("Export a new version for the backend (+ quick test)", export),
    ("Optional: train the model (label, review, train)", train_model),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # menu symbols on Windows
    os.chdir(ROOT)
    while True:
        say("\n══════════ skill-trainer ══════════", BOLD)
        for n, (label, _) in enumerate(MENU, 1):
            print(f"  {n}. {label}")
        print("  0. Exit")
        choice = input("\nChoose: ").strip()
        if choice == "0":
            break
        if not (choice.isdigit() and 1 <= int(choice) <= len(MENU)):
            say("Pick a number from the menu.", YELLOW)
            continue
        label, action = MENU[int(choice) - 1]
        try:
            ok = action()
        except KeyboardInterrupt:
            say("\nStopped. Progress in Qwen steps is saved; run the same option again to continue.", YELLOW)
            continue
        if ok is not False:
            say(f"\n✓ Done: {label}", GREEN)
        input(f"{DIM}Press Enter for the menu...{RESET}")


if __name__ == "__main__":
    main()
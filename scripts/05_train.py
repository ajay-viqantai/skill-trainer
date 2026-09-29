"""Step 5: train the decision-tree models and compare them with regex.

    python scripts/05_train.py

Splits by document (no resume appears in both train and test), trains a
plain decision tree and LightGBM, and compares them with three baselines:
  - library_only: every library match counts as a skill (plain regex)
  - library_mode: what the backend ships today (library + ambiguous-word
                  and single-letter rules from skillx/extractor.py)
  - simple_rules: library match, minus ambiguous words outside skill lists
Saves the LightGBM model to models/ if it trained successfully.

Every run is logged automatically:
  models/logs/train_YYYY-MM-DD_HH-MM-SS.txt   one file per run (history)
  models/train_log.txt                        always the latest run
"""
import json
import sys
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from _common import read_jsonl
import config
from skillx import FEATURE_NAMES, SkillLibrary
from skillx.extractor import SkillExtractor

# LightGBM renamed eval_set in newer versions; the old name still works
warnings.filterwarnings("ignore", message=".*eval_set.*")

LOG_DIR = config.MODELS_DIR / "logs"
LATEST_LOG = config.MODELS_DIR / "train_log.txt"


class _Tee:
    """Print to the terminal AND write the same text to one or more log files."""

    def __init__(self, *paths):
        self.files = []
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files.append(open(path, "w", encoding="utf-8"))
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        for f in self.files:
            f.write(text)

    def flush(self):
        self.stdout.flush()
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            f.close()


def load_dataset():
    mentions = read_jsonl(config.MENTIONS_FILE)
    llm = {r["id"]: r["label"] for r in read_jsonl(config.LLM_LABELS_FILE)}
    human = {r["id"]: r["label"] for r in read_jsonl(config.HUMAN_LABELS_FILE)}

    rows = []
    for m in mentions:
        if m["id"] in human:
            label, src = human[m["id"]], "human"
        elif m["id"] in llm:
            label, src = llm[m["id"]], "llm"
        else:
            continue
        rows.append({"id": m["id"], "doc_id": m["doc_id"], "skill": m["skill_name"],
                     "label": label, "source": src, **m["features"]})
    return pd.DataFrame(rows)


def scores(y, pred):
    return {
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
    }


def best_threshold(y, prob):
    grid = np.arange(0.2, 0.81, 0.05)
    f1s = [f1_score(y, (prob >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(f1s))], 2))


def library_mode_predictions(X):
    """What the backend does today in library mode (no model)."""
    return np.array([
        int(SkillExtractor._library_score({"features": row}) >= 0.5)
        for row in X.to_dict("records")
    ])


def main():
    df = load_dataset()
    if len(df) < 200:
        raise SystemExit(f"Only {len(df)} labeled mentions. Label more before training.")
    library_version = SkillLibrary(config.LIBRARY_FILE).version
    print(f"Library v{library_version} | labeled mentions: {len(df)} from {df.doc_id.nunique()} docs "
          f"({df.label.mean():.1%} real skills, {int((df.source == 'human').sum())} human-checked)")

    X, y, groups = df[FEATURE_NAMES].astype(float), df["label"].values, df["doc_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=1)
    fit_i, val_i = next(inner.split(X.iloc[train_idx], y[train_idx], groups[train_idx]))
    fit_idx, val_idx = train_idx[fit_i], train_idx[val_i]

    Xte, yte = X.iloc[test_idx], y[test_idx]
    results = {}

    # Baselines
    results["library_only (plain regex)"] = scores(yte, np.ones_like(yte))
    lib_mode = library_mode_predictions(Xte)
    results["library_mode (backend now)"] = scores(yte, lib_mode)
    rules = ((Xte["is_ambiguous"] == 0) | (Xte["in_skills"] == 1)
             | (Xte["adjacent_separator"] == 1)).astype(int).values
    results["simple_rules"] = scores(yte, rules)

    # Plain decision tree
    tree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=0)
    tree.fit(X.iloc[train_idx], y[train_idx])
    results["decision_tree"] = scores(yte, tree.predict(Xte))

    # LightGBM (boosted trees) with early stopping on a validation split
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, subsample=0.9, subsample_freq=1,
                               colsample_bytree=0.9, random_state=0, verbose=-1)
    model.fit(X.iloc[fit_idx], y[fit_idx],
              eval_set=[(X.iloc[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    threshold = best_threshold(y[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1])
    prob_te = model.predict_proba(Xte)[:, 1]
    results["lightgbm"] = scores(yte, (prob_te >= threshold).astype(int))

    print(f"\nTest set: {len(test_idx)} mentions from {len(set(groups[test_idx]))} unseen docs")
    print(f"{'model':<30}{'precision':>10}{'recall':>9}{'f1':>8}")
    for name, s in results.items():
        print(f"{name:<30}{s['precision']:>10}{s['recall']:>9}{s['f1']:>8}")

    # Most trustworthy numbers: human-checked mentions in the test set
    human_mask = (df.iloc[test_idx]["source"] == "human").values
    if human_mask.sum() >= 30:
        hs = scores(yte[human_mask], (prob_te[human_mask] >= threshold).astype(int))
        ls = scores(yte[human_mask], lib_mode[human_mask])
        results["lightgbm_on_human_labels"] = hs
        results["library_mode_on_human_labels"] = ls
        print(f"\nOn {human_mask.sum()} human-checked test mentions:")
        print(f"  lightgbm:      {hs}")
        print(f"  library_mode:  {ls}")
    else:
        print(f"\nOnly {human_mask.sum()} human-checked mentions in the test set "
              "(need 30+ for a reliable human-label comparison).")

    # Where the model is still wrong (worth reviewing)
    test_df = df.iloc[test_idx].assign(prob=prob_te)
    wrong = test_df[(test_df.prob >= threshold).astype(int) != test_df.label]
    if len(wrong):
        print("\nSkills with the most test mistakes (LightGBM):")
        print(wrong.skill.value_counts().head(10).to_string())

    importance = sorted(zip(FEATURE_NAMES, model.booster_.feature_importance("gain")),
                        key=lambda t: -t[1])
    print("\nTop features:")
    for name, gain in importance[:10]:
        print(f"  {name:<22}{gain:>12.1f}")

    # Plain-language verdict: is the model worth shipping instead of library mode?
    gain_f1 = results["lightgbm"]["f1"] - results["library_mode (backend now)"]["f1"]
    print(f"\nVerdict: LightGBM F1 is {gain_f1:+.4f} vs library_mode (backend now).")
    if gain_f1 >= 0.01:
        print("  The model is clearly better: consider exporting WITH the model (option 8, answer y).")
    else:
        print("  Not clearly better: keep exporting library-only (option 8, answer n).")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(config.MODELS_DIR / "model.txt"))
    meta = {
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "library_version": library_version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_mentions": int(len(train_idx)),
        "test_mentions": int(len(test_idx)),
        "metrics": results,
        "feature_importance": {k: round(float(v), 1) for k, v in importance},
    }
    (config.MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved model + meta to {config.MODELS_DIR} (threshold {threshold})")


if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log = LOG_DIR / f"train_{stamp}.txt"
    tee = _Tee(run_log, LATEST_LOG)
    sys.stdout = tee
    try:
        print(f"Training run: {stamp}\n")
        main()
        print(f"\nLog saved: {run_log}  (latest copy: {LATEST_LOG})")
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"\nStopped: {e.code}")
        raise
    finally:
        sys.stdout = tee.stdout
        tee.close()
"""
Leave-one-out ablation study for the ConflictRAG policy layer (see PLAN.md).

For each variant:
  1. Drop the feature column(s) from the dev features.
  2. Retrain the One-vs-Rest Logistic Regression on the reduced dev features.
  3. Re-tune theta on OUT-OF-FOLD dev probabilities (5-fold stratified CV, same theta grid
     as joint_grid_search.py), so theta is never selected on predictions from the data the
     model was fitted on.
  4. Drop the same column(s) from the test features.
  5. Evaluate once on the held-out test set.
  6. Report macro F1, per-class F1 and delta F1 against the reference with paired-bootstrap
     95% CIs, plus accuracy broken down by source dataset and passage count k.

The test set is used ONLY in step 5. No selection of any kind is made on test data.
tau_s and the TF-IDF threshold are the locked values in framework/config.py.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework import config
from framework.feature_cache import load_raw_features
from framework.feature_extractor import FEATURE_NAMES, FeatureExtractor, features_from_raw
from framework.logger import log_run
from framework.policy import gated_predict, make_policy_model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DEV_PATH = os.path.join(ROOT_DIR, "data", "dev_set_natural.jsonl")
TEST_PATH = os.path.join(ROOT_DIR, "data", "test_set_1.jsonl")
DEV_RAW_CACHE = os.path.join(ROOT_DIR, "data", "cache", "raw_dev_set_natural.json")
TEST_RAW_CACHE = os.path.join(ROOT_DIR, "data", "cache", "raw_test_set_1.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "results", "ablation_results.json")

LABELS = ["ABSTAIN", "ANSWER", "DISCLAIMER", "DISCLOSE"]
THETA_VALUES = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]  # same grid as joint_grid_search.py
N_FOLDS = 5
N_BOOTSTRAP = 1000

# Indices: 0=dispersion, 1=nli_contradiction, 2=claim_support, 3=mismatch, 4=k_norm, 5=query_len_norm
REFERENCE = "Full Model (theta re-tuned)"
VARIANTS = [
    {"name": "Full Model (locked theta)", "drop": [], "theta": "locked"},
    {"name": REFERENCE, "drop": [], "theta": "tune"},
    {"name": "w/o Dispersion", "drop": [0], "theta": "tune"},
    {"name": "w/o NLI Contradiction", "drop": [1], "theta": "tune"},
    {"name": "w/o Claim Support", "drop": [2], "theta": "tune"},
    {"name": "w/o Mismatch", "drop": [3], "theta": "tune"},
    {"name": "w/o Passage Count k", "drop": [4], "theta": "tune"},
    {"name": "w/o Query Length", "drop": [5], "theta": "tune"},
    # Shortcut controls: k and query length are strongly tied to the source dataset.
    {"name": "Retrieval signals only (w/o k, len)", "drop": [4, 5], "theta": "tune"},
    {"name": "k + len only (shortcut control)", "drop": [0, 1, 2, 3], "theta": "tune"},
    {"name": "w/o Theta Gate (argmax)", "drop": [], "theta": "argmax"},
]


# ==============================================================================
# METRICS
# ==============================================================================
def per_class_f1(y_idx, p_idx):
    f1s = np.zeros(len(LABELS))
    for c in range(len(LABELS)):
        tp = np.sum((y_idx == c) & (p_idx == c))
        fp = np.sum((y_idx != c) & (p_idx == c))
        fn = np.sum((y_idx == c) & (p_idx != c))
        denom = 2 * tp + fp + fn
        f1s[c] = 0.0 if denom == 0 else 2 * tp / denom
    return f1s


def macro_f1(y_idx, p_idx):
    # Macro over all 4 labels (fixed), so a class that is never predicted counts as 0.
    return float(np.mean(per_class_f1(y_idx, p_idx)))


def encode(labels):
    lookup = {lab: i for i, lab in enumerate(LABELS)}
    return np.array([lookup[str(l)] for l in labels])


def paired_bootstrap(y_idx, p_var, p_ref, rng):
    """Paired bootstrap on the test set for delta = F1(variant) - F1(reference)."""
    n = len(y_idx)
    deltas = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        s = rng.integers(0, n, n)
        deltas[b] = macro_f1(y_idx[s], p_var[s]) - macro_f1(y_idx[s], p_ref[s])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p-value for H0: delta = 0
    p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return float(lo), float(hi), float(min(p, 1.0))


# ==============================================================================
# THETA TUNING (dev only)
# ==============================================================================
def tune_theta_oof(X_dev, y_dev):
    """Selects theta on out-of-fold dev probabilities. Returns (theta, mean fold macro-F1)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    folds = list(skf.split(X_dev, y_dev))
    oof = np.zeros((len(y_dev), len(LABELS)))
    for tr, va in folds:
        m = make_policy_model().fit(X_dev[tr], y_dev[tr])
        assert list(m.classes_) == LABELS
        oof[va] = m.predict_proba(X_dev[va])

    y_idx = encode(y_dev)
    best_theta, best_f1 = None, -1.0
    for theta in THETA_VALUES:
        pred_idx = encode(gated_predict(oof, LABELS, theta))
        f1 = float(np.mean([macro_f1(y_idx[va], pred_idx[va]) for _, va in folds]))
        if f1 > best_f1:  # strict '>' keeps the smallest theta on ties (deterministic)
            best_theta, best_f1 = theta, f1
    return best_theta, best_f1


def oof_argmax_f1(X_dev, y_dev):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    y_idx = encode(y_dev)
    scores = []
    for tr, va in skf.split(X_dev, y_dev):
        m = make_policy_model().fit(X_dev[tr], y_dev[tr])
        scores.append(macro_f1(y_idx[va], encode(gated_predict(m.predict_proba(X_dev[va]), m.classes_, None))))
    return float(np.mean(scores))


# ==============================================================================
# BREAKDOWNS
# ==============================================================================
def source_of(record_id):
    return re.split(r"[-_]", str(record_id))[0]


def breakdown(groups, y_idx, p_idx):
    out = {}
    for g in sorted(groups):
        idx = groups[g]
        out[g] = {"n": int(len(idx)), "accuracy": float(np.mean(y_idx[idx] == p_idx[idx]))}
    return out


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    dev_records, dev_raw = load_raw_features(DEV_PATH, DEV_RAW_CACHE, FeatureExtractor)
    test_records, test_raw = load_raw_features(TEST_PATH, TEST_RAW_CACHE, FeatureExtractor)

    # Locked tau_s / TF-IDF threshold from config (never tuned here)
    X_dev = np.array([features_from_raw(r) for r in dev_raw])
    X_test = np.array([features_from_raw(r) for r in test_raw])
    y_dev = np.array([r["gold_action"] for r in dev_records])
    y_test = np.array([r["gold_action"] for r in test_records])
    y_test_idx = encode(y_test)

    by_source, by_k = defaultdict(list), defaultdict(list)
    for i, r in enumerate(test_records):
        by_source[source_of(r["id"])].append(i)
        by_k[f"k={len(r['passages'])}"].append(i)
    by_source = {k: np.array(v) for k, v in by_source.items()}
    by_k = {k: np.array(v) for k, v in by_k.items()}

    print(f"\nDev: {len(y_dev)} | Test: {len(y_test)} | tau_s={config.TAU_S} "
          f"tfidf={config.TFIDF_THRESH} C={config.C} locked theta={config.THETA}")
    print("=" * 100)

    results, preds = [], {}
    for v in VARIANTS:
        keep = [i for i in range(X_dev.shape[1]) if i not in v["drop"]]
        Xd, Xt = X_dev[:, keep], X_test[:, keep]

        if v["theta"] == "locked":
            theta = config.THETA
            dev_f1 = None
        elif v["theta"] == "argmax":
            theta = None
            dev_f1 = oof_argmax_f1(Xd, y_dev)
        else:
            theta, dev_f1 = tune_theta_oof(Xd, y_dev)

        model = make_policy_model().fit(Xd, y_dev)  # final fit on the full dev set
        pred = gated_predict(model.predict_proba(Xt), model.classes_, theta)
        p_idx = encode(pred)
        preds[v["name"]] = p_idx

        cls_f1 = per_class_f1(y_test_idx, p_idx)
        res = {
            "name": v["name"],
            "features": [FEATURE_NAMES[i] for i in keep],
            "theta": theta if theta is not None else "argmax",
            "theta_mode": v["theta"],
            "dev_oof_macro_f1": dev_f1,
            "test_accuracy": float(np.mean(y_test_idx == p_idx)),
            "test_macro_f1": macro_f1(y_test_idx, p_idx),
            "test_class_f1": {lab: float(cls_f1[i]) for i, lab in enumerate(LABELS)},
            "by_source": breakdown(by_source, y_test_idx, p_idx),
            "by_k": breakdown(by_k, y_test_idx, p_idx),
        }
        results.append(res)
        print(f"{v['name']:<38} theta={str(res['theta']):<7} test macro-F1={res['test_macro_f1']:.4f}")

    # Deltas vs. the reference (same theta-tuning procedure as the ablated variants)
    rng = np.random.default_rng(config.RANDOM_STATE)
    ref = next(r for r in results if r["name"] == REFERENCE)
    for r in results:
        if r["name"] == REFERENCE:
            r["delta_f1"], r["delta_ci95"], r["delta_p"] = 0.0, [0.0, 0.0], None
            continue
        lo, hi, p = paired_bootstrap(y_test_idx, preds[r["name"]], preds[REFERENCE], rng)
        r["delta_f1"] = r["test_macro_f1"] - ref["test_macro_f1"]
        r["delta_ci95"], r["delta_p"] = [lo, hi], p

    # ---- Table ----
    print("\n" + "=" * 128)
    print(f"ABLATION RESULTS (test set, n={len(y_test)}); delta vs '{REFERENCE}', "
          f"paired bootstrap B={N_BOOTSTRAP}")
    print("=" * 128)
    hdr = (f"{'Variant':<38}|{'theta':>7} |{'Macro F1':>9} |{'dF1':>8} |{'95% CI':>18} |{'p':>6} |"
           f"{'ABSTAIN':>8} |{'ANSWER':>8} |{'DISCLM':>8} |{'DISCLOSE':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        cf = r["test_class_f1"]
        ci = f"[{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]"
        p = "-" if r["delta_p"] is None else f"{r['delta_p']:.3f}"
        print(f"{r['name']:<38}|{str(r['theta']):>7} |{r['test_macro_f1']:>9.4f} |{r['delta_f1']:>+8.4f} |"
              f"{ci:>18} |{p:>6} |{cf['ABSTAIN']:>8.4f} |{cf['ANSWER']:>8.4f} |"
              f"{cf['DISCLAIMER']:>8.4f} |{cf['DISCLOSE']:>9.4f}")
    print("=" * 128)

    print("\nAccuracy by source / passage count (test):")
    groups = list(results[0]["by_source"]) + list(results[0]["by_k"])
    print(f"{'Variant':<38}|" + "|".join(f"{g:>11}" for g in groups))
    for r in results:
        cells = [r["by_source"][g]["accuracy"] if g in r["by_source"] else r["by_k"][g]["accuracy"] for g in groups]
        print(f"{r['name']:<38}|" + "|".join(f"{c:>11.3f}" for c in cells))

    # ---- Persist ----
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "tau_s": config.TAU_S, "tfidf": config.TFIDF_THRESH, "C": config.C,
                "locked_theta": config.THETA, "theta_grid": THETA_VALUES, "n_folds": N_FOLDS,
                "n_bootstrap": N_BOOTSTRAP, "random_state": config.RANDOM_STATE,
                "reference": REFERENCE, "dev": os.path.basename(DEV_PATH),
                "test": os.path.basename(TEST_PATH),
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    for r in results:
        log_run(
            run_name=f"Ablation: {r['name']}",
            data_path=TEST_PATH,
            parameters={"features": r["features"], "theta": r["theta"], "tau_s": config.TAU_S,
                        "tfidf": config.TFIDF_THRESH, "C": config.C},
            metrics={"test_macro_f1": r["test_macro_f1"], "delta_f1": r["delta_f1"],
                     "delta_ci95": r["delta_ci95"], "delta_p": r["delta_p"],
                     "dev_oof_macro_f1": r["dev_oof_macro_f1"], "test_class_f1": r["test_class_f1"]},
        )


if __name__ == "__main__":
    main()

"""
Joint grid search over (tau_s, TF-IDF threshold, C, theta) with 5-fold stratified CV on
dev_set_natural, for the 4D policy [D, C-hat, S, L]. Uses the SAME FeatureExtractor.extract_raw()
as test-time inference, so dev tuning and test evaluation compute identical features.

The ranges are wider than in the 6D run, where the optimum sat on the grid edge
(tau_s=0.2 was the minimum; C=10 was the maximum). The script warns if the 4D optimum is
again on an edge.

All combinations are saved to results/<RUN_TAG>_grid_search.json for sensitivity analysis.
config.py is NOT updated automatically: adopting new hyperparameters is a methodology decision.
"""
import json
import os
import sys
from itertools import product

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework import config
from framework.feature_cache import load_raw_features
from framework.feature_extractor import FEATURE_NAMES, FeatureExtractor, features_from_raw
from framework.logger import log_run
from framework.policy import gated_predict, make_policy_model

DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_set_natural.jsonl")
RAW_CACHE = os.path.join(ROOT_DIR, "data", "cache", "raw_dev_set_natural.json")
GRID_RESULTS_PATH = os.path.join(ROOT_DIR, "results", f"{config.RUN_TAG}_grid_search.json")

TAU_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TFIDF_VALUES = [0.02, 0.04, 0.06, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
THETA_VALUES = config.THETA_GRID

GRIDS = {"tau_s": TAU_VALUES, "tfidf": TFIDF_VALUES, "C": C_VALUES, "theta": THETA_VALUES}


def main():
    records, raw = load_raw_features(DATA_PATH, RAW_CACHE, FeatureExtractor)
    y = np.array([r["gold_action"] for r in records])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(y)), y))
    print(f"Features ({len(FEATURE_NAMES)}D): {FEATURE_NAMES}")

    all_runs = []
    best_f1, best_params = -1.0, {}
    pbar = tqdm(total=len(TAU_VALUES) * len(TFIDF_VALUES) * len(C_VALUES), desc="Grid Search")
    for tau, tfidf in product(TAU_VALUES, TFIDF_VALUES):
        X = np.array([features_from_raw(r, tau_s=tau, tfidf_thresh=tfidf) for r in raw])
        for c in C_VALUES:
            # Out-of-fold probabilities once per (tau, tfidf, C); theta is applied afterwards.
            oof = np.zeros((len(y), 4))
            classes = None
            for tr, va in folds:
                model = make_policy_model(C=c).fit(X[tr], y[tr])
                classes = list(model.classes_)
                oof[va] = model.predict_proba(X[va])
            for theta in THETA_VALUES:
                fold_f1s = [
                    f1_score(y[va], gated_predict(oof[va], classes, theta), average="macro", zero_division=0)
                    for _, va in folds
                ]
                mean_f1 = float(np.mean(fold_f1s))
                all_runs.append({"tau_s": tau, "tfidf": tfidf, "C": c, "theta": theta,
                                 "mean_f1": mean_f1, "std_f1": float(np.std(fold_f1s))})
                if mean_f1 > best_f1:  # strict '>' keeps the first (smallest) values on ties
                    best_f1 = mean_f1
                    best_params = {"tau_s": tau, "tfidf": tfidf, "C": c, "theta": theta}
            pbar.update(1)
            pbar.set_postfix({"best": f"{best_f1:.4f}"})
    pbar.close()

    edges = [k for k, grid in GRIDS.items() if best_params[k] in (min(grid), max(grid))]

    print("\n" + "=" * 50)
    print(f"JOINT GRID SEARCH COMPLETE ({config.RUN_TAG})")
    print("=" * 50)
    print(f"Best mean 5-fold Macro F1: {best_f1:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    top = sorted(all_runs, key=lambda r: -r["mean_f1"])[:10]
    print("\nTop 10 configurations (mean ± std over folds):")
    for r in top:
        print(f"  tau={r['tau_s']:<5} tfidf={r['tfidf']:<5} C={r['C']:<7} theta={r['theta']:<5} "
              f"F1={r['mean_f1']:.4f} ± {r['std_f1']:.4f}")
    if edges:
        print(f"\nWARNING: optimum is on the grid edge for {edges}. Extend those ranges before adopting.")
    print("NOTE: config.py values are NOT updated automatically. Changing locked hyperparameters "
          "is a methodology decision.")

    os.makedirs(os.path.dirname(GRID_RESULTS_PATH), exist_ok=True)
    with open(GRID_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"run_tag": config.RUN_TAG, "features": FEATURE_NAMES, "grids": GRIDS,
                   "best": best_params, "best_mean_f1": best_f1, "edge_params": edges,
                   "runs": all_runs}, f, indent=2)
    print(f"All {len(all_runs)} configurations saved to {GRID_RESULTS_PATH}")

    log_run(
        run_name=f"Joint Grid Search ({config.RUN_TAG}, 4D, expanded grid)",
        data_path=DATA_PATH,
        parameters=best_params,
        metrics={"macro_f1": best_f1, "edge_params": edges},
    )


if __name__ == "__main__":
    main()
